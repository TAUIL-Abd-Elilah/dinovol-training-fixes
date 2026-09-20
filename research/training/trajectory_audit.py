"""Lightweight intervening-update audit; full gradient finiteness is retained."""
import numpy as np
import torch


def indices(p):
    count = min(128, p.numel())
    return torch.arange(count, device=p.device, dtype=torch.int64) * (p.numel()-1) // max(count-1, 1)


def sample(p):
    return p.detach().flatten()[indices(p)].float().cpu().clone()


def snapshot(branch):
    return {name: sample(p) for name, p in branch.named_parameters()}


def audit(student, teacher, before_student, before_teacher, momentum):
    totals = dict(student_delta_l2_sq=0., teacher_delta_l2_sq=0., gradient_l2_sq=0.,
                  student_changed_parameters=0, teacher_changed_parameters=0,
                  gradient_parameters=0, nonfinite_gradient_parameters=0,
                  teacher_gradient_parameters=0, ema_max_abs_error=0., ema_allclose=True,
                  value_scope='128 equally spaced values per parameter; full gradient finiteness')
    arrays = {k: [] for k in ('gradient','student_before','student_after','student_delta',
                              'teacher_before','teacher_after','teacher_delta')}
    layout = []
    teachers = dict(teacher.named_parameters())
    for name, p in student.named_parameters():
        after, before = sample(p), before_student[name]
        delta = after - before
        t = teachers[name]
        teacher_after, teacher_before = sample(t), before_teacher[name]
        teacher_delta = teacher_after - teacher_before
        grad = sample(p.grad) if p.grad is not None else torch.zeros_like(after)
        if p.grad is not None:
            totals['gradient_parameters'] += 1
            totals['nonfinite_gradient_parameters'] += int(not torch.isfinite(p.grad).all().item())
            totals['gradient_l2_sq'] += float(grad.double().square().sum())
        totals['teacher_gradient_parameters'] += int(t.grad is not None)
        for prefix, d in (('student', delta), ('teacher', teacher_delta)):
            sq = float(d.double().square().sum())
            totals[prefix+'_delta_l2_sq'] += sq
            totals[prefix+'_changed_parameters'] += int(sq > 0)
        expected = teacher_before * momentum + after * (1-momentum)
        totals['ema_max_abs_error'] = max(totals['ema_max_abs_error'], float((teacher_after-expected).abs().max()))
        totals['ema_allclose'] &= bool(torch.allclose(teacher_after, expected, rtol=5e-6, atol=5e-7))
        for key, val in dict(gradient=grad, student_before=before, student_after=after,
                             student_delta=delta, teacher_before=teacher_before,
                             teacher_after=teacher_after, teacher_delta=teacher_delta).items():
            arrays[key].append(val.numpy())
        idx = indices(p).cpu()
        layout.append(dict(name=name, parameter_numel=p.numel(), sample_count=len(idx),
                           sample_indices=idx.tolist(), has_gradient=p.grad is not None))
    return totals, {k: np.concatenate(v) for k,v in arrays.items()}, layout


def self_test():
    # A nonfinite gradient outside the sampled positions must still fail.
    from copy import deepcopy
    s = torch.nn.Linear(257, 2, bias=False)
    t = deepcopy(s)
    bs, bt = snapshot(s), snapshot(t)
    with torch.no_grad():
        s.weight.add_(0.1)
        t.weight.mul_(0.9).add_(s.weight, alpha=0.1)
    s.weight.grad = torch.ones_like(s.weight)
    totals, _, _ = audit(s,t,bs,bt,0.9)
    assert totals['ema_allclose'] and totals['student_changed_parameters']==1
    unsampled = next(i for i in range(s.weight.numel()) if i not in indices(s.weight).tolist())
    s.weight.grad.flatten()[unsampled] = float('inf')
    assert audit(s,t,bs,bt,0.9)[0]['nonfinite_gradient_parameters']==1
    print('PASS: sampled EMA/update audit and full-gradient nonfinite detection')


if __name__ == '__main__':
    self_test()
