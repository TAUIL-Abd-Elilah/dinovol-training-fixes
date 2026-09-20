"""Synthetic CPU regression for explicit non-reentrant Dinovol checkpointing.

Loads the pinned official source and an in-memory copy with the three-line fix.
Does not edit a checkout, use CUDA, download data, or claim training quality.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
import traceback
import types
import warnings
from pathlib import Path
from unittest.mock import patch

PIN = 'f07d33be6a00d12ace7d6a9465efe17c78ed7b47'
RELATIVE = 'dinovol_2/model/dinov2_eva.py'
GIT_RELATIVE = 'dinovol/' + RELATIVE
OLD = 'rope_shape=rope_shape, rope_coords=rope_coords)'
RTOL, ATOL = 1e-5, 1e-7


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dinovol-root', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'STARTED', 'scope': __doc__, 'source_pin': PIN,
              'script_sha256': sha(Path(__file__).read_bytes()),
              'comparison_tolerances': {'rtol': RTOL, 'atol': ATOL}, 'cases': []}
    try:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        sys.path.insert(0, str(args.dinovol_root.resolve()))
        import dinovol_2.model.dinov2_eva as original
        import dinovol_2.model.model as model_module
        report['versions'] = {'python': sys.version, 'torch': torch.__version__}
        # Assert the complete tracked source subtree matches the recorded public
        # revision. No patch is applied to this checkout.
        subprocess.run(['git', '-C', str(args.dinovol_root), 'diff', '--exit-code',
                        PIN, '--', 'dinovol_2'], check=True, capture_output=True)
        source_path = args.dinovol_root / RELATIVE
        source = source_path.read_text(encoding='utf-8')
        source_lines = source.splitlines(keepends=True)
        fixed_lines = []
        changed = []
        for number, line in enumerate(source_lines, 1):
            if 'x = checkpoint(' in line:
                if OLD not in line:
                    raise AssertionError('Checkpoint source shape differs from audited pin')
                line = line.replace(OLD, OLD[:-1] + ', use_reentrant=False)')
                changed.append(number)
            fixed_lines.append(line)
        if changed != [778, 921, 927]:
            raise AssertionError(f'Expected exactly the three audited call sites, got {changed}')
        fixed_source = ''.join(fixed_lines)
        patch_text = ''.join(difflib.unified_diff(source_lines, fixed_lines,
                              fromfile='a/' + GIT_RELATIVE, tofile='b/' + GIT_RELATIVE))
        patch_path = args.output / 'explicit_nonreentrant.patch'
        patch_path.write_text(patch_text, encoding='utf-8', newline='\n')
        repo_root = subprocess.check_output(['git', '-C', str(args.dinovol_root),
                                             'rev-parse', '--show-toplevel'], text=True).strip()
        subprocess.run(['git', '-C', repo_root, 'apply', '--check', str(patch_path.resolve())],
                       check=True, capture_output=True)
        report['source'] = {'original_sha256': sha(source_path.read_bytes()),
                            'normalized_original_sha256': sha(source.encode()),
                            'normalized_fixed_sha256': sha(fixed_source.encode()),
                            'patch_sha256': sha(patch_path.read_bytes()),
                            'changed_lines': changed, 'git_apply_check_passed': True,
                            'checkout_edited': False}
        fixed = types.ModuleType('dinovol_2.model.dinov2_eva_checkpoint_fixed')
        fixed.__file__ = '<pinned dinov2_eva.py with explicit use_reentrant=False>'
        sys.modules[fixed.__name__] = fixed
        exec(compile(fixed_source, fixed.__file__, 'exec'), fixed.__dict__)

        def make_backbone(config, implementation):
            with patch.object(model_module, 'Eva', implementation.Eva), \
                    patch.object(model_module, 'EvaWithChunking', implementation.EvaWithChunking):
                return model_module.DinoVitStudentTeacher._build_backbone(config).cpu().train()

        def evaluate(backbone, inputs, masks, seed):
            x = inputs.detach().clone().requires_grad_(True)
            torch.manual_seed(seed)
            backbone.zero_grad(set_to_none=True)
            with sdpa_kernel(SDPBackend.MATH):
                outputs = backbone(x, masks=masks)
                loss = x.new_zeros(())
                for value in outputs.values():
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        weights = torch.linspace(-0.7, 1.3, value.numel(), dtype=value.dtype).reshape(value.shape)
                        loss = loss + (value * weights).mean() + 0.07 * value.square().mean()
                loss.backward()
            return {'outputs': {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k,v in outputs.items()},
                    'input_gradient': x.grad.detach().clone(),
                    'parameter_gradients': {k: None if p.grad is None else p.grad.detach().clone()
                                           for k,p in backbone.named_parameters()},
                    'loss': float(loss.detach()), 'rng_after_backward': sha(torch.get_rng_state().numpy().tobytes())}

        def compare(a, b):
            if a is None or b is None:
                return {'close': a is None and b is None, 'both_none': a is None and b is None}
            if a.dtype == torch.bool:
                return {'close': bool(torch.equal(a,b)), 'exact': bool(torch.equal(a,b))}
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            return {'close': finite and bool(torch.allclose(a,b,rtol=RTOL,atol=ATOL)),
                    'finite': finite, 'exact': bool(torch.equal(a,b)),
                    'max_abs_error': float((a-b).abs().max()) if a.numel() else 0.0,
                    'reference_l2': float(a.double().norm()), 'difference_l2': float((a-b).double().norm())}

        for model_type in ('v1','v2'):
            for block_chunks in (0,2,4):
                label = f'{model_type}_' + {0:'plain',2:'chunked',4:'chunk_class_unchunked'}[block_chunks]
                config = {'model_type': model_type, 'input_channels':1,
                          'global_crops_size':(16,16,16), 'local_crops_size':(8,8,8),
                          'embed_dim':48, 'depth':4, 'num_heads':2, 'patch_size':(4,4,4),
                          'block_chunks':block_chunks, 'num_reg_tokens':4,
                          'drop_path_rate':0.3, 'grad_checkpointing':False,
                          'rope_shift_coords':0.05, 'rope_jitter_coords':1.05, 'rope_rescale_coords':2.0}
                torch.manual_seed(20260920)
                baseline = make_backbone(config, original)
                repaired = make_backbone({**config,'grad_checkpointing':True}, fixed)
                repaired.load_state_dict(baseline.state_dict(), strict=True)
                inputs = torch.linspace(-1.0,1.0,2*16**3).reshape(2,1,16,16,16)
                masks = torch.zeros((2,64), dtype=torch.bool)
                masks[0,::7] = True; masks[1,3::7] = True
                # Capture the original failure before checking the repaired path.
                baseline.grad_checkpointing = True
                caught = None
                with warnings.catch_warnings(record=True) as caught_warnings:
                    try:
                        baseline(inputs, masks=masks)
                    except Exception as exc:
                        caught = {'type':type(exc).__name__, 'message':str(exc)}
                baseline.grad_checkpointing = False
                if caught is None or 'Unexpected keyword arguments' not in caught['message']:
                    raise AssertionError(f'Expected original checkpoint keyword failure: {caught}')
                result = {'case':label, 'config':config, 'original_failure':caught,
                          'original_warnings':[str(w.message) for w in caught_warnings], 'seeds':[]}
                for seed in (871,872):
                    reference = evaluate(baseline,inputs,masks,seed)
                    actual = evaluate(repaired,inputs,masks,seed)
                    if reference['outputs'].keys() != actual['outputs'].keys():
                        raise AssertionError('Output key mismatch')
                    output_checks = {k:compare(v,actual['outputs'][k]) for k,v in reference['outputs'].items()}
                    if reference['parameter_gradients'].keys() != actual['parameter_gradients'].keys():
                        raise AssertionError('Parameter name mismatch')
                    gradient_checks = {k:compare(v,actual['parameter_gradients'][k])
                                       for k,v in reference['parameter_gradients'].items()}
                    input_check = compare(reference['input_gradient'],actual['input_gradient'])
                    checks = {'all_outputs_close':all(v['close'] for v in output_checks.values()),
                              'input_gradient_close':input_check['close'],
                              'all_parameter_gradients_close':all(v['close'] for v in gradient_checks.values()),
                              'rng_after_backward_equal':reference['rng_after_backward']==actual['rng_after_backward'],
                              'input_gradient_nonzero':bool(reference['input_gradient'].norm()>0),
                              'some_parameter_gradients_nonzero':any(g is not None and bool(g.norm()>0)
                                                                     for g in reference['parameter_gradients'].values())}
                    result['seeds'].append({'seed':seed,'checks':checks,'output_checks':output_checks,
                                            'input_gradient_check':input_check,'parameter_gradient_checks':gradient_checks,
                                            'loss_reference':reference['loss'],'loss_fixed':actual['loss']})
                    if not all(checks.values()):
                        report['cases'].append(result)
                        raise AssertionError(f'CPU parity failed for {label} seed {seed}: {checks}')
                report['cases'].append(result)
                print(json.dumps({'case':label,'seeds':2,'status':'PASS'}),flush=True)
        report['cuda_initialized'] = torch.cuda.is_initialized()
        if report['cuda_initialized']:
            raise AssertionError('CPU regression unexpectedly initialized CUDA')
        report['status'] = 'PASS'
    except Exception:
        report['status'] = 'FAILED'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
