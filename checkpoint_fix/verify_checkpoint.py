"""Real pretrained backbone: checkpoint-off vs explicit non-reentrant checkpoint.

Uses two real 64-cubed PHerc0139 views, train-mode stochastic depth/RoPE, and a
fixed differentiable embedding objective. This is gradient regression evidence,
not DINO training quality or an ink-accuracy experiment.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np
import torch


def file_sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dinovol-root',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--volume',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.85)
    sys.path.insert(0,str(args.dinovol_root.resolve()))
    from dinovol_2.model.model import DinoVitStudentTeacher
    from dinovol_2.dataset.normalization import get_normalization
    from dinovol_2.model import dinov2_eva as eva
    volume=np.load(args.volume,allow_pickle=False)
    assert hashlib.sha256(volume.tobytes()).hexdigest()=='fb5d3659016e1bb760c608336e3ae1f356414464362021ec4e96ce392de94ad5'
    assert file_sha(args.checkpoint)=='e041ca870dd2570f8a44d1dd26db1197b3f74121f62023bc774fbc9d40e51a59'
    normalizer=get_normalization('robust')
    views=[normalizer.run(volume[s:s+64,s:s+64,s:s+64].astype(np.float32,copy=True)) for s in (32,160)]
    x=torch.from_numpy(np.stack(views)[:,None]).cuda()
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=dict(payload['config']['model'])
    model=DinoVitStudentTeacher._build_backbone(config)
    state={k.removeprefix('backbone.'):v for k,v in payload['teacher'].items()}
    model.load_state_dict(state,strict=True)
    del payload,state
    model=model.cuda().train()
    native_checkpoint=eva.checkpoint
    results=[]
    reference=None
    for enabled in (False,True):
        model.zero_grad(set_to_none=True)
        model.grad_checkpointing=enabled
        torch.manual_seed(62920);random.seed(62920);np.random.seed(62920)
        with patch.object(eva,'checkpoint',partial(native_checkpoint,use_reentrant=False)):
            with torch.autocast('cuda'):
                outputs=model(x,view_kind='local')
                cls=outputs['x_norm_clstoken'].float()
                tokens=outputs['x_norm_patchtokens'].float()
                # Nonconstant, asymmetric objective exercises both token paths.
                weights=torch.linspace(-1,1,cls.shape[-1],device=x.device)
                loss=(cls*weights).mean()+(tokens[...,::7].square()).mean()
            loss.backward()
        grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        values={k:outputs[k].detach().cpu() for k in ('x_norm_clstoken','x_norm_patchtokens')}
        entry={'checkpoint_enabled':enabled,'loss':float(loss.detach()),
               'gradient_tensors':len(grads),'gradient_elements':sum(g.numel() for g in grads.values()),
               'finite_gradients':all(bool(torch.isfinite(g).all()) for g in grads.values())}
        if reference is None:
            reference=(grads,values)
        else:
            old_grads,old_values=reference
            assert grads.keys()==old_grads.keys()
            rows=[]
            error_sq=ref_sq=0.0
            for n,g in grads.items():
                a,b=old_grads[n].double(),g.double()
                d=b-a
                e=float(d.square().sum());r=float(a.square().sum())
                error_sq+=e;ref_sq+=r
                rows.append({'name':n,'exact':bool(torch.equal(a,b)),
                             'max_abs':float(d.abs().max()),'relative_l2':(e/max(r,1e-30))**.5})
            entry['gradients']={'all_exact':all(r['exact'] for r in rows),
                                'relative_l2':(error_sq/max(ref_sq,1e-30))**.5,
                                'per_tensor':rows}
            entry['outputs']={k:{'exact':bool(torch.equal(v,old_values[k])),
                                 'max_abs':float((v-old_values[k]).abs().max())} for k,v in values.items()}
        results.append(entry)
        del outputs,grads,values,loss
    report={'scope':__doc__,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'checkpoint_sha256':file_sha(args.checkpoint),
            'eva_sha256':file_sha(args.dinovol_root/'dinovol_2/model/dinov2_eva.py'),
            'input_sha256':hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
            'source_volume':'PHerc0139 20250728140407-9.362um-1.2m-113keV-masked.zarr',
            'source_cube_l0_zyx':[3840,4096,3712,3968,1344,1600],
            'local_cube_starts_zyx':[[32]*3,[160]*3], 'view_size':[64]*3,
            'use_reentrant':False,'results':results}
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    last=results[-1]
    print(json.dumps({'losses':[r['loss'] for r in results],
                      'finite_gradients':last['finite_gradients'],
                      'gradient_tensors':last['gradient_tensors'],
                      'gradient_elements':last['gradient_elements'],
                      'all_gradient_tensors_exact':last['gradients']['all_exact'],
                      'gradient_relative_l2':last['gradients']['relative_l2'],
                      'outputs':last['outputs']},indent=2))
    assert last['finite_gradients'] and last['gradients']['relative_l2'] < 1e-5
    assert all(r['exact'] for r in last['outputs'].values())


if __name__=='__main__':
    main()
