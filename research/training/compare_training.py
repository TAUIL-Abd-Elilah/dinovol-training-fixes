"""Compare paired official training probe receipts without hiding differences."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def metrics(a, b):
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    diff = y-x
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return {"count": x.size, "finite": bool(np.isfinite(y).all()),
            "exact": bool(np.array_equal(x,y)),
            "max_abs": float(np.abs(diff).max(initial=0)),
            "mean_abs": float(np.abs(diff).mean()),
            "relative_l2": float(np.linalg.norm(diff)/max(nx,1e-30)),
            "cosine": float(np.dot(x,y)/(nx*ny)) if nx*ny else None}


def compare(native_dir, candidate_dir):
    reports = [json.loads((p/'report.json').read_text()) for p in (native_dir,candidate_dir)]
    a,b = reports
    if a['status'] != 'TRAINING_PROBE_COMPLETE' or b['status'] != a['status']:
        raise ValueError('Require two fully successful receipts; do not compare failed runs')
    for key in ('source_hashes','checkpoint','protocol','input'):
        if a[key] != b[key]:
            raise ValueError(f'Paired experiment mismatch: {key}')
    configs = [{k:v for k,v in r['config'].items() if k!='output_dir'} for r in reports]
    if configs[0] != configs[1]:
        raise ValueError('Paired configuration mismatch')
    la,lb = [json.loads((p/'parameter_sample_layout.json').read_text()) for p in (native_dir,candidate_dir)]
    if la != lb:
        raise ValueError('Sample layout mismatch')
    if len(a['steps']) != len(b['steps']):
        raise ValueError('Step counts differ')
    result = {'native':str(native_dir),'candidate':str(candidate_dir),
              'scope':'Short deterministic training-step experiment; no convergence, downstream quality, or reading claim.',
              'report_sha256':{str(p):hashlib.sha256((p/'report.json').read_bytes()).hexdigest()
                               for p in (native_dir,candidate_dir)},
              'steps':[]}
    for sa,sb in zip(a['steps'],b['steps']):
        for k in ('number','kind','schedule_step'):
            if sa[k]!=sb[k]:
                raise ValueError('Unaligned steps')
        step = {'number':sa['number'],'kind':sa['kind'],
                'native_checks':sa['checks'],'candidate_checks':sb['checks'],
                'loss': {k:{'native':sa['metrics'][k], 'candidate':sb['metrics'][k],
                           'absolute_delta':sb['metrics'][k]-sa['metrics'][k],
                           'relative_delta':(sb['metrics'][k]-sa['metrics'][k])/max(abs(sa['metrics'][k]),1e-30)}
                        for k in ('loss','dino_global_loss','dino_local_loss','ibot_loss','koleo_loss')},
                'samples':{}}
        for label in ('gradient','student_before','student_after','student_delta','teacher_after','teacher_delta'):
            paths = [p/f"{sa['kind']}_{sa['number']:02d}_{label}_samples.npy" for p in (native_dir,candidate_dir)]
            if not all(p.exists() for p in paths):
                continue
            x,y = [np.load(p,allow_pickle=False) for p in paths]
            step['samples'][label] = metrics(x,y)
            if label=='gradient':
                offset=0
                per_tensor=[]
                for item in la:
                    n=item['sample_count']
                    if item['has_gradient']:
                        m=metrics(x[offset:offset+n],y[offset:offset+n])
                        per_tensor.append({'name':item['name'],**m})
                    offset+=n
                step['gradient_per_parameter']=per_tensor
        result['steps'].append(step)
    times = [r['timing_summary']['median_ms'] for r in reports]
    memory = [max(s['gpu_memory']['peak_allocated_bytes'] for s in r['steps'] if s['kind']=='timed') for r in reports]
    result['performance']={'native_median_ms':times[0],'candidate_median_ms':times[1],
                           'speedup':times[0]/times[1], 'native_peak_allocated_bytes':memory[0],
                           'candidate_peak_allocated_bytes':memory[1],
                           'allocated_reduction_fraction':1-memory[1]/memory[0]}
    return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('native',type=Path);ap.add_argument('candidate',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=compare(args.native,args.candidate)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'performance':result['performance'],
                      'steps':[{'number':s['number'],'loss':s['loss']['loss'],
                                'gradient':s['samples'].get('gradient')} for s in result['steps']]},indent=2))
