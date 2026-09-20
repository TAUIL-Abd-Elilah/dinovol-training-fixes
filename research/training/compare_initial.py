"""Compare first updates at identical seeded initial states, regardless of timing role."""
import argparse
import json
from pathlib import Path

import numpy as np
from compare_training import metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reference',type=Path);ap.add_argument('candidate',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    paths=(args.reference,args.candidate)
    reports=[json.loads((p/'report.json').read_text()) for p in paths]
    if any(r['status']!='TRAINING_PROBE_COMPLETE' for r in reports):
        raise ValueError('Require successful complete runs')
    assert reports[0]['input']==reports[1]['input']
    assert reports[0]['checkpoint']==reports[1]['checkpoint']
    assert reports[0]['protocol']['seed']==reports[1]['protocol']['seed']
    steps=[r['steps'][0] for r in reports]
    assert all(s['number']==0 for s in steps)
    assert steps[0]['schedule_step']==steps[1]['schedule_step']
    configs=[]
    for r in reports:
        c={k:v for k,v in r['config'].items() if k!='output_dir'}
        c['model']={k:v for k,v in c['model'].items() if k!='pad_sdpa_heads'}
        configs.append(c)
    assert configs[0]==configs[1]
    result={'reference':str(args.reference),'candidate':str(args.candidate),
            'scope':'First-update numerical screen, not convergence or a repeated speed benchmark.',
            'source_hashes_equal':reports[0]['source_hashes']==reports[1]['source_hashes'],
            'sample_comparisons':{},'losses':[s['metrics'] for s in steps]}
    for key in ('gradient','student_before','student_after','student_delta','teacher_before','teacher_after','teacher_delta'):
        arrays=[np.load(p/f"{s['kind']}_00_{key}_samples.npy",allow_pickle=False) for p,s in zip(paths,steps)]
        result['sample_comparisons'][key]=metrics(*arrays)
    assert result['sample_comparisons']['student_before']['exact']
    assert result['sample_comparisons']['teacher_before']['exact']
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['sample_comparisons'],indent=2))


if __name__=='__main__':
    main()
