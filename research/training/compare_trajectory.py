"""Compare every recorded step of the integrated-source trajectory experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_training import metrics


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(native, candidate):
    reports = [json.loads((p/'report.json').read_text()) for p in (native,candidate)]
    a,b = reports
    assert a['mode']=='native' and b['mode']=='padded'
    assert all(r['status']=='TRAINING_PROBE_COMPLETE' for r in reports)
    for key in ('source_hashes','source_validation','checkpoint','protocol','input',
                'script_sha256','helper_sha256','audit_protocol','varying_input_protocol'):
        assert a[key]==b[key], 'Pair mismatch: '+key
    for r in reports:
        assert not r['checkpoint_compatibility_adapter']['runtime_adapter_applied']
        assert r['attention_helper_kind']=='integrated_source_guarded'
    configs = [json.loads(json.dumps(r['config'])) for r in reports]
    assert configs[0]['model'].pop('pad_sdpa_heads') is False
    assert configs[1]['model'].pop('pad_sdpa_heads') is True
    for cfg in configs:
        cfg.pop('output_dir')
    assert configs[0]==configs[1], 'Unexpected configuration difference'
    layout = [json.loads((p/'parameter_sample_layout.json').read_text()) for p in (native,candidate)]
    assert layout[0]==layout[1]
    assert len(a['steps'])==len(b['steps'])==100
    result = dict(scope='100 varying-crop/mask CT updates per arm; sampled numerical drift, not convergence or ink accuracy',
                  complete_updates_per_arm=100, identical_source_config_except_attention=True,
                  report_sha256={'native':sha(native/'report.json'),'candidate':sha(candidate/'report.json')},
                  driver_sha256=a['script_sha256'], steps=[], full_audit_updates=[], sample_file_hashes=[])
    labels = ('gradient','student_before','student_after','student_delta','teacher_before','teacher_after','teacher_delta')
    for sa,sb in zip(a['steps'],b['steps']):
        for key in ('number','kind','schedule_step','audit_scope','step_input'):
            assert sa[key]==sb[key], 'Step mismatch: '+key
        assert sa['step_input'] is not None
        assert all(sa['checks'].values()) and all(sb['checks'].values())
        if sa['audit_scope']=='full':
            result['full_audit_updates'].append(sa['number'])
        step = dict(number=sa['number'],kind=sa['kind'],audit_scope=sa['audit_scope'],
                    both_all_checks_pass=True,input_hashes_equal=True,
                    native_ms=sa['elapsed_ms'],candidate_ms=sb['elapsed_ms'],
                    native_scaler=sa['scaler_after'],candidate_scaler=sb['scaler_after'],
                    native_peak_allocated=sa['gpu_memory']['peak_allocated_bytes'],
                    candidate_peak_allocated=sb['gpu_memory']['peak_allocated_bytes'],loss={},samples={})
        for k in ('loss','dino_global_loss','dino_local_loss','ibot_loss','koleo_loss'):
            x,y=sa['metrics'][k],sb['metrics'][k]
            step['loss'][k]=dict(native=x,candidate=y,absolute_delta=y-x,
                                 relative_delta=(y-x)/max(abs(x),1e-30))
        for label in labels:
            rel = f"{sa['kind']}_{sa['number']:02d}_{label}_samples.npy"
            paths = [p/rel for p in (native,candidate)]
            values = [np.load(p,allow_pickle=False) for p in paths]
            assert all(v.shape==(sum(item['sample_count'] for item in layout[0]),) for v in values)
            step['samples'][label]=metrics(*values)
            result['sample_file_hashes'].append(dict(file=rel,native=sha(paths[0]),candidate=sha(paths[1]),
                availability='local raw receipts; hashes for regeneration, most arrays not bundled'))
        if sa['number']==0:
            assert step['samples']['student_before']['exact'] and step['samples']['teacher_before']['exact']
        result['steps'].append(step)
    assert result['full_audit_updates']==[0,24,49,74,99]
    timed = [s for s in result['steps'] if s['kind']=='timed']
    n,c = [float(np.median([s[k] for s in timed])) for k in ('native_ms','candidate_ms')]
    nm,cm = [max(s[k] for s in timed) for k in ('native_peak_allocated','candidate_peak_allocated')]
    result['performance']=dict(native_median_ms=n,candidate_median_ms=c,speedup=n/c,
        native_peak_allocated_bytes=nm,candidate_peak_allocated_bytes=cm,allocated_reduction_fraction=1-cm/nm)
    result['extrema']={}
    for label in labels:
        worst=max(result['steps'],key=lambda s:s['samples'][label]['relative_l2'])
        result['extrema'][label]=dict(worst_update=worst['number'],**worst['samples'][label])
    loss_worst=max(result['steps'],key=lambda s:abs(s['loss']['loss']['absolute_delta']))
    result['extrema']['total_loss']=dict(worst_update=loss_worst['number'],**loss_worst['loss']['loss'])
    result['initial_sampled_states_identical']=True
    result['all_100_batch_hashes_match']=True
    result['all_100_optimizer_and_finiteness_checks_pass']=True
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('native',type=Path);ap.add_argument('candidate',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=compare(args.native,args.candidate)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('steps','sample_file_hashes')},indent=2))
