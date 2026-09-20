"""CPU-only synthetic state/configuration and guarded-helper regression."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import training_sdpa_padding as helper
from make_training_integration import sources, PREFIX, PIN


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--villa',required=True,type=Path)
    args=ap.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    sys.path.insert(0,str((args.villa/'dinovol').resolve()))
    import dinovol_2.model.model as original_model
    original,updated=sources(args.villa)
    eva=types.ModuleType('dinovol_2.model.training_padding_eva_test')
    model=types.ModuleType('dinovol_2.model.training_padding_model_test')
    with patch.dict(sys.modules,{'dinovol_2.model.training_sdpa_padding':helper}):
        exec(compile(updated[PREFIX+'dinov2_eva.py'],'<patched Eva>','exec'),eva.__dict__)
    with patch.dict(sys.modules,{'dinovol_2.model.dinov2_eva':eva}):
        exec(compile(updated[PREFIX+'model.py'],'<patched model>','exec'),model.__dict__)
    receipt={'source_pin':PIN,'scope':__doc__,'torch':torch.__version__,'cases':[],
             'helper_sha256':hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest()}

    def evaluate(backbone):
        torch.manual_seed(12345)
        x=torch.linspace(-1,1,2*16**3).reshape(2,1,16,16,16).requires_grad_(True)
        mask=torch.zeros((2,64),dtype=torch.bool); mask[:,::7]=True
        backbone.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH):
            output=backbone(x,mask)
            value=output['x_norm_patchtokens']
            (value*torch.linspace(-.7,1.3,value.numel()).reshape(value.shape)).mean().backward()
        return output,x.grad,{n:None if p.grad is None else p.grad.clone() for n,p in backbone.named_parameters()}

    def equal(a,b):
        if isinstance(a,dict):
            return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        return a is b if a is None or b is None else bool(torch.equal(a,b))

    for kind in ('v1','v2'):
        for chunks in (0,2,4):
            config={'model_type':kind,'global_crops_size':16,'local_crops_size':8,'patch_size':4,
                    'embed_dim':48,'num_heads':2,'depth':4,'block_chunks':chunks,
                    'drop_path_rate':.3,'rope_shift_coords':.05,'rope_jitter_coords':1.05}
            torch.manual_seed(20260920)
            baseline=original_model.DinoVitStudentTeacher._build_backbone(config).train()
            base=evaluate(baseline)
            for setting in ('omitted',False,True):
                cfg=dict(config)
                if setting!='omitted': cfg['pad_sdpa_heads']=setting
                materialized=model._materialize_backbone_config(cfg)
                # The trainer materializes once, then the backbone materializes again.
                rematerialized=model._materialize_backbone_config(materialized)
                expected=setting is True
                assert materialized['pad_sdpa_heads'] is expected
                assert rematerialized['pad_sdpa_heads'] is expected
                candidate=model.DinoVitStudentTeacher._build_backbone(rematerialized).train()
                assert baseline.state_dict().keys()==candidate.state_dict().keys()
                candidate.load_state_dict(baseline.state_dict(),strict=True)
                attentions=[m for m in candidate.modules() if isinstance(m,eva.EvaAttention)]
                assert len(attentions)==4 and all(m.pad_sdpa_heads is expected for m in attentions)
                with patch.object(eva,'padded_training_sdpa',wraps=helper.padded_training_sdpa) as observed:
                    result=evaluate(candidate)
                assert observed.call_count==(4 if expected else 0)
                assert all(equal(a,b) for a,b in zip(base,result))
                receipt['cases'].append({'model_type':kind,'block_chunks':chunks,'setting':setting,
                    'state_keys_unchanged':True,'double_materialization_preserves_option':True,
                    'helper_calls':observed.call_count,'outputs_input_and_parameter_gradients_exact':True})

    # Real CPU calls must delegate even when enabled; no CUDA capabilities queried.
    q=torch.randn(2,2,7,54,dtype=torch.float64,requires_grad=True)
    k=torch.randn(2,2,7,54,dtype=torch.float64,requires_grad=True)
    v=torch.randn(2,2,7,54,dtype=torch.float64,requires_grad=True)
    native=F.scaled_dot_product_attention
    with patch.object(helper,'_enabled_fused_backends',side_effect=AssertionError('CPU capability query')):
        assert torch.equal(helper.padded_training_sdpa(q,k,v,enabled=True),native(q,k,v))
    # Mock only device/capability gates to exercise differentiable padding math
    # on CPU; this is not evidence of real fused-backend availability.
    gate_checks=[]
    for scale in (None,.19):
        qa,ka,va=[t.detach().clone().requires_grad_(True) for t in (q,k,v)]
        qb,kb,vb=[t.detach().clone().requires_grad_(True) for t in (q,k,v)]
        with sdpa_kernel(SDPBackend.MATH):
            out_a=native(qa,ka,va,scale=scale)
            with patch.object(helper,'_eligible',return_value=True), \
                 patch.object(helper,'_enabled_fused_backends',return_value=[object()]), \
                 patch.object(helper,'_has_fused_backend',side_effect=[False,True]):
                out_b=helper.padded_training_sdpa(qb,kb,vb,scale=scale,enabled=True)
            out_a.square().sum().backward(); out_b.square().sum().backward()
        values=[(out_a,out_b),(qa.grad,qb.grad),(ka.grad,kb.grad),(va.grad,vb.grad)]
        assert all(torch.allclose(a,b,rtol=1e-12,atol=1e-12) for a,b in values)
        gate_checks.append({'case':'mocked_candidate_math','scale':scale,'max_abs':max(float((a-b).abs().max()) for a,b in values)})
    for scenario in ('disabled','already_supported','missing_api','candidate_unsupported','candidate_api_error','no_enabled_backend'):
        with patch.object(helper,'_eligible',return_value=True), \
             patch.object(helper,'_enabled_fused_backends',side_effect=AttributeError('missing') if scenario=='missing_api' else None,
                          return_value=[] if scenario=='no_enabled_backend' else [object()]), \
             patch.object(helper,'_has_fused_backend',side_effect=[False,RuntimeError('unsupported')] if scenario=='candidate_api_error' else None,
                          return_value=scenario=='already_supported'):
            seen=[]
            def spy(a,b,c,**kwargs):
                seen.append(a is q and b is k and c is v)
                return native(a,b,c,**kwargs)
            out=helper.padded_training_sdpa(q,k,v,enabled=scenario!='disabled',native=spy)
            assert seen==[True] and torch.equal(out,native(q,k,v))
        gate_checks.append({'case':scenario,'native_tensor_identity_preserved':True})
    receipt['helper_checks']=gate_checks
    receipt['cuda_initialized']=torch.cuda.is_initialized()
    assert not receipt['cuda_initialized']
    receipt['status']='PASS'
    target=Path(__file__).resolve().parent/'CPU_RECEIPT.json'
    target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','integration_cases':len(receipt['cases']),
                      'helper_cases':len(gate_checks),'cuda_initialized':False}))


if __name__=='__main__':
    main()
