"""Bounded real-CT training-step benchmark using pinned or manifest-verified Dinovol.

Pretrained backbone, freshly initialized full projection heads and AdamW state;
not a resumed checkpoint, convergence test, or complete historical recipe.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

PIN = 'f07d33be6a00d12ace7d6a9465efe17c78ed7b47'
TEACHER_SHA = 'e041ca870dd2570f8a44d1dd26db1197b3f74121f62023bc774fbc9d40e51a59'
SOURCE = 's3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr'
FIXTURES = {
    'fb5d3659016e1bb760c608336e3ae1f356414464362021ec4e96ce392de94ad5': [3840,4096,3712,3968,1344,1600],
    '819b93a9fa7d3ab30735caea4f8d558f6e83439e9957942b886f1118a0a369f4': [4352,4608,3072,3328,2560,2816],
    '8755eceeffffe19ebc89fb1c001aa29a4b7f3c31e5b577714a31297f34268b60': [4608,4864,3072,3328,2560,2816],
}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def verify_integrated_source(manifest_path, dinovol_root):
    """Verify the isolated assembly, every recorded source file, and patches."""
    manifest_path = Path(manifest_path).resolve()
    assembly_root = manifest_path.parent
    assembly = json.loads(manifest_path.read_text(encoding='utf-8'))
    if assembly.get('status') != 'PASS' or assembly.get('base_commit') != PIN:
        raise ValueError('Integrated assembly must pass and use the expected source pin')
    expected_root = (assembly_root / assembly['dinovol_root_relative']).resolve()
    if Path(dinovol_root).resolve() != expected_root:
        raise ValueError('--dinovol-root does not match the integrated assembly root')
    final_manifest = assembly_root / 'FINAL_FILES.json'
    if file_sha(final_manifest) != assembly['final_files_manifest_sha256']:
        raise ValueError('Integrated FINAL_FILES manifest hash mismatch')
    files = json.loads(final_manifest.read_text(encoding='utf-8'))
    if len(files) != assembly['final_file_count']:
        raise ValueError('Integrated source file count mismatch')
    actual_hashes = {}
    for relative, expected in files.items():
        path = (assembly_root / relative).resolve()
        if not path.is_relative_to(expected_root):
            raise ValueError('Integrated source manifest path is outside dinovol root')
        digest = file_sha(path)
        if digest != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError('Integrated source hash/size mismatch: ' + relative)
        actual_hashes[relative] = digest
    # Imports may create bytecode caches, which are not source artifacts.
    actual_files = {p.resolve().relative_to(assembly_root).as_posix()
                    for p in expected_root.rglob('*') if p.is_file()
                    and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual_files != set(files):
        raise ValueError('Integrated source has missing or unrecorded files')
    patches = []
    for entry in assembly['patches']:
        path = (assembly_root / entry['path']).resolve()
        if not path.is_relative_to(assembly_root):
            raise ValueError('Integrated patch path is outside assembly root')
        digest = file_sha(path)
        if digest != entry['sha256'] or not entry['applied'] or entry['apply_check'] != 'PASS':
            raise ValueError('Integrated patch verification failed: ' + entry['label'])
        patches.append({'order':entry['order'],'label':entry['label'],'sha256':digest})
    return {'method':'integrated source manifests; no clean-checkout claim',
            'assembly_manifest_sha256':file_sha(manifest_path),
            'final_files_manifest_sha256':file_sha(final_manifest),
            'base_commit':assembly['base_commit'],'patches':patches,
            'verified_file_count':len(files),'actual_file_sha256':actual_hashes,
            'changed_files':assembly['changed_files'],'added_files':assembly['added_files']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dinovol-root', type=Path, required=True, help='Pinned Villa dinovol/ directory')
    ap.add_argument('--integrated-manifest', type=Path,
                    help='ASSEMBLY_MANIFEST.json for the isolated patched dinovol source')
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--volumes', type=Path, nargs=2, required=True, help='Two distinct known A/B/C 256-cubed CT arrays')
    ap.add_argument('--mode', choices=('native', 'padded', 'math-padded', 'fp32-padded'), required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--warmups', type=int, default=0)
    ap.add_argument('--schedule-step', type=int, default=35000)
    ap.add_argument('--seed', type=int, default=20260920)
    ap.add_argument('--initial-scale', type=float, default=128.0)
    ap.add_argument('--gpu-memory-fraction', type=float, default=0.85)
    ap.add_argument('--global-size', type=int, choices=(64,96,128), default=128)
    ap.add_argument('--recipe', choices=('feasibility','representative'), default='feasibility')
    ap.add_argument('--helper', choices=('research','guarded'), default='research')
    ap.add_argument('--profile', action='store_true', help='Additional separately audited update, excluded from timings')
    args = ap.parse_args()
    if args.steps < 1 or args.warmups < 0 or not (1250 < args.schedule_step < 1000000):
        ap.error('Require steps>=1, warmups>=0, and 1250<schedule-step<1000000')
    if args.initial_scale <= 0:
        ap.error('initial-scale must be positive')
    if not 0 < args.gpu_memory_fraction <= 1:
        ap.error('gpu-memory-fraction must be in (0,1]')
    if args.integrated_manifest and args.mode not in ('native','padded'):
        ap.error('Integrated source mode supports native/padded only; math/FP32 controls use the original source interception path')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
    report = {'status': 'STARTED', 'mode': args.mode, 'scope': __doc__,
              'script_sha256': file_sha(__file__), 'source_pin': PIN,
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'steps': [], 'official_trainer_edited': bool(args.integrated_manifest)}
    try:
        if args.integrated_manifest:
            report['source_validation'] = verify_integrated_source(args.integrated_manifest,args.dinovol_root)
            report['source_checkout_head'] = None
        else:
            # Check the complete tracked source subtree before importing it.
            subprocess.run(['git','-C',str(args.dinovol_root),'diff','--exit-code',PIN,'--','dinovol_2'],
                           check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            report['source_checkout_head'] = subprocess.check_output(
                ['git','-C',str(args.dinovol_root),'rev-parse','HEAD'],text=True).strip()
            report['source_validation'] = {'method':'git diff against pinned official source','base_commit':PIN}
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        sys.path.insert(0, str(args.dinovol_root.resolve()))
        from dinovol_2.pretrain import DinoIBOTPretrainer
        import dinovol_2.model.dinov2_eva as eva_module
        from dinovol_2.dataset.normalization import get_normalization
        from training_sdpa import training_sdpa
        import training_sdpa as helper_module
        if args.integrated_manifest:
            from dinovol_2.model import training_sdpa_padding as helper_module
            selected_helper = None  # The integrated EvaAttention invokes it.
        elif args.helper == 'guarded':
            from integration import training_sdpa_padding as guarded_module
            def selected_helper(q,k,v,**kwargs):
                return guarded_module.padded_training_sdpa(q,k,v,enabled=True,**kwargs)
            helper_module = guarded_module
        else:
            selected_helper = training_sdpa

        # Assert Python actually resolved these modules from the verified tree.
        import dinovol_2.pretrain as pretrain_module
        import dinovol_2.model.model as model_module
        for module, relative in ((pretrain_module,'dinovol_2/pretrain.py'),
                                 (model_module,'dinovol_2/model/model.py'),
                                 (eva_module,'dinovol_2/model/dinov2_eva.py')):
            if Path(module.__file__).resolve() != (args.dinovol_root/relative).resolve():
                raise ValueError('Imported module came from another source tree: '+relative)
        if args.integrated_manifest and Path(helper_module.__file__).resolve() != (
                args.dinovol_root/'dinovol_2/model/training_sdpa_padding.py').resolve():
            raise ValueError('Integrated attention helper imported from another source tree')
        report['source_hashes'] = {p: file_sha(args.dinovol_root / p) for p in (
            'dinovol_2/pretrain.py', 'dinovol_2/model/model.py', 'dinovol_2/model/dinov2_eva.py',
            'dinovol_2/loss/dino_clstoken_loss.py', 'dinovol_2/loss/ibot_patch_loss.py',
            'dinovol_2/loss/koleo_loss.py')}
        report['helper_sha256'] = file_sha(helper_module.__file__)
        report['attention_helper_kind'] = 'integrated_source_guarded' if args.integrated_manifest else args.helper
        report['attention_interception'] = 'log then native only; source helper controls padding' if args.integrated_manifest else 'benchmark helper interception'
        report['versions'] = {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                              'cuda': torch.version.cuda}
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        normalizer = get_normalization('robust')
        samples, provenance = [], []
        for volume_path in args.volumes:
            array = np.load(volume_path, allow_pickle=False)
            if array.shape != (256,)*3 or array.dtype != np.uint8:
                raise ValueError('CT must be 256-cubed uint8')
            digest = hashlib.sha256(array.tobytes(order='C')).hexdigest()
            if digest not in FIXTURES or any(p['array_sha256'] == digest for p in provenance):
                raise ValueError('Require two distinct frozen real CT fixtures')
            sample = {}
            from itertools import product
            local_starts = [(32,32,32),(160,160,160)] if args.recipe == 'feasibility' else list(product((32,160),repeat=3))
            global_starts = [(0,0,0),(256-args.global_size,)*3]
            for key, size, starts in (('global_views',args.global_size,global_starts), ('local_views',64,local_starts)):
                views = []
                for start in starts:
                    z,y,x = start
                    crop = array[z:z+size,y:y+size,x:x+size].astype(np.float32, copy=True)
                    views.append(torch.from_numpy(normalizer.run(crop)).unsqueeze(0).clone())
                sample[key] = views
            samples.append(sample)
            provenance.append({'source_uri': SOURCE, 'bbox_zyx': FIXTURES[digest], 'voxel_size_um': 9.362,
                               'array_sha256': digest, 'file_sha256': file_sha(volume_path)})
        global_crops = torch.stack([s['global_views'][view] for view in range(2) for s in samples])
        n_local_views = len(samples[0]['local_views'])
        local_crops = torch.stack([s['local_views'][view] for view in range(n_local_views) for s in samples])
        n_tokens = (args.global_size//8)**3
        n_per_mask = 128 if args.recipe == 'feasibility' else n_tokens//4
        masked_rows = range(4) if args.recipe == 'feasibility' else (0,3)
        masks = torch.zeros((4,n_tokens), dtype=torch.bool)
        for row in masked_rows:
            masks[row, ((torch.arange(n_per_mask)*n_tokens)//n_per_mask + row*7) % n_tokens] = True
        indices = masks.flatten().nonzero().flatten()
        batch = {'collated_global_crops': global_crops, 'collated_local_crops': local_crops,
                 'collated_masks': masks, 'mask_indices_list': indices,
                 'masks_weight': torch.full((len(indices),), 1/n_per_mask, dtype=torch.float32),
                 'n_masked_patches': torch.tensor([len(indices)],dtype=torch.long),
                 'n_global_views': 2, 'n_local_views': n_local_views, 'batch_size': 2}
        report['input'] = {'volumes': provenance,
            'construction': {'global_size':args.global_size,'global_starts_zyx':global_starts,'local_starts_zyx':local_starts,
                             'normalization':'robust separately per view','view_augmentation':False,'collation':'view-major'},
            'mask_construction': {'recipe':args.recipe,'masked_rows':list(masked_rows),'tokens_per_view':n_tokens,
                                 'masked_tokens_per_masked_view':n_per_mask,'indices':'floor(arange(n)*tokens/n)+row*7 mod tokens'},
            'tensor_hashes': {k: hashlib.sha256(v.numpy().tobytes(order='C')).hexdigest() for k,v in batch.items() if isinstance(v,torch.Tensor)}}
        if file_sha(args.checkpoint) != TEACHER_SHA:
            raise ValueError('Official slim teacher SHA256 mismatch')
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        original_config = payload['config']
        model_config = dict(original_config['model'])
        model_config['grad_checkpointing'] = True
        model_config['global_crops_size'] = [args.global_size]*3
        if args.integrated_manifest:
            model_config['pad_sdpa_heads'] = args.mode == 'padded'
        # Keep the official default full131072 prototype heads. Do not rely on
        # model.pretrained_weights: the pinned trainer materializer drops it.
        config = {k: original_config[k] for k in (
            'lr','min_lr','weight_decay','weight_decay_end','patch_embed_lr_mult',
            'freeze_last_layer_steps','momentum_teacher','final_momentum_teacher',
            'warmup_steps','warmup_teacher_temp','teacher_temp','warmup_teacher_temp_steps',
            'clip_grad','max_iterations','centering')}
        config.update({'model': model_config, 'device': 'cuda', 'use_amp': True,
            'use_ddp': False, 'resume': False, 'auto_resume': False, 'batch_size': 2,
            'num_local_crops': n_local_views, 'ibot_masked_loss_chunk_size': 256,
            'dino_loss_weight': 1.0, 'ibot_loss_weight': 1.0, 'koleo_loss_weight': 0.1,
            'gram': {'enabled': False}, 'point_supervision': {'enabled': False},
            'val_every_n': 0, 'save_every_n': 0, 'task_eval_every': 0,
            'output_dir': str(args.output / 'trainer')})
        report['config'] = config
        report['protocol'] = {'seed': args.seed, 'schedule_start': args.schedule_step,
            'warmup_updates': args.warmups, 'timed_updates': args.steps, 'initial_scaler': args.initial_scale,
            'autocast_dtype': 'default CUDA float16; model/optimizer parameters float32',
            'gpu_memory_fraction': args.gpu_memory_fraction, 'tf32_matmul': False, 'cudnn_benchmark': False,
            'starting_state': 'official pretrained teacher backbone copied into student; fresh seeded DINO/iBOT heads; teacher synchronized; fresh AdamW/scaler/loss state',
            'recipe_deviations': f'B2 versus3;{n_local_views} locals versus8;global{args.global_size} versus128;fixed masks documented in input versus sampled10-50% at probability0.5;checkpointing enabled;loss chunk256;fixed9.362um CT versus original2.4um corpus;no stochastic crop augmentation;not historical resume',
            'timing_scope': 'synchronized original train_step including copies/loss/backward/AdamW/EMA; excludes initialization, view preparation, snapshots, audits, and optional profile'}
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        report['gpu'] = {'name': torch.cuda.get_device_name(0), 'total_memory': torch.cuda.get_device_properties(0).total_memory}
        trainer = DinoIBOTPretrainer(config)
        state = payload['teacher']
        expected = trainer.model_module.student.backbone.state_dict()
        if set(state) != set(expected):
            if not state or not all(k.startswith('backbone.') for k in state):
                raise ValueError('Slim teacher backbone key contract differs')
            state = {k.removeprefix('backbone.'):v for k,v in state.items()}
        if set(state) != set(expected):
            raise ValueError('Strict backbone key mismatch')
        trainer.model_module.student.backbone.load_state_dict(state, strict=True)
        trainer.model_module.synchronize_teacher_from_student()
        report['checkpoint'] = {'sha256': TEACHER_SHA, 'strict_backbone_keys': len(state),
            'student_parameters': sum(p.numel() for p in trainer.model_module.student.parameters()),
            'dino_out_dim': trainer.dino_loss.center.shape[-1], 'ibot_out_dim': trainer.ibot_patch_loss.center.shape[-1]}
        del payload, state, expected
        trainer.scaler = torch.amp.GradScaler('cuda', init_scale=args.initial_scale, enabled=True)
        original_sdpa = F.scaled_dot_product_attention
        original_checkpoint = eva_module.checkpoint
        # The pinned EVA call passes RoPE keyword arguments. Current PyTorch's
        # default reentrant checkpoint rejects these; use its explicit supported
        # non-reentrant implementation equally in all benchmark arms. This is a
        # runtime compatibility adapter, not an edit to the official source.
        checkpoint_lines = [line for line in Path(eva_module.__file__).read_text(encoding='utf-8').splitlines()
                            if 'x = checkpoint(' in line]
        source_checkpoint_fixed = bool(args.integrated_manifest) and len(checkpoint_lines)==3 and all(
            'use_reentrant=False' in line for line in checkpoint_lines)
        checkpoint_adapter = None if source_checkpoint_fixed else functools.partial(original_checkpoint, use_reentrant=False)
        report['checkpoint_compatibility_adapter'] = {
            'symbol': 'dinovol_2.model.dinov2_eva.checkpoint',
            'use_reentrant': False,
            'all_modes': True,
            'runtime_adapter_applied': not source_checkpoint_fixed,
            'source_contains_explicit_nonreentrant_calls': source_checkpoint_fixed,
            'reason': 'Integrated source contains explicit use_reentrant=False; no runtime checkpoint adapter' if source_checkpoint_fixed else 'Pinned EVA passes RoPE kwargs; current PyTorch default reentrant checkpoint rejects unexpected keyword arguments',
        }
        call_counts = {}

        def intercepted(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False,*,scale=None,enable_gqa=False):
            key = str((tuple(q.shape),str(q.dtype),torch.is_grad_enabled()))
            call_counts[key] = call_counts.get(key,0)+1
            options = dict(attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal,scale=scale,enable_gqa=enable_gqa)
            if args.integrated_manifest or args.mode == 'native':
                return original_sdpa(q,k,v,**options)
            if args.mode == 'fp32-padded':
                with torch.autocast(device_type=q.device.type,enabled=False):
                    return training_sdpa(q.float(),k.float(),v.float(),native=original_sdpa,**options).to(q.dtype)
            if args.mode == 'math-padded':
                with sdpa_kernel(SDPBackend.MATH):
                    return training_sdpa(q,k,v,native=original_sdpa,**options)
            return selected_helper(q,k,v,native=original_sdpa,**options)

        def snapshot(branch):
            return {name:p.detach().cpu().clone() for name,p in branch.named_parameters()}

        def audit(before_student,before_teacher,momentum):
            totals = {'student_delta_l2_sq':0.0,'teacher_delta_l2_sq':0.0,'gradient_l2_sq':0.0,
                      'student_changed_parameters':0,'teacher_changed_parameters':0,
                      'gradient_parameters':0,'nonfinite_gradient_parameters':0,'teacher_gradient_parameters':0,
                      'ema_max_abs_error':0.0,'ema_allclose':True}
            sample_arrays = {'gradient':[],'student_before':[],'student_after':[],'student_delta':[],
                             'teacher_before':[],'teacher_after':[],'teacher_delta':[]}
            layout = []
            teacher_parameters = dict(trainer.model_module.teacher.named_parameters())
            for name,p in trainer.model_module.student.named_parameters():
                after = p.detach().cpu()
                before = before_student[name]
                delta = (after-before).double()
                value = float((delta*delta).sum())
                totals['student_delta_l2_sq'] += value
                totals['student_changed_parameters'] += int(value>0)
                grad = p.grad.detach().float().cpu() if p.grad is not None else None
                if grad is not None:
                    totals['gradient_parameters'] += 1
                    totals['nonfinite_gradient_parameters'] += int(not torch.isfinite(grad).all())
                    totals['gradient_l2_sq'] += float((grad.double().square()).sum())
                teacher_p = teacher_parameters[name]
                teacher_after = teacher_p.detach().cpu()
                totals['teacher_gradient_parameters'] += int(teacher_p.grad is not None)
                teacher_delta = (teacher_after-before_teacher[name]).double()
                tv = float(teacher_delta.square().sum())
                totals['teacher_delta_l2_sq'] += tv
                totals['teacher_changed_parameters'] += int(tv>0)
                ema_expected = before_teacher[name]*momentum + after*(1-momentum)
                totals['ema_max_abs_error'] = max(totals['ema_max_abs_error'],float((teacher_after-ema_expected).abs().max()))
                totals['ema_allclose'] &= bool(torch.allclose(teacher_after,ema_expected,rtol=5e-6,atol=5e-7))
                count = min(128,p.numel())
                idx = torch.arange(count,dtype=torch.int64)*(p.numel()-1)//max(count-1,1)
                layout.append({'name':name,'parameter_numel':p.numel(),'sample_count':len(idx),'sample_indices':idx.tolist(),'has_gradient':grad is not None})
                sample_arrays['student_before'].append(before.flatten()[idx].numpy())
                sample_arrays['student_after'].append(after.flatten()[idx].numpy())
                sample_arrays['student_delta'].append(delta.float().flatten()[idx].numpy())
                sample_arrays['teacher_before'].append(before_teacher[name].flatten()[idx].numpy())
                sample_arrays['teacher_after'].append(teacher_after.flatten()[idx].numpy())
                sample_arrays['teacher_delta'].append(teacher_delta.float().flatten()[idx].numpy())
                sample_arrays['gradient'].append(grad.flatten()[idx].numpy() if grad is not None else np.zeros(len(idx),np.float32))
            return totals,{k:np.concatenate(v) for k,v in sample_arrays.items()},layout

        def run_step(number,kind,profile=False):
            before_student = snapshot(trainer.model_module.student)
            before_teacher = snapshot(trainer.model_module.teacher)
            scale_before = trainer.scaler.get_scale()
            named_student = dict(trainer.model_module.student.named_parameters())
            optimizer_steps_before = {name:int(trainer.optimizer.state.get(p,{}).get('step',0)) for name,p in named_student.items()}
            optimizer_before = max(optimizer_steps_before.values(),default=0)
            call_counts.clear()
            schedule_step = args.schedule_step+number
            momentum = float(trainer.momentum_schedule[schedule_step])
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with patch.object(F,'scaled_dot_product_attention',intercepted), \
                    (nullcontext() if checkpoint_adapter is None else patch.object(eva_module,'checkpoint',checkpoint_adapter)):
                if profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        metrics = trainer.train_step(batch,schedule_step)
                    prof.export_chrome_trace(str(args.output/'trace.json'))
                    backend_names = ('aten::_scaled_dot_product_attention_math','aten::_scaled_dot_product_efficient_attention',
                                     'aten::_scaled_dot_product_efficient_attention_backward','aten::_scaled_dot_product_flash_attention',
                                     'aten::_scaled_dot_product_cudnn_attention')
                    report['profile_operators'] = {event.key:event.count for event in prof.key_averages() if event.key in backend_names}
                else:
                    metrics = trainer.train_step(batch,schedule_step)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter()-start)*1000
            memory = {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            totals,arrays,layout = audit(before_student,before_teacher,momentum)
            optimizer_steps_after = {name:int(trainer.optimizer.state.get(p,{}).get('step',0)) for name,p in named_student.items()}
            optimizer_after = max(optimizer_steps_after.values(),default=0)
            every_optimizer_step = all(optimizer_steps_after[name] == optimizer_steps_before[name] + int(p.grad is not None)
                                       for name,p in named_student.items())
            checks = {'finite_loss':all(np.isfinite(v) for v in metrics.values()),
                      'finite_gradients':totals['nonfinite_gradient_parameters']==0,
                      'student_updated':totals['student_changed_parameters']>0,
                      'teacher_updated':totals['teacher_changed_parameters']>0,
                      'teacher_no_gradients':totals['teacher_gradient_parameters']==0,
                      'ema_consistent':totals['ema_allclose'],'optimizer_advanced':optimizer_after==optimizer_before+1,
                      'every_optimizer_parameter_step_correct':every_optimizer_step}
            step_report = {'kind':kind,'number':number,'schedule_step':schedule_step,'elapsed_ms':elapsed_ms if not profile else None,
                'metrics':metrics,'gpu_memory':memory,'scaler_before':scale_before,'scaler_after':trainer.scaler.get_scale(),
                'optimizer_step_before':optimizer_before,'optimizer_step_after':optimizer_after,'teacher_momentum':momentum,
                'optimizer_parameter_steps_before':optimizer_steps_before,'optimizer_parameter_steps_after':optimizer_steps_after,
                'sdpa_calls':dict(call_counts),'audit':totals,'checks':checks}
            report['steps'].append(step_report)
            for key,array in arrays.items():
                path=args.output/f'{kind}_{number:02d}_{key}_samples.npy'
                np.save(path,array,allow_pickle=False)
            (args.output/'parameter_sample_layout.json').write_text(json.dumps(layout,indent=2),encoding='utf-8')
            (args.output/f'{kind}_{number:02d}_parameter_sample_layout.json').write_text(json.dumps(layout,indent=2),encoding='utf-8')
            print(json.dumps({'number':number,'kind':kind,'ms':step_report['elapsed_ms'],'loss':metrics['loss'],'checks':checks}),flush=True)
            if not all(checks.values()):
                raise RuntimeError('Training step audit failed; no automatic retries or changed recipe')

        for number in range(args.warmups+args.steps):
            run_step(number,'warmup' if number<args.warmups else 'timed')
        if args.profile:
            run_step(args.warmups+args.steps,'profile',True)
        timings=[s['elapsed_ms'] for s in report['steps'] if s['kind']=='timed']
        report['timing_summary']={'median_ms':float(np.median(timings)),'p90_ms':float(np.percentile(timings,90)),'timed_steps':len(timings)}
        report['status']='TRAINING_PROBE_COMPLETE'
    except Exception:
        report['status']='FAILED'
        report['traceback']=traceback.format_exc()
        raise
    finally:
        report['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        (args.output/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
