# Dinovol training fixes and an experimental 24 GB training path

Two independently applicable fixes address failures encountered while preparing pretrained Dinovol training on real scroll CT:

- **Requested model options disappeared before construction.** The trainer discarded `pretrained_weights`, loading switches and head settings when normalizing backbone configuration. The fix preserves the authored options and overlays normalized backbone values.
- **Activation checkpointing rejected the model's RoPE arguments.** On the tested PyTorch runtime, the implicit reentrant checkpoint implementation raises `ValueError: Unexpected keyword arguments: rope,rope_shape,rope_coords`. The fix explicitly selects the non-reentrant implementation at all three existing call sites.

A separate, default-off attention option also completed seven real training updates at 128³ resolution with the full model and heads on a 24 GB RTX 3090, where the corresponding native-attention run exhausted memory. This is a verified short training run, with numerical differences and convergence limits detailed below.

Both patches target Villa commit [`f07d33be6a00d12ace7d6a9465efe17c78ed7b47`](https://github.com/ScrollPrize/villa/tree/f07d33be6a00d12ace7d6a9465efe17c78ed7b47/dinovol). They do not require the experimental attention-padding code included separately under `research/training`.

## Real-scroll evidence

The tests use the official ps8 Paris4 step352500 teacher, its full 864-wide, 24-block backbone, and real PHerc0139 CT at 9.362 micrometres. The recorded environment is Windows, RTX 3090, Python 3.14.6 and PyTorch 2.13.0+cu126. Results below apply to these declared tests, rather than all platforms or training recipes.

| Test | Before | After |
|---|---|---|
| Actual trainer warm-start constructor | 24/463 backbone state entries matched the checkpoint; requested weights were discarded | All 463/463 entries exact in both student and teacher |
| Real 128³ CT patch embeddings against a directly loaded reference | Relative L2 error 1.42646 | Class, register and patch outputs all bit-identical |
| Requested 32-output test heads | Silently became 131072 outputs | Requested 32 outputs and matching losses |
| Activation checkpointing with existing RoPE kwargs | Keyword-argument failure on the tested runtime | Forward/backward completed |
| Full pretrained backbone, two real 64³ views: fixed checkpointing versus checkpointing off | Reference computation | Compared class and patch outputs, plus all 215,858,304 produced parameter-gradient values across 438 tensors, bit-identical |

Evidence: [warm-start report](config_fix/real_01/report.json), [real checkpoint-gradient report](checkpoint_fix/real_checkpoint_gradients.json), [configuration CPU regression](config_fix/cpu_regression_01/report.json), and [checkpoint CPU regression](checkpoint_fix/cpu_01/report.json).

The warm-start experiment invokes the actual trainer initializer in each arm. It never repairs either trainer by manually assigning pretrained weights afterward. The 24 matching entries before the fix are matching initialization/default values; they are not evidence that learned checkpoint weights were loaded. A verified serialization adapter adds the `backbone.` key prefix expected by the existing loader. Only the independent reference receives a direct strict load. The requested small heads are a test configuration, not a recommended replacement for the official training recipe.

The checkpoint CPU regression covers six architecture/chunking cases with two seeds each: 12 case/seed runs and 1,092 output/gradient comparisons, all exact or matching absent gradients. It checks stochastic depth, RoPE randomness, masking, input gradients and RNG behavior. The real-gradient test uses a fixed differentiable embedding objective and compares class/patch outputs; it does not compare register outputs or establish DINO convergence. The separate warm-start test compares all three class/register/patch outputs.

## Reproduce the two fixes

Use an existing environment satisfying the pinned Dinovol dependencies. The scripts do not install packages. Keep a pristine checkout for the regression commands: their source checks deliberately reject an already patched tree. Use new output paths inside the indicated package folders.

```sh
git clone https://github.com/ScrollPrize/villa.git /absolute/path/to/villa
git -C /absolute/path/to/villa checkout f07d33be6a00d12ace7d6a9465efe17c78ed7b47

python config_fix/run_cpu_regression.py --villa /absolute/path/to/villa --output /absolute/path/to/dinovol-training-fixes/config_fix/cpu_rerun
python checkpoint_fix/verify_cpu.py --dinovol-root /absolute/path/to/villa/dinovol --output /absolute/path/to/dinovol-training-fixes/checkpoint_fix/cpu_rerun
```

The real-scroll commands additionally require the official slim teacher and the fixed PHerc0139 A256³ uint8 NumPy volume. They verify these inputs against the hashes below. Run GPU tests separately when the device is free.

```sh
python config_fix/run_real_checkpoint.py --villa /absolute/path/to/villa --checkpoint /absolute/path/to/teacher.pt --volume /absolute/path/to/A256.npy --output /absolute/path/to/dinovol-training-fixes/config_fix/real_rerun
python research/training/verify_checkpoint.py --dinovol-root /absolute/path/to/villa/dinovol --checkpoint /absolute/path/to/teacher.pt --volume /absolute/path/to/A256.npy --output /absolute/path/to/dinovol-training-fixes/checkpoint_fix/real_checkpoint_rerun.json
```

| Input | Required identity |
|---|---|
| Teacher file SHA-256 | `e041ca870dd2570f8a44d1dd26db1197b3f74121f62023bc774fbc9d40e51a59` |
| A256³ pixel SHA-256 | `fb5d3659016e1bb760c608336e3ae1f356414464362021ec4e96ce392de94ad5` |
| CT source | `s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr` |
| Level-0 parent box, Z0/Z1/Y0/Y1/X0/X1 | `3840,4096,3712,3968,1344,1600` |

The warm-start test uses the first 128³ subvolume. The gradient test uses 64³ views beginning at offsets `(32,32,32)` and `(160,160,160)`. Both use the official training robust normalization. Checkpoint weights and CT data are not bundled here; the earlier [public diagnostic package](https://github.com/TAUIL-Abd-Elilah/dinovol-attention-diagnostic) documents acquisition of these fixed assets.

After reproduction, apply either fix independently to a separate pinned checkout. To apply both, use this order:

```sh
git -C /absolute/path/to/villa apply --check /absolute/path/to/dinovol-training-fixes/config_fix/preserve_model_config.patch
git -C /absolute/path/to/villa apply /absolute/path/to/dinovol-training-fixes/config_fix/preserve_model_config.patch
git -C /absolute/path/to/villa apply --check /absolute/path/to/dinovol-training-fixes/checkpoint_fix/explicit_nonreentrant.patch
git -C /absolute/path/to/villa apply /absolute/path/to/dinovol-training-fixes/checkpoint_fix/explicit_nonreentrant.patch
```

The warm-start fix concerns authored model initialization options; the separate `resume_from` path is not claimed to be universally broken. Default model behavior passed exact CPU regression. The checkpoint patch leaves the default `grad_checkpointing=False` path unchanged. Neither fix enables attention padding.

## Experimental default-off training attention

The current training candidate pads the 54-wide attention heads to 56 and computes attention internally in FP32, retaining the original attention scale and native autograd, then casting the attention result back to FP16. It remains an explicit, default-off experiment. The [source integration patch](research/training/integration/dinovol_training_head_padding.patch) routes `model.pad_sdpa_heads` through the actual model factory and Eva attention call; unsupported inputs follow the native path. The two bug fixes work independently of this option.

### Actual integrated source at 128³

The [integrated 128³ report](research/training/integrated128_fp32_01/report.json) records **seven completed training updates**, using the source attention implementation; the runtime hook only logged and forwarded calls. No runtime checkpoint-compatibility adapter was used. All recorded checks passed: finite loss and gradients, student and teacher updates, teacher remaining gradient-free, consistent EMA, optimizer advancement, and correct optimizer parameter-step counts.

The recipe uses two real PHerc0139 regions, **batch size two**, **two 128³ global views and eight 64³ local views per sample**, the full **313,254,496 student parameters**, and **131072-output DINO/iBOT heads**. A deterministic mask covers 25% of tokens in two of the four global views, representing a 0.5 view-masking rate. It uses separate robust normalization per view, no stochastic image augmentation, checkpointing, and masked-loss chunks of 256. The pretrained backbone starts with fresh seeded heads, optimizer and scaler; this is not a historical training-state resume. These declared settings differ from the original batch-three, 2.4-micrometre corpus recipe.

One warmup, five timed updates and one profiled update completed. Median timed step duration was **7,417 ms** and the maximum timed-step allocated memory was **12,479,086,592 bytes (11.62 GiB)**. The profile recorded 120 efficient-attention forward calls and 48 backward calls. The [corresponding native-attention 128³ attempt](research/training/representative128_native_01/report.json) exhausted memory with the same batch/view recipe and a 95% CUDA allocator ceiling. This establishes that the tested patched workflow fits this GPU; it provides no paired 128³ numerical-quality comparison because the native attempt could not complete.

### Paired numerical check at 96³

In the [paired 96³ comparison](research/training/comparison96_fp32.json), both arms performed seven actual optimizer/teacher updates: one warmup, five timed and one profiled. This comparison used benchmark-side attention substitution; the separate 128³ run validates the actual source integration. Median synchronized training-step time fell from **3,038 to 2,527 ms**, a **1.202× speedup**, with **15.79% lower peak allocated memory**. Timing includes the original step's copies, losses, backward, AdamW and teacher update, excluding initialization, input preparation and numerical audits.

Across those seven updates, the maximum absolute difference in **total loss** was **9.5367e-6**. Relative L2 differences in 58,240 sampled post-clipping gradient values ranged from **0.1282% to 0.4220%**. Sampled Adam-update differences were **2.9561%** for the first update and **0.3122%–0.5026%** thereafter. These are sampled update comparisons, not an exhaustive gradient-equivalence or convergence result. The small total-loss difference does not establish equivalent ink accuracy.

The earlier FP16-internal candidate reached 2.015× step speed in [its archived comparison](research/training/comparison96.json), but its first sampled gradient/update differences were 23.1884%/34.3202%. It was rejected as the current candidate. Its separate raw-padding 128³ feasibility result is retained as research history and does not validate the current source integration.

To evaluate the experimental source path, apply the attention patch after the two fixes and add `"pad_sdpa_heads": true` to the existing model configuration. Default behavior stays disabled. This remains a short-run training experiment; convergence and downstream ink quality have not been established.

## 100-update integrated-source trajectory

A subsequent [paired trajectory experiment](research/training/trajectory96/README.md)
completed **100 updates per arm with changing real CT crops and masks**, using
the actual integrated source with padding off/on. All 100 input hashes matched;
every optimizer-step and full-gradient finiteness check passed. Full audits ran
at five checkpoints, with sampled parameter/EMA audits between them.

Median step speed was **1.155x** native, with **15.75%** lower peak
allocated memory. Worst sampled gradient/update relative L2 differences were
**1.2039% / 3.8747%**; final sampled student/teacher parameter differences were
**0.000291% / 0.000117%**. Maximum absolute total-loss difference was
**4.57763672e-05**. These are drift measurements, not evidence of equivalent
convergence or ink accuracy. This extends the same contribution and remains
experimental and default-off.

## Scope and attribution

The two training bug fixes are independent new September contributions. The experimental training attention path extends the earlier frozen-inference padding work; it is not presented as a separate discovery of head padding or a second claim for the same earlier result. This repository demonstrates restored configuration behavior and functioning checkpointed gradients on real CT. It does not establish improved ink accuracy, newly readable text, convergence, upstream adoption or an award outcome.

Development used AI assistance. Executable regressions, exact source/input hashes, patches, numerical failures and limitations are included for review. Upstream source and adapted-file notices are retained, including the applicable Apache-2.0 and Dinovol notices; see the license and third-party notice files alongside the patches.

Nonsensitive local execution paths are intentionally retained in raw reports to preserve the original receipts. They identify the recorded runs; use the portable command arguments above for a different checkout or machine.
