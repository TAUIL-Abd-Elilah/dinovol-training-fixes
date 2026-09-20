# Preserve authored Dinovol model options before trainer construction

The native trainer replaces the entire authored `model` mapping with a backbone-only mapping before it builds the student/teacher and losses. This discards the supported warm-start checkpoint path, backbone-only/unchunk switches, and head settings. The result can be a randomly initialized backbone and default 131072-way heads even when the user requested pretrained weights and smaller heads.

This is specifically the model warm-start path. The later `resume_from` / automatic resume path is separate; this report does not claim that all checkpoint resumption is broken.

## Source and actual use

- Tested source: `f07d33be6a00d12ace7d6a9465efe17c78ed7b47`. The available checkout is `8d6e1e9065976f44442c88d1de645e8f711e7400`; its complete clean Dinovol subtree is identical to that pin. `SOURCE_RECEIPT.json` records both facts and file hashes.
- Filtering happens in [pretrain.py:124–127](https://github.com/ScrollPrize/villa/blob/f07d33be6a00d12ace7d6a9465efe17c78ed7b47/dinovol/dinovol_2/pretrain.py#L124), followed by model construction at line 180 and loss construction at lines 198–208.
- The whitelist is [model.py:228–260](https://github.com/ScrollPrize/villa/blob/f07d33be6a00d12ace7d6a9465efe17c78ed7b47/dinovol/dinovol_2/model/model.py#L228). The model itself supports checkpoint loading at lines 397–404 and head overrides at lines 433–462.
- The [authored README refinement recipe](https://github.com/ScrollPrize/villa/blob/f07d33be6a00d12ace7d6a9465efe17c78ed7b47/dinovol/README.md#L66) and [finetune_ink_emb.json](https://github.com/ScrollPrize/villa/blob/f07d33be6a00d12ace7d6a9465efe17c78ed7b47/dinovol/dinovol_2/configs/finetune_ink_emb.json#L70) request the dropped `pretrained_weights` / `pretrained_backbone_only` options.
- The intended local workflow is warm-starting the official full-size Dinovol teacher before a bounded real-scroll training experiment. The full historical training checkpoint is not required to reproduce the model configuration loss: a verified slim teacher is sufficient for its supported backbone-only path.

## Minimal change

`preserve_model_config.patch` copies the authored mapping, then overlays `_materialize_backbone_config` results. Normalized backbone sizes, aliases and RoPE options retain their established precedence. Nonbackbone model options survive. The model's own `_build_backbone` still filters backbone arguments before calling Eva.

The patch changes two lines in the initializer. `source_contract.py` checks the source contract, creates the patch, verifies `git apply --check`, and loads the candidate initializer in memory under a separate module name. It does not edit the Villa checkout.

## CPU regression: passed

`cpu_regression_01/report.json` was produced by the actual native and candidate `DinoIBOTPretrainer` constructors on CPU, using Python 3.14.6 and Torch 2.13.0+cu126. CUDA was never initialized. No stub replaces the model, heads, losses or loader.

| Observed property | Native | Patched |
|---|---:|---:|
| Requested DINO output width 16 | 131072 | 16 |
| Requested iBOT output width 24 | 131072 | 24 |
| Requested hidden widths 32 / 40 | 2048 / 2048 | 32 / 40 |
| Checkpoint backbone sentinel 0.375 | 0.0074531557 | 0.375 |
| Model parameters in the tiny regression architecture | 187927672 | 24320 |
| Requested checkpoint tensors restored | Ignored | All 40, exact in both branches |

The parameter-count difference illustrates silently ignored head settings; it is not a performance or model-quality claim. Additional checks passed for missing and incomplete requested checkpoint errors, backbone-only loading, unchunk option preservation, head/loss shape consistency, caller mapping immutability, size/alias normalization, native checkpoint save/load configuration roundtrip, and exact preservation of default model tensors for the same seed.

Reproduce in this folder using a new output directory:

```powershell
& 'C:/Users/PC/miniconda3/envs/vesuvius/python.exe' run_cpu_regression.py --villa 'D:/Competition/Vesuvius progress prizes/villa/_worktrees/ink9um' --output '<absolute path inside this folder>/cpu_regression_02'
```

## Real-scroll reproduction: passed on the full pretrained backbone

`run_real_checkpoint.py` completed successfully on the RTX3090; see `real_01/report.json`. Before the fix24/463 state entries matched the checkpoint (initial/default values); afterward463/463 matched in both student and teacher. Class, register and patch embeddings on the real crop matched the directly loaded reference exactly after the fix. The native patch-embedding relative L2 error was1.42646. The script verifies the fixed official slim teacher SHA `e041ca870dd2570f8a44d1dd26db1197b3f74121f62023bc774fbc9d40e51a59` and PHerc0139 A256-cubed pixel SHA `fb5d3659016e1bb760c608336e3ae1f356414464362021ec4e96ce392de94ad5`.

The driver uses the full 864-wide, 24-block, 16-head ps8 backbone and the official training robust normalizer on A[0:128,0:128,0:128]. It verifies every checkpoint key, shape and dtype before deriving the `backbone.`-prefixed serialization required by the native training loader. Only the direct reference receives a manual strict state load. Both actual trainer constructors receive the same requested checkpoint configuration; the candidate changes only the two-line preservation patch. The small 32-output heads make the requested configuration explicit; the native constructor is allowed to demonstrate its real default head allocation.

It measures exact initialized state agreement for student and teacher, and native backbone class/register/patch embeddings against the direct reference. All arms use bf16 evaluation, identical input and native attention. The original and patched backbones are never patched or manually loaded after construction. It records hashes, configuration, versions, saved embeddings, numerical differences and memory. A passing patched arm requires exact tensor restoration and bit-identical embeddings to the same-kernel reference.

```powershell
& 'C:/Users/PC/miniconda3/envs/vesuvius/python.exe' run_real_checkpoint.py --villa '<Villa checkout>' --checkpoint '<official slim .pt>' --volume '<fixed A256.npy>' --output '<absolute new directory inside this folder>'
```

Run when the GPU is free. The process uses two CPU threads, a CUDA allocator fraction of 0.85, sequential model cleanup, and pre/post-stage host memory guards. These guards are not an operating-system hard memory limit. This validates initialization and embeddings on one real crop; it does not train a model, establish downstream ink quality, recover text, or establish prize value.

## Limited duplicate check and license

On 2026-09-20 current official main was `23c3d759e44eb310c4e507beea71228a77904c88`. Its retrieved `pretrain.py` still replaces the complete mapping in the same way. The checkpoint loader and native training SDPA call remain present. Limited GitHub issue/PR searches did not identify this defect or training head-padding work; `OVERLAP_AUDIT.json` records queries, unrelated hits and rate-limit failures. This does not establish uniqueness or exclude private/in-progress work.

The training files carry Meta Apache-2.0 notices. Retain those notices and `dinovol/LICENSES/Apache-2.0.txt` with any copied source or patch distribution; do not inherit the unrelated consumer package's blanket MIT labeling. No upstream PR or external message was created. Villa's contribution guidelines require a real-scroll reproduction and human-written relevance commentary for an LLM-assisted PR; the parent task handles any subsequent publication decision.
