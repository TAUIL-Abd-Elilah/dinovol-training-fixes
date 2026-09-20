# Explicit non-reentrant Dinovol activation checkpointing

This is a three-line compatibility fix for the official Dinovol backbone: pass
`use_reentrant=False` to its three `torch.utils.checkpoint.checkpoint` calls.
At Villa commit `f07d33be6a00d12ace7d6a9465efe17c78ed7b47`, enabling
`grad_checkpointing` passes RoPE keyword arguments to checkpoint. The installed
PyTorch 2.13.0+cu126 defaults to its reentrant implementation and rejects these
arguments before the first attention call. Explicit non-reentrant checkpointing
supports the existing call contract. The default `grad_checkpointing=False`
behavior is unchanged.

[The patch](explicit_nonreentrant.patch) changes only
`dinovol/dinovol_2/model/dinov2_eva.py` at the plain EVA, chunked EVA, and
unchunked `EvaWithChunking` call sites. It does not change model configuration,
attention, pretrained weights, losses, or optimizer behavior. No upstream
checkout was edited to create these artifacts.

## CPU regression

`verify_cpu.py` checks the entire tracked `dinovol_2` subtree against the stated
pin, constructs the exact three-line patch, and runs `git apply --check` without
applying it. It evaluates the official original module with checkpointing off
and an in-memory patched module with checkpointing on, using strictly identical
weights, inputs, masks, and random seeds. It also records the original keyword
argument failure with checkpointing enabled.

The six synthetic CPU cases cover v1 axial RoPE and v2 mixed RoPE, each with
plain EVA, two chunks, and the unchunked `EvaWithChunking` path. Every case uses
two seeds, training mode, stochastic depth 0.3, shifted/jittered/rescaled RoPE,
float32 math attention, batch two, 16-cubed inputs, four blocks, width 48, two
heads, patch size four, four register tokens, and fixed masks. They compare
every output, the complete input gradient, every parameter gradient including
the matching absence of gradients, and RNG state after backward. Tolerances
were fixed at rtol 1e-5 and atol 1e-7 before execution. These are small synthetic
regressions, not real CT, full training, or quality evidence.

All six cases and both seeds passed in [cpu_01/report.json](cpu_01/report.json).
All 1,092 output/gradient comparisons were bitwise identical or matched absent
gradients; the maximum absolute difference was zero. The original failure was
reproduced in every case. CUDA was never initialized.
The report contains per-output and per-parameter comparison measurements,
source and patch hashes, exact configurations, and runtime versions.

Run from this folder with the existing environment that provides Dinovol's
dependencies; use a new output folder each time:

```text
python verify_cpu.py --dinovol-root /path/to/villa/dinovol --output cpu_rerun
```

To review application against a separate clean Villa checkout at the pinned
revision, run from that checkout root:

```text
git apply --check /path/to/explicit_nonreentrant.patch
git apply /path/to/explicit_nonreentrant.patch
```

The CPU check makes no speed, memory reduction, convergence, or prize-readiness
claim. Full-size real-CT validation is a separate experiment run by the parent
task and is not included in this synthetic result.

## Attribution

The patched Dinovol file identifies itself as adapted from
[dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures).
Copyright 2022 Division of Medical Image Computing, German Cancer Research
Center (DKFZ), Heidelberg, Germany. It retains its upstream
[Apache-2.0 notice](LICENSES/Apache-2.0.txt); Dinovol modifications are covered
by its [MIT notice](LICENSES/Dinovol-MIT.txt) where permitted.
[The original Dinovol notices](UPSTREAM_THIRD_PARTY_NOTICES.md) are retained
verbatim. Those notices also describe dependencies imported from the supplied
checkout; this folder does not redistribute their implementation files.
The new regression script is provided under the included MIT terms.
