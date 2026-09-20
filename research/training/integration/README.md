# Experimental FP32 Dinovol training attention integration

This default-off source patch adds `model.pad_sdpa_heads` to the official
`dinovol/dinovol_2` training backbone at Villa revision
`f07d33be6a00d12ace7d6a9465efe17c78ed7b47`. It forwards the option through
`Eva`, `EvaBlock`, and `EvaAttention`, and installs
`dinovol_2/model/training_sdpa_padding.py` beside them. It does not modify the
separate `vesuvius` inference consumer.

The helper promotes eligible FP16 Q/K/V to FP32, preserves the original
attention scale while zero-extending 54-wide heads to 56, and disables autocast
for the attention call. It removes the two added output channels and casts the
output back to FP16. Cast, pad, and slice retain ordinary PyTorch autograd.
Head padding is established prior art; this is a
narrow application to the measured Dinovol training path, not a new attention
algorithm. Numerical changes from a different backend remain possible. This
package alone makes no convergence or general training-quality claim.

This remains an experimental candidate. Seven real-CT updates at 96-cubed
completed, with 1.202x median training-step throughput and 15.8% lower peak
allocated memory than native attention. The 58,240 sampled post-clipping
gradients differed by 0.128-0.422% relative L2; sampled AdamW updates differed
by 2.956% on the first update and 0.312-0.503% thereafter. This is short-run
numerical evidence, not convergence or ink-quality validation. See
[the comparison](../comparison96_fp32.json).

The actual assembled source patches also completed seven 128-cubed updates at
batch two with eight64-cube local views and full131072-output heads, using
11.62 GiB peak allocated memory and 7.417 seconds median per timed step.
The same-size native comparison ran out of memory; no finite speedup is claimed
for that size. See [the source integration receipt](../integrated128_fp32_01/report.json).

The prior FP16
attention integration is frozen in the sibling `integration_fp16` folder with
its original patch and receipts. In the parent's 96-cubed first-step comparison,
that FP16 path showed sampled gradient relative L2 difference about 0.232 and
AdamW update difference about 0.343 despite finite-step checks. It is not a
recommended training replacement. Both the rejected path and the first-step
FP32 diagnostic remain available as research history. The two spatial sizes
are separate experiments; 96-cubed numerical results cannot establish the
missing same-size128-cubed native comparison.

## Enable explicitly

Apply [dinovol_training_head_padding.patch](dinovol_training_head_padding.patch)
from a clean Villa checkout root at the pinned revision:

```text
git apply --check /path/to/dinovol_training_head_padding.patch
git apply /path/to/dinovol_training_head_padding.patch
```

Add this one key to the existing training model configuration, retaining the
rest of the recipe:

```yaml
model:
  pad_sdpa_heads: true
```

The omitted and explicit `false` settings call the original SDPA branch
directly. This option creates no parameters or persistent buffers, and existing
state dictionaries load strictly with either setting. Non-boolean values are
rejected. It applies to both student and teacher backbones built from the model
configuration.

This patch is separate from the three-call `use_reentrant=False` checkpoint
compatibility fix and the trainer configuration-preservation fix. It does not
repair or silently incorporate either one. A recipe enabling the affected
activation-checkpoint path still needs the compatibility fix on this PyTorch
version. Changing pretrained-loading behavior remains a separate concern.

## Narrow runtime contract

The helper considers padding only for eager CUDA float16, dense four-dimensional
Q/K/V with matching devices/dtypes/batch/head dimensions, unit last stride,
nonempty dimensions, and all three head widths exactly 54. Attention masks,
nonzero dropout, GQA, CPU, other dtypes/widths, tracing, scripting, and compilation
delegate to native SDPA. Autograd is allowed; dropout remains the caller's
original value and is never silently disabled.

It queries only enabled Flash, efficient, and cuDNN capabilities. If the
original FP16 inputs already have an eligible fused backend, capability APIs are
unavailable, or padded FP32 inputs have no eligible fused backend, it delegates with
the original tensors and original options. No backend is forced or globally
enabled. Allocation and actual kernel execution failures are not swallowed.

For isolated benchmark interception, call
`padded_training_sdpa(q, k, v, enabled=True, native=original_sdpa, ...)`.
The explicit native callable avoids recursion if the benchmark intercepts
`torch.nn.functional.scaled_dot_product_attention`. Normal source integration
does not monkeypatch PyTorch and uses the default native callable.

## Local verification

Run with the existing environment providing Dinovol dependencies:

```text
python make_training_integration.py --villa /path/to/villa
python check_training_integration.py --villa /path/to/villa
```

The generator checks the tracked `dinovol_2` subtree against the pin, creates
the patch, validates Python syntax, and runs `git apply --check`. It never edits
the source checkout. [PATCH_RECEIPT.json](PATCH_RECEIPT.json) records hashes.

The [CPU receipt](CPU_RECEIPT.json) records 18 passing synthetic integration
cases: v1/v2, plain/chunked/unchunked chunk class, and omitted/false/true settings.
Each verifies double materialization of configuration, every block's option,
unchanged state keys, strict loading, exact outputs, and exact input/parameter
gradients against the original CPU backbone. It also verifies that disabled
settings never call the helper.

Eight helper cases check original-tensor delegation for disabled, already
supported, missing capability, unsupported candidate, capability-error, and
no-enabled-backend paths. Two of these cases mock only device/capability gates
to compare FP16-input CPU math outputs and Q/K/V gradients against unpadded FP32
attention, for default and explicit scales at rtol 1e-3 and atol 1e-4. They check
that the actual attention callable receives FP32 56-wide Q/K/V, CPU autocast is
disabled even inside an outer autocast context, and output/gradient dtypes
return to FP16 with nonzero gradients. Those mocks are not evidence of CUDA backend
availability or GPU numerical equivalence. CUDA was never initialized by these
tests. GPU validation belongs to the separate real-CT training experiment.

## License and attribution

New helper and tooling use the included [MIT terms](LICENSE). The patch retains
upstream source headers. Dinovol's `model.py` derives from Meta DINOv2;
`dinov2_eva.py` derives from dynamic-network-architectures, copyright 2022
Division of Medical Image Computing, German Cancer Research Center (DKFZ),
Heidelberg, Germany. Their original [Apache-2.0 terms](LICENSES/Apache-2.0.txt)
remain applicable. [Original Dinovol notices](UPSTREAM_THIRD_PARTY_NOTICES.md)
and referenced license texts are retained. Dependencies are imported from the
supplied checkout rather than bundled here.
