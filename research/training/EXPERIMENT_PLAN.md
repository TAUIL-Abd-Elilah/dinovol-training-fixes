# Dinovol training attention experiment — 20 September 2026

This protocol is written before the first full trainer run. The earlier public
inference optimization did not establish training safety or speed. This work is
an extension of that contribution, not an independent attention invention.

## Hypothesis and controls

Zero-extending a 54-wide attention head to 56, retaining 54**-0.5 scale and
slicing the output, may enable an existing fused CUDA SDPA forward/backward on
this Windows RTX 3090. Compare unchanged official training, padded training, and
a forced-math padded control. Do not change the published inference-only helper.

Use the official DinoIBOTPretrainer.train_step including DINO, iBOT, KoLeo,
backward, gradient clipping, AdamW, and EMA. Checkpoint initialization is a
strictly loaded official step352500 teacher backbone with newly initialized
projection heads and optimizer; it is not a resumed full training checkpoint.
Both branches receive identical initial weights, RNG seeds, real CT views,
masks, loss settings, schedule position, AMP and checkpointing configuration.

Use already verified PHerc0139 cubes A and B first. The scan is 9.362um and
different from the high-resolution source training scan. These are a compute
and numerical test, not a quality evaluation, reading result or held-out
accuracy experiment. No download, rented compute or additional dependency.

## Staged gate

1. Confirm float64 math output and Q/K/V gradients; screen full-length GPU
   fp16 and bf16 finite gradients. Record numerical differences without
   describing fused kernels as bitwise equivalent. (Completed in
   autograd_screen.json before this plan; float64 output and gradients exact.)
2. Feasibility: B=2, two128-cube global views and two64-cube local views per
   source,128/4096 masked tokens per global view, official full heads,
   gradient checkpointing, fp16 autocast. This reduces local count and masking
   from the source recipe; it is only a feasibility screen. Source recipe has
   B=3, eight locals, checkpointing off and different masking distribution.
3. If feasible, select a representative B=2/eight-local recipe with masks in
   the official10–50% range before measuring final speed. Preserve full model
   and heads. Keep all controls identical; if the original cannot fit, disclose
   that rather than deriving an unsupported finite speedup.
4. Require finite loss/gradients, actual nonzero optimizer parameter changes,
   and EMA changes. A skipped AMP optimizer step is not a successful update.
   Record loss scaling and optimizer step state. A padded change must not
   silently skip updates the native comparison performs.
5. Quantify first-step losses, representative gradients spanning model layers,
   parameter-update differences and subsequent training loss trajectory where
   resources permit. Stop a broad release if differences are unacceptably
   large or training is unstable. Short trajectory checks cannot establish
   convergence or final ink quality.
6. Synchronize CUDA around complete train_step timing; compare fresh processes
   with identical warmups and measured counts. Report peak allocated and
   reserved memory, exact recipe, current hardware/software, and exclusions.
   Profile a separate step; do not count its time in the timing median.

Only publish measured useful results with reproducible code, limitations and
prior-art attribution. Publish no award guarantee, and do not count the same
Dinovol improvement as several independent prize entries.

## Feasibility findings and next protocol (before representative results)

The first unchanged run failed before attention: the official checkpoint calls
pass RoPE keyword arguments while using PyTorch's implicit reentrant mode,
which rejects those keywords. All subsequent arms use the same explicit
non-reentrant checkpoint adapter. Its standalone fix and gradient regression
are separate from the attention change.

After that correction, native attention failed in backward allocating another
4.01 GiB with the initial 85% allocator cap. The first padded run returned from
train_step but its post-run audit failed because float32 linspace rounded a
large parameter's final sample index out of bounds. This was a bug in our audit,
not a completed successful receipt; indices now use exact integer arithmetic.

Next, test B2, eight64-cube local views, two128-cube global views, and fixed25%
masking on two of four global views (matching the official masking probability
and range, with a deterministic pattern). Use a95% allocator cap for both arms
to give the native comparison access to the available GPU memory. Keep failed
receipts and do not infer a finite speedup from an OOM.

If128-cube native attention cannot complete, use a declared96-cube global-view
comparison for numerical/timing evidence. Keep the two resolutions separate:
the smaller comparison cannot establish the missing128-cube training parity.
These are resource-feasibility adjustments, not selection by downstream quality.

## Numerical hold after the paired96 experiment

The seven-update paired96 experiment completed, including the separate profiler
update. Median timed train_step fell from3038.36 to1507.82ms. However, the first
step's58,240 sampled post-clipping gradients differed by23.19% relative L2, and
its sampled AdamW update by34.32%. Identical initial sampled parameters and
nearly identical losses are not sufficient evidence of safe training equivalence.
The fp16 fused training candidate is therefore held as experimental, default-off.

Next diagnostic, specified before its result: run one96-cube step using
FP32 attention internally (q/k/v promoted from the same fp16 tensors, output cast
back to fp16, original scale and padding preserved). Compare against the paired
native run's first step. This tests a possible accumulation explanation; it is
not an already validated fix or a new pre-registered headline benchmark. Also
retain a forced-math padding control and assess the identified bugs independently.
