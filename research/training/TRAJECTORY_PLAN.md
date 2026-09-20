# Paired trajectory protocol, specified before results

Continue the existing contribution; this is not another prize entry. Existing
Windows RTX 3090 only. Compare the same verified assembled source with
pad_sdpa_heads false/true. No runtime attention substitution. All other model,
optimizer, initialization, normalization and schedule settings remain identical.

Run 100 actual updates per arm (5 warmups + 95 timed), global96/local64, B2,
two globals and eight locals per sample, full backbone and131072-output heads.
Use the existing PHerc0139 A/B256 fixtures; generate changing deterministic
view origins and 25% masks on two global views from an independent NumPy RNG
seeded by experiment seed and update number. Store origins and input hashes at
each step so both arms must match exactly. No image augmentation beyond crops.

Use the existing exhaustive parameter/gradient/EMA audit at updates0,24,49,74,99.
At intervening updates collect the same128 equally spaced values per parameter
for update/EMA/gradient comparison. Still check every full gradient for finite
values, every optimizer state step, and that the teacher has no gradients.
Distinguish sampled audits from exhaustive audits explicitly. Timing excludes
data preparation and audits; no profiler update and no convergence claim.

Stop an arm on nonfinite loss/gradients, missing optimizer advancement, no
student/teacher changes, or failed EMA checks. Keep failures. Do not silently
change seed, precision, inputs or tolerances to obtain a successful result.
If both finish, report all100 paired loss differences and sampled gradient,
optimizer-update and accumulated parameter differences, including their worst
values and where they occur. Report initial-state equality and input equality.
No numerical-equivalence, convergence or ink-accuracy threshold is inferred
from success. Material drift must be prominently disclosed, and the option
remains experimental and disabled by default.

Publish a compact result and reproducible driver, omitting weights, large raw
arrays and repeated sample layouts. Preserve local raw receipts and hashes.
