Original Dinovol source is owned by its upstream contributors. The retained
MIT license and UPSTREAM_THIRD_PARTY_NOTICES.md describe that source. Patches
to dinovol_2/pretrain.py and dinovol_2/model/model.py modify DINOv2-derived
code, copyright Meta Platforms, Inc. and affiliates, under Apache-2.0.
Patches to dinovol_2/model/dinov2_eva.py modify code adapted from Dynamic
Network Architectures, copyright the Division of Medical Image Computing,
German Cancer Research Center (DKFZ), also under Apache-2.0. See
LICENSES/Apache-2.0.txt. Upstream headers remain intact when applying patches.

New diagnostic scripts and helper code in this repository are available under
the included MIT license. Development and validation used AI assistance.

Zero-padding attention heads is established prior art. This project applies
existing PyTorch attention and checkpointing capabilities to a specific
Dinovol workflow. It does not introduce a new attention algorithm. Its earlier
inference-only work is documented at:
https://github.com/TAUIL-Abd-Elilah/dinovol-attention-diagnostic

Checkpoint weights, CT inputs, source archives and large raw profiler traces
are not redistributed here. Reports identify the original public inputs and
their hashes. Follow the original data/model and upstream software licenses
when obtaining them.
