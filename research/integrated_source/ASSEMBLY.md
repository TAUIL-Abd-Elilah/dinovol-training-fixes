# Reproduce the assembled FP32 training source

The public package includes the assembler and manifests. It does not bundle
the generated Villa source tree, archive, checkpoints or CT. Generate the
source from the pinned public Villa commit; the three patches are elsewhere
in this repository.

Copy assemble.py into an empty output directory, then run it with these
arguments (absolute paths are easiest):

```text
python assemble.py --source-repo /path/to/villa --config-patch /path/to/dinovol-training-fixes/config_fix/preserve_model_config.patch --checkpoint-patch /path/to/dinovol-training-fixes/checkpoint_fix/explicit_nonreentrant.patch --attention-patch /path/to/dinovol-training-fixes/research/training/integration/dinovol_training_head_padding.patch
```

It archives only tracked dinovol files at commit
f07d33be6a00d12ace7d6a9465efe17c78ed7b47, checks all extraction destinations,
applies each patch after an application check, retains upstream licenses and
records the exact resulting bytes. Existing source/archive destinations are
refused. No dependencies or model weights are installed.

The supplied manifests are receipts from the Windows execution environment.
They deliberately retain local execution paths; reproduce with newly generated
manifests for your own paths and Git newline settings. The assembled helper
and standalone helper differ only in CRLF versus LF line endings in this run.

Pass the generated ASSEMBLY_MANIFEST.json and dinovol directory to
run_training_probe.py using --integrated-manifest and --dinovol-root. That
mode verifies the entire source inventory and runs the actual patched model;
attention interception only logs calls and forwards to PyTorch. It applies no
runtime checkpoint adapter because the checkpoint fix is in the source.
