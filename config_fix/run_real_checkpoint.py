"""Reproduce model warm-start loss on real CT via actual trainer constructors.

This is an initialization/embedding correctness experiment, not a training or
ink-quality benchmark. Run separately when the GPU is free. No weights are ever
manually loaded into either trainer: only the direct reference is loaded that
way. The candidate differs solely by preserve_model_config.patch.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

from source_contract import ROOT, load_trainers, sha_file

CHECKPOINT_SHA = "e041ca870dd2570f8a44d1dd26db1197b3f74121f62023bc774fbc9d40e51a59"
PIXEL_SHA = "fb5d3659016e1bb760c608336e3ae1f356414464362021ec4e96ce392de94ad5"
SOURCE_URI = "s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr"
OUTPUT_KEYS = ("x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens")


def state_fingerprint(state, torch):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def host_guard(label, report, psutil, *, require_available_gib=2):
    snapshot = {"stage": label, "rss_bytes": psutil.Process().memory_info().rss,
                "available_bytes": psutil.virtual_memory().available}
    report.setdefault("host_memory_checks", []).append(snapshot)
    # These are pre/post stage checks, not an OS-enforced hard allocation cap.
    if snapshot["rss_bytes"] > 12 * 1024**3:
        raise MemoryError("Process exceeded the declared 12 GiB working-set guard")
    if snapshot["available_bytes"] < require_available_gib * 1024**3:
        raise MemoryError(f"Less than {require_available_gib} GiB host memory available at {label}")


def compare_state(model, expected, torch):
    actual = model.state_dict()
    if set(actual) != set(expected):
        raise ValueError("Backbone key set differs from the verified official checkpoint")
    shape_mismatches = [key for key in actual if actual[key].shape != expected[key].shape]
    dtype_mismatches = [key for key in actual if actual[key].dtype != expected[key].dtype]
    if shape_mismatches or dtype_mismatches:
        raise ValueError(f"State type/shape mismatch: {shape_mismatches}, {dtype_mismatches}")
    equal = [key for key in actual if torch.equal(actual[key], expected[key])]
    return {"keys": len(actual), "exact_keys": len(equal), "all_tensors_exact": len(equal) == len(actual),
            "nonidentical_keys": sorted(set(actual) - set(equal)),
            "state_sha256": state_fingerprint(actual, torch)}


def run_backbone(model, image, output, name, torch, np):
    """Same native backbone call, dtype, data and RNG for every arm."""
    model.eval().requires_grad_(False)
    model.to(device="cuda:0", dtype=torch.bfloat16)
    x = image.to(device="cuda:0", dtype=torch.bfloat16)
    torch.manual_seed(20260920)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        native_output = model(x, masks=None, is_training=False, view_kind="global")
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    tensors = {key: native_output[key].detach().cpu().contiguous() for key in OUTPUT_KEYS}
    if tensors["x_norm_patchtokens"].shape != (1, 4096, 864):
        raise ValueError("Unexpected patch-token shape")
    if not all(bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("Nonfinite reference/arm embedding")
    output_path = output / (name + "_embeddings.npz")
    np.savez(output_path, **{key: value.float().numpy() for key, value in tensors.items()})
    info = {"elapsed_ms_single_pass_not_benchmark": elapsed,
            "precision": "backbone/input bfloat16; no autocast, compile or backend forcing",
            "output_shapes": {key: list(value.shape) for key, value in tensors.items()},
            "embedding_sha256": state_fingerprint(tensors, torch),
            "output_npz_sha256": sha_file(output_path),
            "output_npz_encoding": "lossless float32 representation of bfloat16 outputs",
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    del native_output, x
    return tensors, info


def compare_outputs(actual, reference, torch):
    comparisons = {}
    for key, actual_value in actual.items():
        reference_value = reference[key]
        a, b = actual_value.double(), reference_value.double()
        difference = a - b
        comparisons[key] = {
            "exact": bool(torch.equal(actual_value, reference_value)),
            "exact_fraction": float((actual_value == reference_value).double().mean()),
            "max_abs_error": float(difference.abs().max()),
            "mean_abs_error": float(difference.abs().mean()),
            "relative_l2_error": float(torch.linalg.vector_norm(difference) /
                                       torch.linalg.vector_norm(b).clamp_min(1e-30)),
        }
    return comparisons


def describe_heads(trainer):
    student = trainer.model_module.student
    return {"dino_out_dim": student.dino_head.last_layer.out_features,
            "ibot_out_dim": student.ibot_head.last_layer.out_features,
            "dino_hidden_dim": student.dino_head.mlp[0].out_features,
            "ibot_hidden_dim": student.ibot_head.mlp[0].out_features,
            "dino_loss_shape": list(trainer.dino_loss.center.shape),
            "ibot_loss_shape": list(trainer.ibot_patch_loss.center.shape),
            "model_parameter_count": sum(p.numel() for p in trainer.model_module.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--villa", type=Path, required=True, help="Checkout whose entire Dinovol tree matches f07d33be")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Fixed official slim ps8 step352500 teacher")
    parser.add_argument("--volume", type=Path, required=True, help="Fixed PHerc0139 A256-cubed uint8 .npy")
    parser.add_argument("--output", type=Path, required=True, help="New folder inside this experiment directory")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must stay inside this experiment directory")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "STARTED", "scope": "Real-scroll warm-start initialization and embeddings; no training, ink labels or reading result",
              "runner_sha256": sha_file(Path(__file__)),
              "source_contract_sha256": sha_file(ROOT / "source_contract.py")}

    def save_report():
        (output / "report.json").write_text(json.dumps(report, indent=2))

    try:
        import numpy as np
        import psutil
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        host_guard("before imports/model", report, psutil, require_available_gib=8)
        native, fixed, report["source"] = load_trainers(args.villa)
        from dinovol_2.model.model import DinoVitStudentTeacher
        from dinovol_2.dataset import normalization
        report["normalization_source_sha256"] = sha_file(normalization.__file__)
        report["versions"] = {"python": sys.version, "platform": platform.platform(),
                              "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda}
        if not torch.cuda.is_available():
            raise RuntimeError("This real-CT driver requires CUDA; no implicit CPU forward")
        torch.cuda.set_per_process_memory_fraction(0.85, 0)
        report["cuda"] = {"allocator_fraction": 0.85, "device": torch.cuda.get_device_name(0),
                          "capacity_bytes": torch.cuda.get_device_properties(0).total_memory}
        report["protocol"] = {
            "constructor_device": "cpu", "embedding_device": "cuda:0", "cpu_threads": 2,
            "reference_loading": "strict direct load into training backbone for reference only",
            "trainer_loading": "native trainer constructor from authored model.pretrained_weights; no manual weight assignment",
            "precision": "exact float32 checkpoint comparison before identical bf16 evaluation",
            "head_override": "32 output, 32 hidden, 32 bottleneck, 2 layers for each head",
            "normalization": "official training robust normalization separately on A[0:128,0:128,0:128]",
            "input_augmentation": "none", "updates": 0, "heads_used_for_embedding": False,
            "cleanup": "one reference/trainer at a time; gc and empty_cache between stages",
            "host_guard": "8 GiB initially available; pre/post stage RSS checks <=12 GiB; not an OS-enforced allocation ceiling",
        }
        if args.volume.suffix.lower() != ".npy":
            raise ValueError("Volume must be a .npy array")
        raw = np.load(args.volume, allow_pickle=False)
        if raw.shape != (256, 256, 256) or raw.dtype != np.uint8:
            raise ValueError("Expected fixed 256-cubed uint8 fixture")
        pixel_digest = hashlib.sha256(raw.tobytes(order="C")).hexdigest()
        if pixel_digest != PIXEL_SHA:
            raise ValueError("Real CT fixture pixel SHA mismatch")
        crop = raw[:128, :128, :128].astype(np.float32, copy=True)
        normalizer = normalization.get_normalization("robust")
        normalized = normalizer.run(crop)
        image = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).contiguous()
        report["input"] = {"source_uri": SOURCE_URI, "voxel_size_um": 9.362,
                           "parent_box_zyx": [3840, 4096, 3712, 3968, 1344, 1600],
                           "used_box_zyx": [3840, 3968, 3712, 3840, 1344, 1472],
                           "volume_pixel_sha256": pixel_digest, "volume_file_sha256": sha_file(args.volume),
                           "normalized_input_sha256": state_fingerprint({"input": image}, torch),
                           "shape": list(image.shape)}
        del raw, crop, normalized

        digest = sha_file(args.checkpoint)
        if digest != CHECKPOINT_SHA:
            raise ValueError("Official checkpoint SHA mismatch")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("config"), Mapping):
            raise ValueError("Expected official payload with config mapping")
        model_config = dict(payload["config"]["model"])
        if (model_config["embed_dim"], model_config["depth"], model_config["num_heads"]) != (864, 24, 16):
            raise ValueError("Unexpected full backbone architecture")
        if tuple(model_config["patch_size"]) != (8, 8, 8) or model_config.get("grad_checkpointing", False):
            raise ValueError("Expected native ps8 backbone with checkpointing disabled")
        torch.manual_seed(17)
        reference_model = DinoVitStudentTeacher._build_backbone(model_config)
        teacher = payload["teacher"]
        if not isinstance(teacher, Mapping):
            raise ValueError("Expected teacher state mapping")
        expected_keys = set(reference_model.state_dict())
        if set(teacher) == expected_keys:
            state = dict(teacher)
            adapter = "exact unprefixed teacher keys; add exactly one backbone. prefix for native training checkpoint loader"
        elif teacher and all(key.startswith("backbone.") for key in teacher):
            state = {key.removeprefix("backbone."): value for key, value in teacher.items()}
            adapter = "teacher keys uniformly prefixed backbone.; preserve native training loader convention"
        else:
            raise ValueError("Unexpected checkpoint key convention")
        if set(state) != expected_keys:
            raise ValueError("Slim checkpoint does not exactly match training backbone keys")
        template = reference_model.state_dict()
        if any(not isinstance(value, torch.Tensor) or value.shape != template[key].shape or
               value.dtype != template[key].dtype for key, value in state.items()):
            raise ValueError("Slim checkpoint tensor shapes/dtypes do not match native training backbone")
        reference_model.load_state_dict(state, strict=True)
        del template, teacher, payload
        expected_fingerprint = state_fingerprint(state, torch)
        # A verified local derived file adapts only the slim serialization. It is
        # loaded by the actual native constructor, never into a trainer by hand.
        training_checkpoint = output / "verified_training_backbone.pt"
        torch.save({"teacher": {"backbone." + key: value for key, value in state.items()}}, training_checkpoint)
        report["checkpoint"] = {"original_path": str(args.checkpoint.resolve()), "original_sha256": digest,
                                "architecture": model_config, "adapter": adapter,
                                "derived_path": str(training_checkpoint), "derived_sha256": sha_file(training_checkpoint),
                                "state_keys": len(state), "state_sha256": expected_fingerprint,
                                "weights_only_original_read": True}
        report["reference_state"] = compare_state(reference_model, state, torch)
        host_guard("reference constructed", report, psutil)
        reference_output, report["reference_forward"] = run_backbone(reference_model, image, output, "reference", torch, np)
        del reference_model
        gc.collect()
        torch.cuda.empty_cache()
        save_report()

        heads = {"dino_out_dim": 32, "ibot_out_dim": 32,
                 "dino_head_hidden_dim": 32, "ibot_head_hidden_dim": 32,
                 "dino_head_bottleneck_dim": 32, "ibot_head_bottleneck_dim": 32,
                 "dino_head_nlayers": 2, "ibot_head_nlayers": 2}
        authored_model = {**model_config, **heads, "pretrained_weights": str(training_checkpoint),
                          "pretrained_backbone_only": True, "pretrained_unchunk": False}
        config = {"device": "cpu", "use_amp": False, "use_ddp": False, "warmup_steps": 0,
                  "max_iterations": 2, "batch_size": 1, "lr": 1e-4,
                  "point_supervision": {"enabled": False}, "task_eval_every": 0,
                  "model": authored_model}
        report["authored_config"] = deepcopy(config)
        for name, module in (("native", native), ("fixed", fixed)):
            print(f"Constructing {name} trainer through its actual initializer...", flush=True)
            host_guard("before " + name + " constructor", report, psutil, require_available_gib=6)
            torch.manual_seed(17)
            arm_config = {**config, "output_dir": str(output / name)}
            snapshot = deepcopy(arm_config)
            trainer = module.DinoIBOTPretrainer(arm_config)
            arm = {"effective_model_config": trainer.model_config, "heads": describe_heads(trainer),
                   "student_state": compare_state(trainer.model_module.student.backbone, state, torch),
                   "teacher_state": compare_state(trainer.model_module.teacher.backbone, state, torch)}
            assert arm_config == snapshot, "Caller configuration mutated"
            if name == "native":
                assert "pretrained_weights" not in trainer.model_config
                assert arm["heads"]["dino_out_dim"] == arm["heads"]["ibot_out_dim"] == 131072
                assert not arm["student_state"]["all_tensors_exact"]
            else:
                assert trainer.model_config["pretrained_weights"] == str(training_checkpoint)
                assert arm["heads"]["dino_out_dim"] == arm["heads"]["ibot_out_dim"] == 32
                assert arm["heads"]["dino_loss_shape"] == [1, 32]
                assert arm["heads"]["ibot_loss_shape"] == [1, 1, 32]
                assert arm["student_state"]["all_tensors_exact"] and arm["teacher_state"]["all_tensors_exact"]
            host_guard(name + " constructed", report, psutil)
            result, arm["forward"] = run_backbone(trainer.model_module.student.backbone, image, output, name, torch, np)
            arm["embedding_vs_direct_reference"] = compare_outputs(result, reference_output, torch)
            report[name] = arm
            save_report()
            if name == "fixed":
                assert all(item["exact"] for item in arm["embedding_vs_direct_reference"].values()), "Fixed embeddings not bit-identical to same-kernel direct reference"
            else:
                assert not arm["embedding_vs_direct_reference"]["x_norm_patchtokens"]["exact"]
            trainer._close_auxiliary_datasets()
            trainer._finish_wandb()
            del trainer, result
            gc.collect()
            torch.cuda.empty_cache()
            host_guard("after " + name + " cleanup", report, psutil)
        report["status"] = "PASS"
    except Exception:
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        save_report()
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
