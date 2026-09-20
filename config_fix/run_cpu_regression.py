"""Exercise the actual trainer constructors on CPU; never run a training step."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

# Must precede importing torch. It also prevents checkpoint RNG capture from
# initializing CUDA merely because a GPU is present in this workstation.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
import torch

from source_contract import ROOT, load_trainers, sha_file


def fingerprint(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def describe(trainer):
    model = trainer.model_module
    dino, ibot = model.student.dino_head, model.student.ibot_head
    return {
        "dino_out_dim": dino.last_layer.out_features,
        "ibot_out_dim": ibot.last_layer.out_features,
        "dino_hidden_dim": dino.mlp[0].out_features,
        "ibot_hidden_dim": ibot.mlp[0].out_features,
        "dino_bottleneck_dim": dino.last_layer.in_features,
        "ibot_bottleneck_dim": ibot.last_layer.in_features,
        "dino_loss_center_shape": list(trainer.dino_loss.center.shape),
        "ibot_loss_center_shape": list(trainer.ibot_patch_loss.center.shape),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "backbone_sentinel_value": float(model.student.backbone.cls_token.flatten()[0]),
        "model_sha256": fingerprint(model.state_dict()),
    }


def finish(trainer):
    trainer._close_auxiliary_datasets()
    trainer._finish_wandb()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--villa", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Regression output must stay inside its owned source folder")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "STARTED", "scope": "Synthetic CPU regression, not real-scroll usefulness evidence"}
    try:
        if torch.cuda.is_available() or torch.cuda.is_initialized():
            raise RuntimeError("CPU-only process contract failed")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        native, fixed, report["source"] = load_trainers(args.villa)
        from dinovol_2.model.model import DinoVitStudentTeacher, _materialize_backbone_config

        architecture = dict(model_type="v2", global_crops_size=16, local_crops_size=None,
                            patch_size=4, input_channels=1, embed_dim=24, depth=1,
                            num_heads=4, num_register_tokens=2, drop_path_rate=0.0)
        heads = dict(dino_out_dim=16, ibot_out_dim=24, dino_head_hidden_dim=32,
                     ibot_head_hidden_dim=40, dino_head_bottleneck_dim=8,
                     ibot_head_bottleneck_dim=12, dino_head_nlayers=2,
                     ibot_head_nlayers=2, dino_head_norm_last_layer=True)
        torch.manual_seed(31)
        source_model = DinoVitStudentTeacher({**architecture, **heads})
        with torch.no_grad():
            source_model.student.backbone.cls_token.fill_(0.375)
            source_model.student.dino_head.mlp[0].weight.fill_(0.125)
            source_model.student.ibot_head.mlp[0].weight.fill_(-0.0625)
        source_model.synchronize_teacher_from_student()
        expected = {k: v.detach().clone() for k, v in source_model.teacher.state_dict().items()}
        fixture = output / "tiny_known_weights.pt"
        torch.save({"teacher": expected}, fixture)
        del source_model
        gc.collect()
        model_config = {**architecture, **heads, "pretrained_weights": str(fixture),
                        "pretrained_backbone_only": False, "pretrained_unchunk": False}
        config = dict(device="cpu", use_amp=False, use_ddp=False, warmup_steps=0,
                      max_iterations=2, batch_size=1, lr=1e-4,
                      point_supervision={"enabled": False}, task_eval_every=0,
                      model=model_config, output_dir=str(output / "native"))
        authored_snapshot = deepcopy(config)
        report["fixture"] = {"path": str(fixture), "sha256": sha_file(fixture),
                             "backbone_sentinel": 0.375, "model_keys": len(expected)}

        # This intentionally constructs the real native default heads. Reducing
        # them through a stub would hide the configuration bug being reproduced.
        print("Constructing native trainer (requested 16/24 outputs)...", flush=True)
        torch.manual_seed(17)
        before = native.DinoIBOTPretrainer(config)
        report["before"] = describe(before)
        assert report["before"]["dino_out_dim"] == 131072
        assert report["before"]["ibot_out_dim"] == 131072
        assert report["before"]["backbone_sentinel_value"] != 0.375
        assert "pretrained_weights" not in before.config["model"]
        native_default_fingerprint = report["before"]["model_sha256"]
        effective_default_config = deepcopy(before.model_config)
        assert config == authored_snapshot, "Native input mapping unexpectedly mutated"
        finish(before)
        del before
        gc.collect()

        print("Constructing patched trainer and checking checkpoint tensors...", flush=True)
        torch.manual_seed(17)
        after_config = {**config, "output_dir": str(output / "fixed")}
        after = fixed.DinoIBOTPretrainer(after_config)
        report["after"] = describe(after)
        for branch in (after.model_module.student, after.model_module.teacher):
            actual = branch.state_dict()
            assert actual.keys() == expected.keys()
            assert all(torch.equal(actual[k], expected[k]) for k in actual)
        assert report["after"]["dino_out_dim"] == 16 and report["after"]["ibot_out_dim"] == 24
        assert report["after"]["dino_hidden_dim"] == 32 and report["after"]["ibot_hidden_dim"] == 40
        assert report["after"]["dino_bottleneck_dim"] == 8 and report["after"]["ibot_bottleneck_dim"] == 12
        assert report["after"]["dino_loss_center_shape"] == [1, 16]
        assert report["after"]["ibot_loss_center_shape"] == [1, 1, 24]
        assert not after.model_module.student.dino_head.last_layer.parametrizations.weight.original0.requires_grad
        assert after.model_config["global_crops_size"] == (16, 16, 16)
        assert after.model_config["local_crops_size"] == (16, 16, 16)
        assert after.model_config["patch_size"] == (4, 4, 4)
        assert after.model_config["num_reg_tokens"] == 2
        assert all(after.model_config[k] == v for k, v in heads.items())
        assert after.model_config["pretrained_weights"] == str(fixture)
        assert after.config["model"] is after.model_config
        assert config == authored_snapshot, "Patch mutated caller input mapping"
        roundtrip = after.save_checkpoint(1)
        saved = torch.load(roundtrip, map_location="cpu", weights_only=False)
        assert saved["config"]["model"] == after.config["model"]
        roundtrip_digest = fingerprint(after.model_module.state_dict())
        finish(after)
        del after
        gc.collect()

        restored = fixed.DinoIBOTPretrainer({**saved["config"], "output_dir": str(output / "roundtrip")})
        assert restored.load_checkpoint(roundtrip) == 1
        assert fingerprint(restored.model_module.state_dict()) == roundtrip_digest
        assert list(restored.dino_loss.center.shape) == [1, 16]
        assert list(restored.ibot_patch_loss.center.shape) == [1, 1, 24]
        report["checkpoint_roundtrip"] = "PASS: authored options and normalized backbone config retained; trainer/loss state restored"
        finish(restored)
        del restored, saved
        gc.collect()

        print("Checking strict failures and backbone-only loading...", flush=True)
        missing = {**model_config, "pretrained_weights": str(output / "missing_requested.pt")}
        try:
            fixed.DinoIBOTPretrainer({**config, "model": missing})
        except FileNotFoundError:
            report["missing_requested_checkpoint"] = "PASS: FileNotFoundError"
        else:
            raise AssertionError("A missing requested checkpoint was silently ignored")
        broken = dict(expected)
        broken.pop("backbone.cls_token")
        broken_path = output / "incomplete_known_weights.pt"
        torch.save({"teacher": broken}, broken_path)
        try:
            fixed.DinoIBOTPretrainer({**config, "model": {**model_config, "pretrained_weights": str(broken_path)}})
        except RuntimeError as error:
            assert "Missing keys" in str(error)
            report["incomplete_requested_checkpoint"] = "PASS: strict missing-key failure"
        else:
            raise AssertionError("Incomplete requested model weights were accepted")

        torch.manual_seed(23)
        backbone_only = fixed.DinoIBOTPretrainer({**config, "output_dir": str(output / "backbone_only"),
            "model": {**model_config, "pretrained_backbone_only": True, "pretrained_unchunk": True}})
        loaded = backbone_only.model_module.student.state_dict()
        assert all(torch.equal(loaded[k], expected[k]) for k in loaded if k.startswith("backbone."))
        assert not torch.equal(loaded["dino_head.mlp.0.weight"], expected["dino_head.mlp.0.weight"])
        assert list(backbone_only.dino_loss.center.shape) == [1, 16]
        report["backbone_only_option"] = "PASS: backbone restored, heads newly initialized with requested dimensions"
        finish(backbone_only)
        del backbone_only, loaded
        gc.collect()

        print("Checking unchanged default model tensors...", flush=True)
        torch.manual_seed(17)
        default = fixed.DinoIBOTPretrainer({**config, "model": effective_default_config,
                                          "output_dir": str(output / "fixed_defaults")})
        report["default_behavior"] = describe(default)
        assert report["default_behavior"]["model_sha256"] == native_default_fingerprint
        assert report["default_behavior"]["dino_loss_center_shape"] == report["before"]["dino_loss_center_shape"]
        assert report["default_behavior"]["ibot_loss_center_shape"] == report["before"]["ibot_loss_center_shape"]
        finish(default)
        del default
        gc.collect()
        report["cuda_initialized"] = torch.cuda.is_initialized()
        assert not report["cuda_initialized"]
        report["versions"] = {"torch": torch.__version__, "python": sys.version}
        report["status"] = "PASS"
    except Exception:
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["runner_sha256"] = sha_file(Path(__file__))
        (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
