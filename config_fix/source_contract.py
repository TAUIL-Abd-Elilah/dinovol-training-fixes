"""Pinned-source patch construction and isolated trainer loading."""
from __future__ import annotations

import ast
import difflib
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parent
PIN = "f07d33be6a00d12ace7d6a9465efe17c78ed7b47"
TARGET = "dinovol/dinovol_2/pretrain.py"
OLD = '        self.model_config = _materialize_backbone_config(self.config["model"])\n'
NEW = '''        self.model_config = dict(self.config["model"])
        self.model_config.update(_materialize_backbone_config(self.model_config))
'''


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_source(villa):
    villa = Path(villa).resolve()
    # Accept another checkout HEAD only when its entire tracked Dinovol tree is
    # identical to the pin, and its Dinovol subtree has no local/untracked edits.
    subprocess.run(["git", "-C", str(villa), "diff", "--exit-code", PIN, "--", "dinovol"],
                   check=True, capture_output=True)
    status = subprocess.check_output(["git", "-C", str(villa), "status", "--porcelain",
                                      "--untracked-files=all", "--", "dinovol"], text=True)
    if status.strip():
        raise ValueError("Dinovol source subtree is dirty: " + status)
    original = subprocess.check_output(["git", "-C", str(villa), "show", PIN + ":" + TARGET], text=True)
    if original.count(OLD) != 1:
        raise ValueError("Pinned initializer anchor did not occur exactly once")
    edited = original.replace(OLD, NEW, 1)
    ast.parse(edited)
    patch_text = "diff --git a/" + TARGET + " b/" + TARGET + "\n" + "".join(difflib.unified_diff(
        original.splitlines(keepends=True), edited.splitlines(keepends=True),
        fromfile="a/" + TARGET, tofile="b/" + TARGET))
    patch_path = ROOT / "preserve_model_config.patch"
    patch_path.write_text(patch_text, newline="\n")
    subprocess.run(["git", "-C", str(villa), "apply", "--check", str(patch_path)],
                   check=True, capture_output=True)
    files = [TARGET, "dinovol/dinovol_2/model/model.py", "dinovol/dinovol_2/model/dinov2_eva.py",
             "dinovol/dinovol_2/loss/dino_clstoken_loss.py", "dinovol/dinovol_2/loss/ibot_patch_loss.py",
             "dinovol/README.md", "dinovol/dinovol_2/configs/finetune_ink_emb.json",
             "dinovol/LICENSE", "dinovol/LICENSES/Apache-2.0.txt"]
    receipt = {"base_commit": PIN,
               "checkout_head": subprocess.check_output(["git", "-C", str(villa), "rev-parse", "HEAD"], text=True).strip(),
               "dinovol_tree_identical_to_pin": True, "source_clean_verified": True,
               "source_sha256": {p: sha_file(villa / p) for p in files},
               "patched_pretrain_sha256": hashlib.sha256(edited.encode()).hexdigest(),
               "patch_sha256": sha_file(patch_path), "patch_apply_check": "PASS"}
    (ROOT / "SOURCE_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    return villa, original, edited, receipt


def load_trainers(villa):
    villa, original, edited, receipt = prepare_source(villa)
    sys.path.insert(0, str(villa / "dinovol"))
    native = importlib.import_module("dinovol_2.pretrain")
    if Path(native.__file__).resolve() != (villa / TARGET).resolve():
        raise ValueError("Trainer source shadowed by another import")
    fullname = "dinovol_2.pretrain_config_preserved"
    candidate = types.ModuleType(fullname)
    candidate.__file__ = str(ROOT / "pretrain_fixed_in_memory.py")
    candidate.__package__ = "dinovol_2"
    sys.modules[fullname] = candidate
    exec(compile(edited, candidate.__file__, "exec"), candidate.__dict__)
    return native, candidate, receipt
