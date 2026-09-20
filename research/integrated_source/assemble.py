"""Archive pinned tracked Dinovol source, then apply three explicit local patches.

No checkout/worktree mutations, installs, imports of model code, or CUDA calls.
Run in a fresh directory containing this script; an existing dinovol is refused.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

PIN = "f07d33be6a00d12ace7d6a9465efe17c78ed7b47"
ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def inventory():
    return {p.relative_to(ROOT).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted((ROOT / "dinovol").rglob("*")) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--config-patch", required=True, type=Path)
    parser.add_argument("--checkpoint-patch", required=True, type=Path)
    parser.add_argument("--attention-patch", required=True, type=Path)
    args = parser.parse_args()
    if (ROOT / "dinovol").exists() or (ROOT / "base_dinovol.tar").exists():
        raise FileExistsError("Refusing to replace an existing source snapshot/archive")
    patches = [args.config_patch.resolve(), args.checkpoint_patch.resolve(), args.attention_patch.resolve()]
    if not all(p.is_file() for p in patches):
        raise FileNotFoundError("All three patches must already exist")
    source = args.source_repo.resolve()
    resolved_pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", PIN], text=True).strip()
    if resolved_pin != PIN:
        raise ValueError("Unexpected source commit")
    archive = ROOT / "base_dinovol.tar"
    subprocess.run(["git", "-C", str(source), "archive", "--format=tar", "--output", str(archive),
                    PIN, "--", "dinovol"], check=True)
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or not name.parts or name.parts[0] != "dinovol"
                    or ".." in name.parts or "\\" in member.name or ":" in member.name
                    or not (member.isfile() or member.isdir())):
                raise ValueError("Unsafe or non-ordinary archive member: " + member.name)
            destination = (ROOT / Path(*name.parts)).resolve()
            if not destination.is_relative_to(ROOT / "dinovol"):
                raise ValueError("Archive destination escapes the new Dinovol subtree")
        bundle.extractall(ROOT, members=members, filter="data")
    base = inventory()
    (ROOT / "BASE_FILES.json").write_text(json.dumps(base, indent=2))
    patch_folder = ROOT / "patches"
    patch_folder.mkdir(exist_ok=False)
    # Prevent git from discovering any repository above the isolated tree.
    environment = os.environ.copy()
    environment["GIT_CEILING_DIRECTORIES"] = str(ROOT.parent)
    environment["AGENTS_AGENT_MODE"] = "1"
    environment.pop("GIT_DIR", None)
    environment.pop("GIT_WORK_TREE", None)
    probe = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT,
                           env=environment, capture_output=True, text=True)
    if probe.returncode == 0:
        raise RuntimeError("Isolated patch application unexpectedly found a Git worktree")
    manifest = {"status": "STARTED", "base_commit": PIN, "base_subtree": "dinovol",
                "base_subtree_object": subprocess.check_output(
                    ["git", "-C", str(source), "rev-parse", PIN + ":dinovol"], text=True).strip(),
                "archive_sha256": sha(archive), "base_file_count": len(base),
                "base_files_manifest_sha256": sha(ROOT / "BASE_FILES.json"),
                "assemble_script_sha256": sha(Path(__file__)),
                "patches": [], "git_worktree_created": False, "model_imported": False,
                "installs": False, "cuda_called": False}
    try:
        for number, (label, patch) in enumerate(zip(("config", "checkpoint", "attention"), patches), 1):
            copied = patch_folder / f"{number:02d}_{label}.patch"
            shutil.copyfile(patch, copied)
            checked = subprocess.run(["git", "apply", "--check", str(copied)], cwd=ROOT, env=environment,
                                     capture_output=True, text=True, check=True)
            applied = subprocess.run(["git", "apply", str(copied)], cwd=ROOT, env=environment,
                                     capture_output=True, text=True, check=True)
            manifest["patches"].append({"order": number, "label": label,
                                         "path": copied.relative_to(ROOT).as_posix(), "sha256": sha(copied),
                                         "apply_check": "PASS", "applied": True,
                                         "warnings": checked.stderr + applied.stderr})
        final = inventory()
        added, deleted = sorted(set(final) - set(base)), sorted(set(base) - set(final))
        changed = sorted(p for p in base.keys() & final.keys() if base[p] != final[p])
        if added != ["dinovol/dinovol_2/model/training_sdpa_padding.py"] or deleted:
            raise ValueError("Unexpected source additions/deletions")
        if changed != sorted(["dinovol/dinovol_2/pretrain.py", "dinovol/dinovol_2/model/model.py",
                              "dinovol/dinovol_2/model/dinov2_eva.py"]):
            raise ValueError("Unexpected modified source files: " + repr(changed))
        for path in base:
            if "LICENSE" in path or path.endswith("THIRD_PARTY_NOTICES.md"):
                if base[path] != final[path]:
                    raise ValueError("An original license/notice changed: " + path)
        with tarfile.open(archive, "r:") as bundle:
            for path in changed:
                original = bundle.extractfile(path).read().splitlines()[:3]
                if (ROOT / path).read_bytes().splitlines()[:3] != original:
                    raise ValueError("Original attribution header changed: " + path)
        python_files = sorted((ROOT / "dinovol").rglob("*.py"))
        for path in python_files:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        (ROOT / "FINAL_FILES.json").write_text(json.dumps(final, indent=2))
        manifest.update({"status": "PASS", "final_file_count": len(final),
                         "changed_files": changed, "added_files": added, "deleted_files": deleted,
                         "final_files_manifest_sha256": sha(ROOT / "FINAL_FILES.json"),
                         "source_python_ast": "PASS", "python_file_count": len(python_files),
                         "original_licenses_and_modified_source_headers_preserved": True,
                         "dinovol_root_relative": "dinovol",
                         "runtime_validation": "Pending: AST and patch applicability only; no model imports or GPU execution"})
    finally:
        (ROOT / "ASSEMBLY_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
