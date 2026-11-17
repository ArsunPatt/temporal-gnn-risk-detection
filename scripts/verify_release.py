"""Verify a dated release snapshot without needing private scientific artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("results/stage5/release_manifest.json")
SIDECAR = MANIFEST.with_suffix(".sha256")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def release_members(root, checkout):
    """List physical package files, or Git's publication candidates in checkout mode."""
    if checkout:
        if not (root / ".git").exists():
            raise ValueError("--checkout requires the root of a Git checkout")
        entries = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
        )
        return {os.fsdecode(name) for name in entries.split(b"\0") if name}

    def raise_walk_error(error):
        raise error

    members = set()
    for directory, subdirectories, files in os.walk(root, onerror=raise_walk_error):
        folder = Path(directory)
        # Treat symlinked directories as entries instead of walking their targets.
        files.extend(name for name in subdirectories if (folder / name).is_symlink())
        members.update((folder / name).relative_to(root).as_posix() for name in files)
    return members


def verify_release(root=ROOT, require_private=False, *, checkout=False):
    root = Path(root)
    path = root / MANIFEST
    checksum = (root / SIDECAR).read_text().split()
    if not checksum or digest(path) != checksum[0]:
        raise ValueError("Release manifest checksum changed")
    manifest = json.loads(path.read_text())
    public = manifest["public_files_sha256"]
    private = manifest["private_artifacts_sha256"]
    required = set(public) | {MANIFEST.as_posix(), SIDECAR.as_posix()}
    allowed = required | set(private) if require_private and not checkout else required
    actual = release_members(root, checkout)
    unexpected = sorted(actual - allowed)
    missing_files = sorted(required - actual)
    if unexpected or missing_files:
        details = []
        if unexpected:
            details.append("Unexpected release files: " + ", ".join(unexpected))
        if missing_files:
            details.append("Missing release files: " + ", ".join(missing_files))
        raise ValueError("\n".join(details))
    for name in sorted(actual):
        if (root / name).is_symlink():
            raise ValueError(f"Symbolic links are not release files: {name}")
    for name, expected in public.items():
        file = root / name
        if not file.is_file() or digest(file) != expected:
            raise ValueError(f"Missing or changed public release file: {name}")
    available = missing = 0
    for name, expected in private.items():
        file = root / name
        if not file.is_file():
            missing += 1
            if require_private:
                raise FileNotFoundError(f"Private release artifact unavailable: {name}")
        else:
            if digest(file) != expected:
                raise ValueError(f"Private artifact checksum changed: {name}")
            available += 1
    original = root / manifest["original_frozen_protocol"]["path"]
    if digest(original) != manifest["original_frozen_protocol"]["sha256"]:
        raise ValueError("Original frozen protocol changed")
    return {"status":"passed", "release_date":manifest["release_date"], "public_files_verified":len(manifest["public_files_sha256"]),
            "file_membership_verified":True, "membership_scope":"git publication candidates" if checkout else "physical files",
            "private_artifacts_present_and_verified":available,"private_artifacts_absent":missing,
            "private_artifact_check_required":require_private,"independent_inference_reproduction":False,
            "scope":"Dated release integrity only; cannot retroactively establish original complete code freeze"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", action="store_true",
                        help="Verify Git publication candidates and check declared private artifacts separately")
    parser.add_argument("--require-private", action="store_true", help="Require every declared private artifact")
    args=parser.parse_args()
    try:
        print(json.dumps(verify_release(require_private=args.require_private, checkout=args.checkout), indent=2))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
