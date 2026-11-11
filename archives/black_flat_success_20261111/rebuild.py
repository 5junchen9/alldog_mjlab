"""Verify and rebuild the frozen Black release, then apply type-only overlays."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def join(root, names, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        for name in names:
            with (root / name).open("rb") as source:
                shutil.copyfileobj(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--exact-frozen", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError("Use a new destination")
    layout = json.loads((root / "CORE_LAYOUT.json").read_text())
    parts = json.loads((root / "GITHUB_BINARY_PARTS.json").read_text())
    for name, record in parts.items():
        path = root / name
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
        check(path, record["sha256"])
    check(root / layout["evidence_core"]["path"], layout["evidence_core"]["sha256"])
    destination.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="black_rebuild_") as temporary:
        source_bundle = Path(temporary) / "source.tar.gz"
        join(root, layout["source"]["parts"], source_bundle)
        check(source_bundle, layout["source"]["sha256"])
        for bundle in [source_bundle, root / layout["evidence_core"]["path"]]:
            with tarfile.open(bundle, "r:gz") as archive:
                archive.extractall(destination, filter="data")
    for model in layout["models"].values():
        target = destination / model["restore_path"]
        join(root, model["parts"], target)
        check(target, model["sha256"])
    records = json.loads((root / "CORE_RESTORED_FILES.json").read_text())
    for name, record in records.items():
        check(destination / name, record["sha256"])
    if not args.exact_frozen:
        overlay = root / "publication_overlay"
        for source in overlay.rglob("*"):
            if source.is_file():
                shutil.copy2(source, destination / source.relative_to(overlay))
    for name in ["run_local.sh", "cpu_policy_check.py", "verify_raw.py"]:
        shutil.copy2(root / name, destination / name)
    print(
        json.dumps(
            {
                "verified_frozen_files": len(records),
                "publication_overlay_applied": not args.exact_frozen,
                "omitted_raw_files": layout["omitted_raw_files"],
                "training_started": False,
            }
        )
    )


if __name__ == "__main__":
    main()
