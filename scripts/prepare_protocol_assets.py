"""Restore losslessly packaged protocol metadata. No downloads or model imports."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).absolute().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "assets/protocol_bundle_manifest.json").read_text(encoding="utf-8"))
    archive = root / "assets/protocol_assets.zip"
    if sha(archive.read_bytes()) != manifest["archive_sha256"]:
        raise ValueError("Protocol archive SHA-256 mismatch")
    records = {r["path"]: r for r in manifest["files"]}
    if len(records) != len(manifest["files"]):
        raise ValueError("Duplicate metadata member")
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != set(records) or z.testzip() is not None:
            raise ValueError("Protocol ZIP inventory or CRC differs")
        # Verify every member and existing destination before changing any file.
        for name, record in records.items():
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or p.parts[0] != "assets" or "\\" in name:
                raise ValueError("Unsafe protocol member")
            target = (root / name).resolve()
            target.relative_to(root)
            data = z.read(name)
            if len(data) != record["bytes"] or sha(data) != record["sha256"]:
                raise ValueError("Protocol member SHA-256 mismatch: " + name)
            if target.exists() and (not target.is_file() or sha(target.read_bytes()) != record["sha256"]):
                raise FileExistsError("Existing metadata differs; inspect rather than overwrite: " + name)
        for name in records:
            target = root / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(z.read(name))
    for directory in ["test_scratch", "checks", "queue_plans"]:
        (root / directory).mkdir(exist_ok=True)
    return {"valid": True, "metadata_files": len(records), "dataset_images": 0, "downloads": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    print(json.dumps(prepare(args.root), indent=2))
