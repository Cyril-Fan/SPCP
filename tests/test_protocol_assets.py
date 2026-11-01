"""Validate portable protocol metadata without dataset downloads or a GPU."""

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile
import unittest
import zipfile

from scripts.prepare_protocol_assets import prepare

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


class ProtocolAssetChecks(unittest.TestCase):
    def test_preparation_only_creates_protocol_assets_and_is_repeatable(self):
        scratch = ROOT / "test_scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(scratch)) as directory:
            root = Path(directory)
            (root / "assets").mkdir()
            for name in ("protocol_assets.zip", "protocol_bundle_manifest.json"):
                shutil.copyfile(ROOT / "assets" / name, root / "assets" / name)
            first = prepare(root)
            before = {
                p.relative_to(root).as_posix(): digest(p.read_bytes())
                for p in root.rglob("*")
                if p.is_file()
            }
            self.assertEqual(prepare(root), first)
            after = {
                p.relative_to(root).as_posix(): digest(p.read_bytes())
                for p in root.rglob("*")
                if p.is_file()
            }
            self.assertEqual(before, after)
            self.assertEqual({p.name for p in root.iterdir()}, {"assets"})

    def test_archive_inventory_and_hashes(self):
        manifest = json.loads(
            (ROOT / "assets/protocol_bundle_manifest.json").read_text(encoding="utf-8")
        )
        archive_path = ROOT / "assets/protocol_assets.zip"
        self.assertEqual(digest(archive_path.read_bytes()), manifest["archive_sha256"])
        with zipfile.ZipFile(archive_path) as archive:
            self.assertIsNone(archive.testzip())
            records = {item["path"]: item for item in manifest["files"]}
            self.assertEqual(len(records), len(manifest["files"]))
            self.assertEqual(set(archive.namelist()), set(records))
            for name, record in records.items():
                with self.subTest(member=name):
                    data = archive.read(name)
                    self.assertEqual(len(data), record["bytes"])
                    self.assertEqual(digest(data), record["sha256"])

    def test_dg_ledgers_are_portable_and_cover_published_splits(self):
        receipt = json.loads(
            (ROOT / "assets/dg_v2/pixel_provenance/data_dg_v2_ready.json").read_text(
                encoding="utf-8"
            )
        )
        manifest = json.loads(
            (ROOT / "assets/protocol_bundle_manifest.json").read_text(encoding="utf-8")
        )
        with zipfile.ZipFile(ROOT / "assets/protocol_assets.zip") as archive:
            splits = {
                item["sha256"]: archive.read(item["path"])
                for item in manifest["files"]
                if "/dg_v2/splits/" in item["path"]
            }
            for dataset in receipt["results"]:
                with self.subTest(dataset=dataset["dataset"]):
                    name = "assets/dg_v2/pixel_provenance/" + dataset["ledger_path"]
                    data = archive.read(name)
                    self.assertEqual(digest(data), dataset["ledger_sha256"])
                    expected = {
                        identifier: (partition, label, classname)
                        for partition, rows in json.loads(splits[dataset["split_sha256"]]).items()
                        for identifier, label, classname in rows
                    }
                    seen = set()
                    for line in data.splitlines():
                        row = json.loads(line)
                        identifier = row["id"]
                        self.assertNotIn(identifier, seen)
                        seen.add(identifier)
                        self.assertNotIn("path", row)
                        self.assertFalse(PurePosixPath(identifier).is_absolute())
                        self.assertFalse(PureWindowsPath(identifier).is_absolute())
                        self.assertNotIn("..", PurePosixPath(identifier).parts)
                        self.assertEqual(
                            (row["partition"], row["label"], row["classname"]), expected[identifier]
                        )
                        self.assertEqual(len(row["sha256"]), 64)
                        int(row["sha256"], 16)
                        self.assertGreater(row["bytes"], 0)
                        self.assertEqual(row["decoded_mode"], "RGB")
                    self.assertEqual(seen, set(expected))
                    self.assertEqual(len(seen), dataset["images_verified"])


if __name__ == "__main__":
    unittest.main()
