"""Check bundled split defaults and actionable errors without image downloads."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from spcp import dataset_loader
from spcp.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class DatasetLoaderChecks(unittest.TestCase):
    def test_bundled_defaults_match_explicit_raw_splits(self):
        for suffix in ("patternnet", "rsicd", "resisc45", "mlrsnet"):
            with self.subTest(dataset=suffix):
                config = load_config(ROOT / "configs" / ("SPCP_b2n_" + suffix + ".yaml"), 1)
                config["data"]["metadata_only"] = True
                explicit = dataset_loader.load_dataset(config, seed=1)
                default_config = copy.deepcopy(config)
                default_config["data"].pop("split")
                default = dataset_loader.load_dataset(default_config, seed=1)
                self.assertEqual(vars(default), vars(explicit))

    def test_missing_bundled_split_explains_preparation(self):
        missing_module = ROOT / "test_scratch" / "absent_checkout" / "spcp" / "dataset_loader.py"
        self.assertFalse(missing_module.parent.exists())
        with patch.object(dataset_loader, "__file__", str(missing_module)):
            with self.assertRaisesRegex(
                FileNotFoundError, "python scripts/prepare_protocol_assets.py"
            ):
                dataset_loader.load_dataset(
                    {"data": {"dataset": "RSICD", "metadata_only": True}}, seed=1
                )

    def test_explicit_split_path_is_not_silently_replaced(self):
        config = {
            "_config_dir": str(ROOT),
            "data": {
                "dataset": "RSICD",
                "metadata_only": True,
                "split": "assets/absent_split.json",
            },
        }
        with self.assertRaisesRegex(FileNotFoundError, "absent_split.json"):
            dataset_loader.load_dataset(config, seed=1)

    def test_default_split_still_checks_declared_hash(self):
        config = {"data": {"dataset": "RSICD", "metadata_only": True, "split_sha256": "0" * 64}}
        with self.assertRaisesRegex(ValueError, "Split SHA256 mismatch"):
            dataset_loader.load_dataset(config, seed=1)

    def test_dg_never_falls_back_to_raw_split(self):
        for dataset in ("PatternNet", "PatternNetv2"):
            with self.subTest(dataset=dataset):
                config = {"data": {"dataset": dataset, "task": "dg", "metadata_only": True}}
                with self.assertRaisesRegex(ValueError, "DG requires an explicit data.split"):
                    dataset_loader.load_dataset(config, seed=1)

    def test_custom_dataset_requires_explicit_split(self):
        with self.assertRaisesRegex(ValueError, "data.split must be a non-empty filesystem path"):
            dataset_loader.load_dataset(
                {"data": {"dataset": "CustomScene", "metadata_only": True}}, seed=1
            )


if __name__ == "__main__":
    unittest.main()
