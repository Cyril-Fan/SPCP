"""Local source edits are allowed; tokenizer and weight identities remain checked."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import bind_external_assets as binding
from spcp.checkpoint import file_hash, write_json
from spcp.cli import offline_clip
from tests.fake_clip import FakeClip, tokenize

ROOT = Path(__file__).resolve().parents[1]


class ExternalAssetChecks(unittest.TestCase):
    def test_source_comments_do_not_block_binding_or_loading(self):
        scratch = ROOT / "test_scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(scratch)) as directory:
            root = Path(directory)
            clip_root = root / "original_clip"
            package = clip_root / "clip"
            package.mkdir(parents=True)
            for name in binding.CLIP_SOURCE_SHA:
                (clip_root / name).write_text("# locally reformatted source\n\n", encoding="utf-8")
            bpe = package / "bpe_simple_vocab_16e6.txt.gz"
            bpe.write_bytes(b"synthetic tokenizer")
            weights = root / "weights.pt"
            weights.write_bytes(b"synthetic weights")
            weight_sha, bpe_sha = file_hash(weights), file_hash(bpe)
            with patch.object(binding, "CLIP_WEIGHT_SHA", weight_sha), patch.object(
                binding, "CLIP_BPE_SHA", bpe_sha
            ):
                value = binding.bind(clip_root, weights)
                legacy = copy.deepcopy(value)
                legacy["runtime"]["clip_source_hashes"] = {"clip/clip.py": "old checksum"}
                legacy["original_CLIP_source_guard"] = "historical metadata"
                path = root / "binding.json"
                write_json(path, legacy)
                binding.main(["--check-binding", str(path)])
                config = {"runtime": value["runtime"], "train": {"device": "cpu"}}
                fake = SimpleNamespace(
                    __file__=str(package / "__init__.py"),
                    tokenize=tokenize,
                    load=lambda *args, **kwargs: (FakeClip(), "preprocess"),
                )
                with patch.dict("sys.modules", {"clip": fake}):
                    model, module, transform = offline_clip(config)
                    self.assertIs(module, fake)
                    self.assertEqual(transform, "preprocess")
                    (package / "clip.py").write_text(
                        "# changed only comments and spacing\n", encoding="utf-8"
                    )
                    offline_clip(config)
                    weights.write_bytes(b"wrong weights")
                    with self.assertRaisesRegex(ValueError, "checkpoint SHA"):
                        offline_clip(config)
                    with self.assertRaisesRegex(ValueError, "weights differ"):
                        binding.bind(clip_root, weights)
                    weights.write_bytes(b"synthetic weights")
                    bpe.write_bytes(b"wrong tokenizer")
                    with self.assertRaisesRegex(ValueError, "BPE"):
                        offline_clip(config)
                    with self.assertRaisesRegex(ValueError, "BPE"):
                        binding.bind(clip_root, weights)


if __name__ == "__main__":
    unittest.main()
