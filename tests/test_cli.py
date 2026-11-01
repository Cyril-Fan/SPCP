"""Exercise CLI training, replay and epoch recovery with CPU synthetic assets."""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spcp import cli
from spcp.config import request_identity
from spcp.training import seed_everything
from tests import test_integration as integration
from tests.fake_clip import FakeClip, CLIP_MODULE

ROOT = Path(__file__).resolve().parents[1]


class CLIChecks(unittest.TestCase):
    def test_train_evaluate_skip_and_resume_entry_points(self):
        import spcp.training as training

        scratch = ROOT / "test_scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(scratch)) as directory:
            root = Path(directory)
            bundle, config = integration.IntegrationChecks().fixture(root / "source", "SPCP")
            config["_request_identity"] = request_identity(config)

            def local_clip(config):
                seed_everything(123)
                return FakeClip(input_size=224), CLIP_MODULE, config["_eval_preprocess"]

            def run(command, output, extra=()):
                return cli.main(
                    [command, "--config", "synthetic.yaml", "--seed", "1", "--output", str(output)]
                    + list(extra)
                )

            with patch.object(
                cli, "load_config", side_effect=lambda *args: copy.deepcopy(config)
            ), patch.object(cli, "offline_clip", side_effect=local_clip), patch(
                "spcp.dataset_loader.load_dataset", return_value=bundle
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                full = root / "full"
                first = run("train", full)
                reused = run("train", full, ["--skip-completed"])
                self.assertEqual(reused["status"], "reused_verified_complete")
                replay = run("evaluate", root / "replay", ["--checkpoint", str(full / "final.pt")])
                self.assertEqual(first["state_sha256"], replay["state_sha256"])
                partial = root / "partial"
                original_save = training.save_checkpoint

                def interrupt(path, saved):
                    original_save(path, saved)
                    if saved["completed_epochs"] == 1:
                        raise RuntimeError("test interruption")

                with patch.object(training, "save_checkpoint", interrupt):
                    with self.assertRaisesRegex(RuntimeError, "test interruption"):
                        run("train", partial)
                self.assertTrue((partial / "failure.json").is_file())
                resumed = run("train", partial, ["--resume", str(partial / "last.pt")])
                self.assertEqual(first["state_sha256"], resumed["state_sha256"])
                self.assertEqual(first["hm"], resumed["hm"])


if __name__ == "__main__":
    unittest.main()
