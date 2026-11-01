"""Compatibility is determined by execution settings and state, not prose."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


from spcp.checkpoint import (
    completed_run,
    finish_run,
    load_checkpoint,
    restore_model,
    save_checkpoint,
    state_hash,
    write_json,
)
from spcp.config import configuration_identity, load_config, request_identity
from spcp.training import Trainer
from tests import test_integration as integration

ROOT = Path(__file__).resolve().parents[1]


class CheckpointCompatibilityChecks(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "test_scratch"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(scratch))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = integration.IntegrationChecks()
        self.bundle, self.config = self.fixture.fixture(self.root / "source", "SPCP")
        self.config["_request_identity"] = request_identity(self.config)

    def trainer(self, name, config=None):
        config = config or self.config
        output = self.root / name
        output.mkdir(exist_ok=True)
        return Trainer(self.fixture.model("SPCP", config), self.bundle, config, output)

    def legacy_metadata(self, saved):
        saved["metadata"]["request_identity"] = {
            "configuration_sha256": "old-identity-including-prose",
            "source_hashes": {"training.py": "previous-source-bytes"},
            "seed": 1,
        }
        saved["metadata"]["config"]["paper"] = {"title": "Historical title", "status": "draft"}
        saved["metadata"]["config"]["assumptions"] = {"note": "Historical explanation"}
        saved["metadata"]["config"]["runtime"].update(
            phase="formal",
            progress_interval=10,
            clip_source_hashes={"clip.py": "old"},
        )
        saved["metadata"]["clip_source_hashes"] = {"clip.py": "old source, same model"}
        return saved

    def test_old_checkpoint_replays_and_completed_run_is_reused(self):
        trainer = self.trainer("original")
        result = trainer.run()
        path = trainer.output / "final.pt"
        saved = self.legacy_metadata(load_checkpoint(path))
        save_checkpoint(path, saved)
        write_json(trainer.output / "metadata.json", saved["metadata"])
        finish_run(trainer.output, saved["metadata"]["request_identity"], saved["state_sha256"])
        changed = copy.deepcopy(self.config)
        changed.update(paper={"title": "Edited title"}, assumptions={"note": "Reworded"})
        changed["_source_hashes"] = {"training.py": "new formatting"}
        changed["_clip_source_hashes"] = {"clip.py": "edited comments"}
        changed["_request_identity"] = request_identity(changed)
        self.assertTrue(completed_run(trainer.output, changed["_request_identity"]))
        replay = self.trainer("replay", changed).evaluate_checkpoint(path)
        self.assertEqual(result["state_sha256"], replay["state_sha256"])
        self.assertEqual(result["hm"], replay["hm"])

    def test_old_checkpoint_resumes_identical_trajectory(self):
        import spcp.training as training

        full = self.trainer("full").run()
        partial = self.trainer("partial")
        original_save = training.save_checkpoint

        def interrupt(path, saved):
            original_save(path, self.legacy_metadata(saved))
            if saved["completed_epochs"] == 1:
                raise RuntimeError("test interruption")

        with patch.object(training, "save_checkpoint", interrupt):
            with self.assertRaisesRegex(RuntimeError, "test interruption"):
                partial.run()
        changed = copy.deepcopy(self.config)
        changed["paper"] = {"title": "Changed explanation only"}
        changed["_clip_source_hashes"] = {"clip.py": "new formatting"}
        resumed = self.trainer("partial", changed).run(resume=partial.output / "last.pt")
        self.assertEqual(full["state_sha256"], resumed["state_sha256"])
        self.assertEqual(full["hm"], resumed["hm"])

    def test_execution_config_seed_and_assets_still_rejected(self):
        trainer = self.trainer("full")
        trainer.run()
        saved = self.legacy_metadata(load_checkpoint(trainer.output / "final.pt"))
        for section, key, value in (
            ("train", "learning_rate", 0.1),
            ("train", "workers", 1),
            ("model", "prompt_length", 8),
            ("method", "eta", 0.25),
            ("runtime", "clip_bpe_sha256", "other tokenizer"),
            ("data", "split_sha256", "other split"),
        ):
            with self.subTest(section=section, key=key):
                bad = copy.deepcopy(saved)
                bad["metadata"]["config"][section][key] = value
                with self.assertRaisesRegex(ValueError, "execution configuration"):
                    trainer._validate_saved(bad, final=True)
        for key in (
            "seed",
            "source_identity",
            "taxonomy_sha256",
            "clip_checkpoint_sha256",
            "frozen_backbone_before",
            "trainable_parameters",
        ):
            with self.subTest(provenance=key):
                bad = copy.deepcopy(saved)
                bad["metadata"][key] = "mismatch"
                with self.assertRaisesRegex(ValueError, "provenance"):
                    trainer._validate_saved(bad, final=True)

    def test_prompt_names_shapes_and_graph_are_still_checked(self):
        trainer = self.trainer("full")
        trainer.run()
        saved = load_checkpoint(trainer.output / "final.pt")
        name = next(iter(saved["prompts"]))
        for kind in ("name", "shape", "nonfinite", "graph"):
            with self.subTest(kind=kind):
                bad = copy.deepcopy(saved)
                if kind == "name":
                    bad["prompts"].pop(name)
                elif kind == "shape":
                    bad["prompts"][name] = bad["prompts"][name][:1]
                elif kind == "nonfinite":
                    bad["prompts"][name].fill_(float("nan"))
                else:
                    bad["method_state"]["buffers"]["source_delta"] += 0.1
                # Test structural checks independently of the artifact checksum.
                bad["state_sha256"] = state_hash(
                    {"prompts": bad["prompts"], "method": bad["method_state"]}
                )
                with self.assertRaises(ValueError):
                    restore_model(self.fixture.model("SPCP", self.config), bad)

    def test_prose_and_old_runtime_fields_do_not_affect_configuration(self):
        for path in sorted((ROOT / "configs").glob("*.yaml")):
            original = load_config(path)
            edited = copy.deepcopy(original)
            edited["paper"] = {"title": "New title", "status": "new status"}
            edited["assumptions"] = {"explanation": "Edited documentation"}
            edited["_source_hashes"] = {"config.py": "changed comments"}
            edited["runtime"].update(
                phase="formal",
                progress_interval=10,
                source_validation_consumed=False,
                clip_source_hashes={"old": "hash"},
            )
            self.assertEqual(configuration_identity(original), configuration_identity(edited))
            self.assertEqual(request_identity(original), request_identity(edited))


if __name__ == "__main__":
    unittest.main()
