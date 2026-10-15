"""Fixed six-configuration, three-seed exploratory target-domain DG protocol."""
from .config import require
from .round2_settings import METHODS
from .selection_protocol_v2 import candidate_settings

SCHEMA = "t2methods.dg_screen3.v1"
TARGETS = ("RSICDv2", "NWPU-RESISC45v2", "MLRSNetv2")
GRID = tuple((method, candidate, seed) for method in METHODS for candidate in (0, 1) for seed in (1, 2, 3))


def validate_dg3_config(config):
    runtime = config["runtime"]
    method = config["method"]["name"]
    candidate = runtime.get("candidate")
    require(runtime.get("experiment_protocol") == SCHEMA, "DG3 requires its own explicit protocol")
    require(config["data"]["task"] == "dg" and config["data"]["dataset"] == "PatternNetv2", "DG3 source must be PatternNetv2")
    require((method, candidate, config["_seed"]) in GRID and type(candidate) is int, "DG3 grid/seed is unregistered")
    require(config["method"] == candidate_settings(method, candidate), "DG3 candidate changed after preregistration")
    require(runtime.get("require_source_selection") is False and runtime.get("require_winner") is False,
            "DG3 does not consume source-selection or winner receipts")
    require(not runtime.get("control_overrides") and runtime.get("source_validation_consumed") is False,
            "DG3 has no ablations or source-validation selection")
    require(runtime.get("save_scores") is True, "DG3 saves complete per-image scores")
    require(tuple(t["dataset"] for t in config["targets"]) == TARGETS, "DG3 requires all three targets")

