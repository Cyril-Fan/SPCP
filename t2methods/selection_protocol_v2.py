"""Immutable source-selection v2 registration, independent from legacy v1."""
import copy
from .checkpoint import json_hash
from .round2_settings import DEFAULTS, settings

SCHEMA = "t2methods.source_selection.v2"
WINNER_SCHEMA = "t2methods.source_winner.v2"
VIEWS = ["clean", "jpeg75", "horizontal_rgb_gain_0.85_1.15",
         "bicubic_224_160_224", "gaussian_kernel9_sigma0.7"]
VAL_SHA = "fb7e5d2c12102aef04d68a2b968e3bca5084237681752a8fde6bb39abbe7b0b4"
GRID = {"PTR": ("rho", [.5, 1.]), "BMC": ("weight", [.10, .25]), "CPTA": ("eta", [.25, .50])}
PRIORITY = ["PTR", "CPTA", "BMC"]
RULE = dict(schema=SCHEMA, candidates_per_method=2, views=VIEWS, cosine_scale=10.,
            risk="class-balanced mean of 0.5 clean CE + 0.5 worst-corruption CE",
            direction="minimize", epsilon=1e-4, within_tie="risk then clean CE then candidate index",
            across_tie="risk then clean CE then PTR/CPTA/BMC; ignore candidate index",
            final_checkpoint="fixed final epoch", source_images=64, source_classes=16,
            images_per_class=4, cumulative_extra_labels=64, new_labels=0,
            source_ids_sha256=VAL_SHA, targets_used=False)
RULE_SHA = json_hash(RULE)


def candidate_settings(method, index):
    if type(index) is not int or index not in (0, 1):
        raise ValueError("v2 candidate index must be 0 or 1")
    key, values = GRID[method]
    return settings({**DEFAULTS[method], key: values[index]})


def choose(rows, across=False):
    import math
    if not rows or any(not all(math.isfinite(r[k]) for k in ("risk", "clean_ce")) for r in rows):
        raise ValueError("Selection requires finite risk and clean CE")
    low = min(row["risk"] for row in rows)
    tied = [row for row in rows if row["risk"] <= low + RULE["epsilon"]]
    clean = min(row["clean_ce"] for row in tied)
    finalists = [row for row in tied if row["clean_ce"] <= clean + RULE["epsilon"]]
    selected = min(finalists, key=(lambda r: PRIORITY.index(r["method"])) if across else (lambda r: r["candidate_order"]))
    return selected, {"risk_min": low, "risk_tie_count": len(tied), "clean_ce_min_in_risk_tie": clean,
                      "final_tie_count": len(finalists), "epsilon": RULE["epsilon"],
                      "across_methods": across, "risk_gap_max_min": max(r["risk"] for r in rows)-low}
