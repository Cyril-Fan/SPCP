"""Summarize only content-verified completed results; no runtime imports."""
import argparse
import json
from pathlib import Path
import statistics

from .checkpoint import absolute_path, completed_run, file_hash, write_json


def summarize(root):
    groups, rejected = {}, []
    for marker_path in sorted(Path(root).rglob("complete.json")):
        output = marker_path.parent
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not completed_run(output, marker.get("request_identity")):
            rejected.append(str(output))
            continue
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        result = json.loads((output / "results.json").read_text(encoding="utf-8"))
        if result.get("phase") == "screening" or metadata["config"]["runtime"].get("control_overrides"):
            continue
        key = (metadata["model_name"], metadata["task"], metadata["dataset"])
        groups.setdefault(key, []).append({"seed": metadata["seed"], "score": result["hm"] if metadata["task"] == "b2n" else result["target_mean"],
                                          "training_seconds": result.get("training_seconds"), "output": str(output), "result": result})
    rows = []
    for (method, task, dataset), values in sorted(groups.items()):
        seeds = [value["seed"] for value in values]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Multiple completed fits for the same method/task/dataset/seed; select one prospectively")
        rows.append({"method": method, "task": task, "dataset": dataset, "seeds": sorted(seeds),
                     "three_seed_complete": set(seeds) == {1, 2, 3}, "score_mean": statistics.mean(v["score"] for v in values),
                     "score_sample_std": statistics.stdev(v["score"] for v in values) if len(values) > 1 else None,
                     "metric": "per-seed base/new HM" if task == "b2n" else "per-seed equal target-domain mean",
                     "runs": values})
    return {"source": "measured_verified_artifacts", "groups": rows, "rejected": rejected}


def select_source_configuration(root, method):
    """After screening, create a reviewable receipt using source metrics only."""
    candidates = []
    for marker_path in sorted(Path(root).rglob("complete.json")):
        output = marker_path.parent
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not completed_run(output, marker.get("request_identity")):
            continue
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        result = json.loads((output / "results.json").read_text(encoding="utf-8"))
        if metadata["model_name"] != method or result.get("phase") != "screening":
            continue
        if result.get("targets_consumed") is not False or result.get("target_evaluations") != 0 or result.get("source_validation_count") != 64:
            raise ValueError("A screening candidate consumed targets or undeclared source validation")
        candidates.append({"output": str(absolute_path(output)), "request_identity": marker["request_identity"],
                           "candidate_order": metadata["config"]["runtime"]["candidate"],
                           "selection_score": result["selection_score"], "source_validation_ids_sha256": result["source_validation_ids_sha256"],
                           "results_sha256": file_hash(output / "results.json"), "checkpoint_sha256": file_hash(output / "final.pt"),
                           "method_config": metadata["config"]["method"]})
    if len(candidates) != 3 or {row["candidate_order"] for row in candidates} != {0, 1, 2} or len({row["source_validation_ids_sha256"] for row in candidates}) != 1:
        raise ValueError("All three predeclared candidates must complete using the same source validation cohort")
    selected = sorted(candidates, key=lambda row: (-row["selection_score"], row["candidate_order"]))[0]
    return {"schema": "t2methods.source_selection.v1", "method": method,
            "selection_rule": "highest equal-weight source validation score; tie lowest predeclared candidate order",
            "selected_output": selected["output"], "method_config": selected["method_config"], "candidates": candidates,
            "targets_used": False, "extra_source_validation_labels": 64}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--select-screening", choices=("SGRP", "SCA", "LSP"))
    args = parser.parse_args(argv)
    value = select_source_configuration(args.root, args.select_screening) if args.select_screening else summarize(args.root)
    if args.output:
        write_json(args.output, value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


if __name__ == "__main__":
    main()
