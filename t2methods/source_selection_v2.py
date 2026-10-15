"""Strict two-candidate selection and one-winner freeze. Never reads target metrics."""
import argparse
import copy
from pathlib import Path
from .checkpoint import absolute_path, completed_run, file_hash, json_hash, source_hashes, write_json, load_checkpoint
from .config import COMMON, read_json, require, request_identity, validate_config
from .selection_protocol_v2 import SCHEMA, WINNER_SCHEMA, RULE, RULE_SHA, VAL_SHA, GRID, choose, candidate_settings


def inspect_candidate(output, method):
    from .source_evaluation_v2 import recompute_scores
    output = absolute_path(output)
    marker = read_json(output / "complete.json")
    require(completed_run(output, marker.get("request_identity")), "v2 candidate is incomplete or hash-invalid")
    metadata, result = read_json(output / "metadata.json"), read_json(output / "results.json")
    require(metadata["model_name"] == method and metadata["seed"] == 1 and metadata["task"] == "dg", "Wrong source candidate identity")
    require(metadata["request_identity"] == marker["request_identity"] and marker["request_identity"]["source_hashes"] == source_hashes(), "Source code/request identity changed")
    config = copy.deepcopy(metadata["config"])
    config.update(_seed=1, _source_hashes=source_hashes())
    validate_config(config)
    require(request_identity(config) == marker["request_identity"], "Candidate configuration identity differs")
    candidate = config["runtime"]["candidate"]
    require(config["runtime"]["phase"] == "screening" and config["method"] == candidate_settings(method, candidate), "v2 candidate is not preregistered")
    require(result.get("selection_schema") == SCHEMA and result.get("selection_rule_sha256") == RULE_SHA
            and result.get("selection_direction") == "minimize", "v1/v2 or source rule mismatch")
    require(result.get("phase") == "screening" and result.get("targets_consumed") is False and result.get("target_evaluations") == 0,
            "Source selection candidate used targets")
    require(result.get("source_validation_ids_sha256") == VAL_SHA and result.get("source_validation_count") == 64
            and result.get("extra_validation_labels") == 64 and result.get("new_validation_labels") == 0,
            "Validation cohort/label budget changed")
    require(result.get("completed_epochs") == 10 and metadata.get("checkpoint_selection") == "final_epoch", "Only fixed final epoch is eligible")
    saved = load_checkpoint(output / "final.pt")
    require(saved.get("run_kind") == "training" and saved.get("completed_epochs") == 10 and saved.get("steps") == 160,
            "Checkpoint is not a complete fixed-epoch source fit")
    require(saved["metadata"]["request_identity"] == marker["request_identity"] and
            saved["state_sha256"] == result["state_sha256"] and saved["state_sha256"] == marker["state_sha256"],
            "Candidate checkpoint/config/state linkage differs")
    require(saved["method_state"]["method"] == method and saved["method_state"]["config"] == config["method"],
            "Candidate method state differs")
    require(file_hash(output / "source_validation_pixels.json") == result["source_validation_pixels_sha256"], "Source pixel audit differs")
    taxonomy = read_json(config["data"]["taxonomy"])
    metrics = recompute_scores(output, result["source_validation"], [taxonomy["classes"][c]["leaf"] for c in COMMON])
    require(abs(metrics["risk"] - result["selection_score"]) < 1e-10 and abs(metrics["clean_ce"] - result["selection_clean_ce"]) < 1e-10,
            "Continuous scores do not reproduce the source risk")
    expected_rows = read_json(config["source_validation"]["ids_file"])["records"]
    require([(r["id"], r["label"]) for r in metrics["per_image"]] == [(r["id"], r["label"]) for r in expected_rows],
            "Saved scores differ from preregistered source IDs/labels")
    pixels = read_json(output / "source_validation_pixels.json")
    require([(r["id"], r["label"]) for r in pixels] == [(r["id"], r["label"]) for r in expected_rows], "Source pixel ID order differs")
    ready = read_json(config["data"]["v2_data_ready"])
    proof = next(r for r in ready["results"] if r["dataset"] == "PatternNetv2")
    ledger = Path(config["data"]["ledger_root"]) / Path(proof["ledger_path"]).name
    require(file_hash(ledger) == proof["ledger_sha256"], "Source pixel ledger identity differs")
    import json
    known = {r["id"]: r for r in map(json.loads, ledger.read_text(encoding="utf-8").splitlines())}
    require(all(r["sha256"] == known[r["id"]]["sha256"] for r in pixels), "Saved source pixels differ from pinned pixel ledger")
    require(not set(r["id"] for r in expected_rows).intersection(metadata["source_identity"]["support_ids"]), "Validation/training overlap")
    return dict(method=method, candidate_order=candidate, output=str(output), request_identity=marker["request_identity"],
                risk=metrics["risk"], clean_ce=metrics["clean_ce"], method_config=config["method"],
                results_sha256=file_hash(output / "results.json"), checkpoint_sha256=file_hash(output / "final.pt"),
                configuration_sha256=marker["request_identity"]["configuration_sha256"],
                source_validation_ids_sha256=VAL_SHA, source_validation_pixels_sha256=result["source_validation_pixels_sha256"],
                source_identity_sha256=json_hash(metadata["source_identity"]),
                environment_sha256=json_hash(metadata.get("environment", {})))


def build_receipt(rows, method):
    require(len(rows) == 2 and {r["candidate_order"] for r in rows} == {0, 1}, "v2 requires exactly two distinct registered candidates")
    for key in ("source_validation_pixels_sha256", "source_identity_sha256", "environment_sha256"):
        require(len({r[key] for r in rows}) == 1, "Candidate cohort/environment differs: " + key)
    selected, ties = choose(rows)
    return dict(schema=SCHEMA, rule=RULE, rule_sha256=RULE_SHA, method=method,
                candidates=sorted(rows, key=lambda r: r["candidate_order"]), selected_output=selected["output"],
                selected_candidate=selected["candidate_order"], method_config=selected["method_config"],
                risk=selected["risk"], clean_ce=selected["clean_ce"], tie_decision=ties,
                targets_used=False, extra_source_validation_labels=64, new_source_validation_labels=0)


def select_method(root, method):
    require(method in GRID, "Unknown v2 method")
    rows = []
    for path in sorted(Path(root).rglob("complete.json")):
        metadata = read_json(path.parent / "metadata.json")
        if metadata.get("model_name") == method and metadata.get("config", {}).get("runtime", {}).get("phase") == "screening":
            rows.append(inspect_candidate(path.parent, method))
    return build_receipt(rows, method)


def verify_receipt(path):
    receipt = read_json(path)
    require(receipt.get("schema") == SCHEMA and receipt.get("rule_sha256") == RULE_SHA, "Wrong v2 receipt schema/rule")
    rows = [inspect_candidate(row["output"], receipt["method"]) for row in receipt["candidates"]]
    expected = build_receipt(rows, receipt["method"])
    require(receipt == expected, "v2 selection receipt does not reproduce the registered decision")
    return receipt


def build_winner(receipts):
    require(len(receipts) == 3 and {r["method"] for r in receipts} == set(GRID), "Winner needs all three method selections")
    rows = [{"method": r["method"], "risk": r["risk"], "clean_ce": r["clean_ce"]} for r in receipts]
    common_rows = [r["candidates"][0] for r in receipts]
    for key in ("source_validation_pixels_sha256", "source_identity_sha256", "environment_sha256"):
        require(len({r[key] for r in common_rows}) == 1, "Cross-method source/environment mismatch")
    selected, ties = choose(rows, across=True)
    return dict(schema=WINNER_SCHEMA, rule_sha256=RULE_SHA, winner=selected["method"], tie_decision=ties,
                method_receipts={r["method"]: r for r in receipts}, targets_used=False, admitted_methods=1,
                budget=dict(screening_new_fits=6, formal_new_fits=17, main_new_fits=23, optional_automatically_run=0))


def verify_winner(path):
    saved = read_json(path)
    require(saved.get("schema") == WINNER_SCHEMA, "v2 winner receipt required")
    receipts = []
    for method, receipt in saved["method_receipts"].items():
        current = build_receipt([inspect_candidate(r["output"], method) for r in receipt["candidates"]], method)
        require(receipt == current, "Winner's method receipt changed")
        receipts.append(current)
    require(saved == build_winner(receipts), "Winner selection does not follow preregistered rule")
    return saved


def apply_selection(config, path):
    saved = read_json(path)
    if saved.get("schema") == WINNER_SCHEMA:
        winner = verify_winner(path)
        require(winner["winner"] == config["method"]["name"], "Formal execution is limited to the frozen winner")
        receipt = winner["method_receipts"][winner["winner"]]
        config["_winner_receipt"] = {"path": str(absolute_path(path)), "sha256": file_hash(path)}
    else:
        require(not config["runtime"].get("require_winner", False), "Formal round2 requires the cross-method winner receipt")
        receipt = verify_receipt(path)
    require(receipt["method"] == config["method"]["name"], "Receipt method differs")
    config["method"] = copy.deepcopy(receipt["method_config"])
    config["method"].update(config["runtime"].get("control_overrides", {}))
    config["_selection_receipt"] = dict(path=str(absolute_path(path)), sha256=file_hash(path), selected_output=receipt["selected_output"])
    validate_config(config)
    config["_request_identity"] = request_identity(config)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs_round2")
    parser.add_argument("--method", choices=tuple(GRID))
    parser.add_argument("--winner", nargs=3, metavar="METHOD_RECEIPT")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    require(bool(args.method) != bool(args.winner), "Choose one method selection or a three-receipt winner freeze")
    result = build_winner([verify_receipt(path) for path in args.winner]) if args.winner else select_method(args.root, args.method)
    destination = Path(args.output)
    if destination.exists():
        require(read_json(destination) == result, "Refuse to overwrite a different frozen selection")
    else:
        write_json(destination, result)
    print("source-only selection frozen: " + str(destination))
    return result


if __name__ == "__main__": main()
