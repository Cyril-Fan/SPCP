"""Offline real execution; help/config validation never load torch or CLIP."""
import argparse
import json
from pathlib import Path
import sys
import traceback
import uuid

from .checkpoint import (absolute_path, archive_failed_attempt, completed_run, failed_retry_kind,
                         file_hash, interrupted_retry_kind, load_checkpoint, record_interruption,
                         validate_epoch_resume, write_json)
from .config import CLIP_SOURCE_SHA, apply_asset_binding, apply_source_selection, load_config, target_config, validate_config, verify_dg_pixels


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate", "validate-config"):
        child = commands.add_parser(command)
        child.add_argument("--config", required=True)
        child.add_argument("--seed", type=int, default=1)
        child.add_argument("--device", choices=("cpu", "cuda"))
        child.add_argument("--workers", type=int)
        child.add_argument("--selection-receipt", help="Explicit source-only screening receipt freezing the formal method configuration")
        child.add_argument("--asset-binding", help="One offline binding for external CLIP/BPE/weights and optional image roots")
        if command != "validate-config":
            child.add_argument("--output")
            child.add_argument("--clip-root")
            child.add_argument("--clip-checkpoint", help="Existing local ViT-B/16 checkpoint; names/URLs/downloads are prohibited")
            child.add_argument("--retry-failed", action="store_true", help="Archive recorded failed attempts and recover a checkpoint; restart only if no checkpoint exists")
            child.add_argument("--recover-interrupted", action="store_true", help="Recover an inactive unrecorded interruption with a committed final/epoch checkpoint; never start a new fit")
        if command == "train":
            child.add_argument("--resume", help="Explicit completed-epoch last.pt from this same output")
            child.add_argument("--skip-completed", action="store_true")
        if command == "evaluate":
            child.add_argument("--checkpoint", required=True)
            child.add_argument("--inference-off", action="store_true", help="Replay method-off inference without a new fit")
            child.add_argument("--finalize-reused-fit", action="store_true", help="Mark a matching selected source fit as formally evaluated without retraining")
    return result


def offline_clip(config):
    """Load only an explicit file: original clip.load cannot download by name."""
    root = Path(config["runtime"].get("clip_root", "")).resolve(strict=True)
    package = root / "clip"
    for name in ("__init__.py", "clip.py", "model.py", "simple_tokenizer.py", "bpe_simple_vocab_16e6.txt.gz"):
        if not (package / name).is_file():
            raise FileNotFoundError("Complete original offline CLIP source is required: " + str(package / name))
    actual_source = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(package.rglob("*.py"))}
    if actual_source != CLIP_SOURCE_SHA:
        raise ValueError("External CLIP source SHA inventory differs from the sealed-v3 original code")
    bpe_expected = config["runtime"].get("clip_bpe_sha256")
    if not isinstance(bpe_expected, str) or len(bpe_expected) != 64 or file_hash(package / "bpe_simple_vocab_16e6.txt.gz") != bpe_expected:
        raise ValueError("Declare and verify the external official BPE asset SHA before real execution")
    weights = Path(config["runtime"].get("clip_checkpoint", "")).resolve(strict=True)
    if not weights.is_file():
        raise FileNotFoundError("Explicit local ViT-B/16 weight file is required")
    digest = file_hash(weights)
    expected = config["runtime"].get("clip_checkpoint_sha256")
    if expected and expected != digest:
        raise ValueError("CLIP checkpoint SHA differs")
    import importlib
    import torch
    if config["train"]["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Declared CUDA device is unavailable; do not silently change execution device")
    existing = sys.modules.get("clip")
    if existing is not None:
        Path(existing.__file__).resolve().relative_to(package.resolve())
    sys.path.insert(0, str(root))
    try:
        clip = importlib.import_module("clip")
    finally:
        sys.path.remove(str(root))
    Path(clip.__file__).resolve().relative_to(package.resolve())
    # Passing the checked existing path is crucial: passing ViT-B/16 would run
    # the loader's network/cache branch. No runtime downloads are supported.
    clip_model, preprocess = clip.load(str(weights), device=config["train"]["device"], jit=False)
    clip_model.float().eval()
    config["_clip_checkpoint_sha256"] = digest
    config["_clip_source_hashes"] = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(package.rglob("*"))
                                      if p.is_file() and (p.suffix == ".py" or p.name.endswith(".txt.gz"))}
    return clip_model, clip, preprocess


def execute(args):
    overrides = {}
    for attribute, section, key in (("device", "train", "device"), ("workers", "train", "workers"),
                                    ("clip_root", "runtime", "clip_root"), ("clip_checkpoint", "runtime", "clip_checkpoint")):
        value = getattr(args, attribute, None)
        if value is not None:
            overrides.setdefault(section, {})[key] = value
    config = load_config(args.config, args.seed, overrides)
    if args.asset_binding:
        apply_asset_binding(config, args.asset_binding)
    if args.selection_receipt:
        apply_source_selection(config, args.selection_receipt)
    if args.command == "validate-config":
        value = validate_config(config)
        value["request_identity"] = config["_request_identity"]
        print(json.dumps(value, ensure_ascii=False), flush=True)
        return value
    if args.command == "evaluate" and args.inference_off and config["method"]["name"] != "SGRP":
        raise ValueError("--inference-off is implemented only for SGRP; SCA/LSP must use their explicitly declared controls")
    default_name = "%s_%s_%s_seed%d" % (config["method"]["name"], config["data"]["task"], config["data"]["dataset"], args.seed)
    output = absolute_path(args.output or Path(config["runtime"]["output_root"]) / default_name)
    if args.command == "train" and getattr(args, "skip_completed", False) and completed_run(output, config["_request_identity"]):
        print(json.dumps({"status": "reused_verified_complete", "output": str(output)}), flush=True)
        return {"status": "reused_verified_complete", "output": str(output)}
    resume = getattr(args, "resume", None)
    retry_kind = None
    interrupted_recovery = getattr(args, "recover_interrupted", False)
    if sum((resume is not None, bool(getattr(args, "retry_failed", False)), bool(interrupted_recovery))) > 1:
        raise ValueError("Use only one of --resume, --retry-failed, or --recover-interrupted")
    if interrupted_recovery:
        retry_kind = interrupted_retry_kind(output)
        if retry_kind == "resume_epoch":
            resume = output / "last.pt"
    if getattr(args, "retry_failed", False) and output.exists():
        if resume is not None:
            raise ValueError("Use either --resume or --retry-failed, not both")
        retry_kind = failed_retry_kind(output)
        if retry_kind == "resume_epoch":
            resume = output / "last.pt"
    if args.command == "evaluate" and retry_kind == "resume_epoch":
        raise ValueError("An epoch checkpoint must be recovered by train, never ignored by an evaluation command")
    if resume is not None:
        resume = absolute_path(resume)
        if not resume.is_file():
            raise FileNotFoundError("Resume checkpoint is unavailable")
        if resume.parent != output or resume.name != "last.pt":
            raise ValueError("Resume must explicitly use this output's last.pt epoch checkpoint")
        if not output.is_dir():
            raise ValueError("Resume output does not exist")
        validate_epoch_resume(output, resume)
    elif output.exists() and retry_kind is None:
        raise FileExistsError("Output exists; verified reuse or explicit epoch resume is required: " + str(output))
    elif not output.exists():
        output.mkdir(parents=True)
    archived_retry = False
    if retry_kind == "restart_pre_checkpoint":
        archive_failed_attempt(output, retry_kind)
        archived_retry = True
    try:
        if config["runtime"].get("require_source_selection", False) and not args.selection_receipt:
            raise ValueError("Formal execution requires an explicit source-only selection receipt; dry-run defaults are placeholders")
        # Imports below are intentionally below argument/config/output checks.
        from .dataset_loader import load_dataset
        from .models import build_model
        from .training import Trainer, seed_everything
        seed_everything(args.seed, config["train"]["device"])
        bundle = load_dataset(config, seed=args.seed)
        targets = ([] if config["runtime"].get("phase") == "screening" else
                   [(target, load_dataset(target_config(config, target), task="eval", seed=args.seed)) for target in config.get("targets", [])])
        if config["data"]["task"] == "dg":
            verify_dg_pixels(config, [bundle] + [item[1] for item in targets])
        clip_model, clip_module, preprocess = offline_clip(config)
        config["_eval_preprocess"] = preprocess
        model = build_model(config["method"]["name"], clip_model, clip_module, config).to(config["train"]["device"])
        trainer = Trainer(model, bundle, config, output, targets=targets)
        if retry_kind is not None and not archived_retry:
            if retry_kind != "restart_pre_checkpoint":
                committed = output / ("final.pt" if retry_kind == "evaluate_final" else "last.pt")
                saved = load_checkpoint(committed)
                trainer._validate_recovery_checkpoint(committed, saved, final=retry_kind == "evaluate_final")
                if interrupted_recovery:
                    record_interruption(output, retry_kind, saved)
            archive_failed_attempt(output, retry_kind)
            archived_retry = True
        if retry_kind == "evaluate_final":
            result = trainer.evaluate_checkpoint(output / "final.pt", recover_final=True)
        elif args.command == "evaluate":
            if args.finalize_reused_fit:
                selected = config.get("_selection_receipt", {}).get("selected_output")
                if config["data"]["task"] != "dg" or args.seed != 1 or selected is None or absolute_path(args.checkpoint) != absolute_path(Path(selected) / "final.pt"):
                    raise ValueError("Formal fit reuse must use exactly the selected source-screening DG seed1 checkpoint")
            model.inference_off = args.inference_off
            result = trainer.evaluate_checkpoint(absolute_path(args.checkpoint), finalize_reused=args.finalize_reused_fit)
        else:
            result = trainer.run(resume=resume)
        print(json.dumps({"output": str(output), "result": result}, ensure_ascii=False), flush=True)
        return result
    except Exception as error:
        if retry_kind is not None and not archived_retry:
            write_json(output / ("retry_rejected_" + uuid.uuid4().hex + ".json"), {"type": type(error).__name__, "message": str(error),
                                                                                 "traceback": traceback.format_exc(), "original_artifacts_preserved": True})
            raise
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                                             "traceback": traceback.format_exc(), "measured_success": False})
        write_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__, "message": str(error)})
        raise


def main(argv=None):
    return execute(parser().parse_args(argv))


if __name__ == "__main__":
    main()
