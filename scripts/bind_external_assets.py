"""Verify local CLIP assets and bind dataset paths to SPCP configurations."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from spcp.checkpoint import absolute_path, file_hash, write_json
from spcp.config import CLIP_SOURCE_SHA, CLIP_WEIGHT_SHA, CLIP_BPE_SHA, apply_asset_binding, load_config


def bind(clip_root, checkpoint, image_roots=None, expected_bpe=None):
    root, weights = absolute_path(clip_root), absolute_path(checkpoint)
    package = root / "clip"
    required = list(CLIP_SOURCE_SHA) + ["clip/bpe_simple_vocab_16e6.txt.gz"]
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete external original CLIP assets: " + ", ".join(missing))
    actual = {path.relative_to(root).as_posix(): file_hash(path) for path in sorted(package.rglob("*.py"))}
    if actual != CLIP_SOURCE_SHA:
        raise ValueError("External CLIP source differs from the pinned file inventory")
    if not weights.is_file() or file_hash(weights) != CLIP_WEIGHT_SHA:
        raise ValueError("Explicit external ViT-B/16 checkpoint does not match the official guarded SHA")
    bpe = file_hash(package / "bpe_simple_vocab_16e6.txt.gz")
    if bpe != CLIP_BPE_SHA:
        raise ValueError("CLIP BPE tokenizer SHA-256 differs")
    if expected_bpe is not None and expected_bpe != bpe:
        raise ValueError("External BPE SHA differs from the explicitly supplied expected value")
    roots = {}
    for dataset, path in (image_roots or {}).items():
        if dataset not in {"PatternNet", "RSICD", "NWPU-RESISC45", "MLRSNet", "PatternNetv2", "RSICDv2", "NWPU-RESISC45v2", "MLRSNetv2"}:
            raise ValueError("Unknown image domain: " + dataset)
        path = absolute_path(path)
        if not path.is_dir():
            raise FileNotFoundError("Explicit image root is unavailable: " + str(path))
        roots[dataset] = str(path)
    return {"schema": "spcp.external_assets.v1",
            "runtime": {"clip_root": str(root), "clip_checkpoint": str(weights),
                        "clip_checkpoint_sha256": CLIP_WEIGHT_SHA, "clip_source_hashes": CLIP_SOURCE_SHA, "clip_bpe_sha256": bpe},
            "image_roots": roots, "original_CLIP_source_guard": "pinned four-file SHA-256 inventory",
            "weight_guard": "ViT-B/16 SHA declared by the original CLIP local loader URL",
            "BPE_provenance": "CoOp-distributed CLIP tokenizer at the pinned source commit",
            "BPE_sha256": CLIP_BPE_SHA, "image_pixels_verified": False,
            "GPU_model_network_calls": 0, "dataset_images_copied": 0, "weights_copied": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-root")
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--image-roots-json", help="Optional explicit dataset-name to existing path mapping")
    parser.add_argument("--bpe-sha256", help="Optional independently known expected BPE content hash")
    parser.add_argument("--output", help="New binding JSON; no overwrite")
    parser.add_argument("--check-binding", help="Read-only content verification of an existing binding")
    parser.add_argument("--materialize-configs", help="Optional fresh directory for all configs with this binding applied")
    args = parser.parse_args(argv)
    if args.check_binding:
        original = json.loads(Path(args.check_binding).read_text(encoding="utf-8"))
        runtime = original["runtime"]
        value = bind(runtime["clip_root"], runtime["clip_checkpoint"], original.get("image_roots"), runtime["clip_bpe_sha256"])
        if runtime != value["runtime"]:
            raise ValueError("Existing binding runtime declaration differs from measured assets")
        binding_path = absolute_path(args.check_binding)
    else:
        if not args.clip_root or not args.clip_checkpoint or not args.output:
            parser.error("Supply explicit --clip-root, --clip-checkpoint, --output, or --check-binding")
        roots = json.loads(Path(args.image_roots_json).read_text(encoding="utf-8")) if args.image_roots_json else {}
        value = bind(args.clip_root, args.clip_checkpoint, roots, args.bpe_sha256)
        binding_path = absolute_path(args.output)
        if binding_path.exists():
            raise FileExistsError("External asset binding exists; inspect/check it instead of replacing it")
        write_json(binding_path, value)
    if args.materialize_configs:
        import yaml
        destination = absolute_path(args.materialize_configs)
        if destination.exists():
            raise FileExistsError("Materialized config directory must be fresh")
        destination.mkdir(parents=True)
        for path in sorted((ROOT / "configs").glob("*.yaml")):
            seed = 1
            config = load_config(path, seed)
            apply_asset_binding(config, binding_path)
            public = {key: item for key, item in config.items() if not key.startswith("_")}
            (destination / path.name).write_text(yaml.safe_dump(public, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps({"binding": str(binding_path), "sha256": file_hash(binding_path), "valid": True,
                      "GPU_model_network_calls": 0, "dataset_images_copied": 0, "weights_copied": 0,
                      "BPE_verified": True}))
    return value


if __name__ == "__main__":
    main()
