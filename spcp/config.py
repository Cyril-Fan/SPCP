"""Configuration and pinned protocol checks; no model runtime imports."""
import copy
import json
import math
from pathlib import Path

from .checkpoint import absolute_path, file_hash, json_hash, source_hashes

COMMIT = "42bc4654311da9773b59979dc71fe9f60534ab84"
CLIP_WEIGHT_SHA = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
CLIP_BPE_SHA = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
CLIP_SOURCE_SHA = {
    "clip/__init__.py": "d7930eb290ebed432116795c91473200d098451a94c0aedd4ae0f934f97955e8",
    "clip/clip.py": "f42d29d6ff24715a4d3dd6860c820ea21a3839e3a45b42ad92c7bc37019cd3c6",
    "clip/model.py": "c0ba5cb62eafbf21e6c4e861e9caf6933d1c7e82313d37f953cd2c1ae85f78de",
    "clip/simple_tokenizer.py": "d1b09f10ed3d1e2c343619cb98cbfacb92363dd8607526a43aa7fe2d60d15bad",
}
RAW_SPECS = {
    "PatternNet": ("db971a7a485c8c1d2d2e6e962e8bbadb5ec36977b4947c42bdad3c0168db29e4", "87a1ecb96698adae0125f23a43f55262fbe9d9f5f439161b5ab497fd67338da2"),
    "RSICD": ("9d31238de620ab89152bd9ed2897ff8b3552d7bddf88a9c00b0e342c86a96a97", "e431249914169baf8456140e4d113cbd3d78d15a5f25576867a52cf5c18e0490"),
    "NWPU-RESISC45": ("9458dfb3e0c5cff5bfc305398d1237f2bb6bf6adb199c09229c3df0417ec1a70", "8850f1a2e58175a640abc84879914fd88d1e0150ca7523588f525e41132f1133"),
    "MLRSNet": ("057e6764291156799ba85c14bf8835918880c986a6ef33fad28e7eb3ca651f68", "814c1370a3da3fac359c35bfe7af3e0e80f3712cf76c02330721978678c795ee"),
}
COMMON = ["baseballfield", "sparse_residential", "harbor", "intersection", "forest", "bridge",
          "overpass", "railway", "parking", "meadow", "river", "desert", "stadium", "storage_tank",
          "dense_residential", "beach"]
DG_TAXONOMY_SHA = "f6ea2f64da13e7563239a729599a156969e6cfcb15ae2d6ed9863f5dc9f5a5d0"
DG_PREPARATION_SHA = "d753d94f5c5a64212d9e370b5d29f61abfb91e49948955836e5786300799595c"
DG_SPECS = {
    "PatternNetv2": ("2db9f8f14226c52f5d857e0d323b1cc21761ed2d1a378b7cd520e52d1c1b8df0", {"train": 6272, "val": 2688, "test": 3840}, "1K-GZ2KjQ3hn17JJBrxnmXsTxAFeg2XUT"),
    "RSICDv2": ("191adf02a1a76d64e4e840fa8383bd141ab2b45f93ef29263a5d5fc25ea2ea77", {"train": 2620, "val": 1132, "test": 1608}, "1uhlTHQCHkE0KD04YGBAKsxPgG14eQez_"),
    "NWPU-RESISC45v2": ("8e8c7bf1b316e0c6936dcba33a393903c52910ba375ef5fed47bf577319d0f49", {"train": 5488, "val": 2352, "test": 3360}, "1Zfsko5swyQqu5HiuRwZe5jIGoUKfBgxq"),
    "MLRSNetv2": ("759e826e7f706afc822253947c4dd6926a9e5906664685b4f5d44dcdf12ca4d6", {"train": 19163, "val": 8219, "test": 11741}, "1OJrAwU1i9hYe7kEsHIIq_TodJDiwnnAz"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(value, base):
    path = Path(value).expanduser()
    return str(absolute_path(path if path.is_absolute() else base / path))


def load_config(path, seed=1, overrides=None):
    import yaml
    # Resolve metadata paths without requiring image directories to exist.
    path = Path(path).absolute()
    require(path.is_file(), "Configuration file is unavailable")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "Configuration must be a mapping")
    config = copy.deepcopy(config)
    for section, updates in (overrides or {}).items():
        config.setdefault(section, {}).update(updates)
    config["_config_dir"], config["_config_path"], config["_seed"] = str(path.parent), str(path), seed
    for section in [config.get("data", {})] + config.get("targets", []):
        for key in ("split", "taxonomy", "image_root", "v2_protocol_manifest", "v2_data_ready", "ledger_root"):
            if section.get(key) is not None:
                section[key] = _resolve(section[key], path.parent)
    runtime = config.get("runtime", {})
    for key in ("clip_root", "clip_checkpoint", "output_root"):
        if runtime.get(key) is not None:
                runtime[key] = _resolve(runtime[key], path.parent)
    if config.get("source_validation", {}).get("ids_file"):
        config["source_validation"]["ids_file"] = _resolve(config["source_validation"]["ids_file"], path.parent)
    if config.get("method", {}).get("uniform_reference_checkpoint"):
        config["method"]["uniform_reference_checkpoint"] = _resolve(config["method"]["uniform_reference_checkpoint"], path.parent)
    train = config.get("train", {})
    if config.get("data", {}).get("task") != "b2n":
        train["epochs"] = train.get("other_task_epochs", 10)
    validate_config(config)
    config["_source_hashes"] = source_hashes()
    config["_request_identity"] = request_identity(config)
    return config


def _check_taxonomy(data):
    require(Path(data["split"]).is_file() and Path(data["taxonomy"]).is_file(), "Split/taxonomy file is unavailable")
    taxonomy = read_json(data["taxonomy"])
    require(isinstance(taxonomy.get("classes"), dict), "Taxonomy classes missing")
    for name, values in taxonomy["classes"].items():
        require(isinstance(values.get("leaf"), str) and isinstance(values.get("template"), str), "Invalid taxonomy class: " + name)
    if data.get("split_sha256"):
        require(file_hash(data["split"]) == data["split_sha256"], "Split SHA differs")
    if data.get("dataset") in RAW_SPECS:
        split_sha, taxonomy_sha = RAW_SPECS[data["dataset"]]
        require(file_hash(data["split"]) == split_sha and file_hash(data["taxonomy"]) == taxonomy_sha,
                "Raw split/name-only taxonomy differs from the published asset")


def validate_config(config):
    for section in ("model", "method", "data", "train", "runtime"):
        require(isinstance(config.get(section), dict), "Missing configuration section: " + section)
    method, data, train, model = [config[key] for key in ("method", "data", "train", "model")]
    require(method.get("name") == "SPCP", "This SPCP release exposes the full method only")
    from .settings import settings
    config["method"] = method = settings(method)
    require(method == settings({"name": "SPCP"}), "Published configs require the full eta=0.50 SPCP setting")
    require(config["runtime"].get("clip_bpe_sha256") == CLIP_BPE_SHA, "CLIP BPE identity differs")
    require(type(method.get("enabled", True)) is bool, "method.enabled must be boolean")
    require(config["runtime"].get("clip_source_hashes") == CLIP_SOURCE_SHA and config["runtime"].get("clip_checkpoint_sha256") == CLIP_WEIGHT_SHA,
            "External original CLIP code/weight SHA guards must match the published CLIP assets")
    require(data.get("task") in ("b2n", "cd", "dg"), "Only B2N/CD/DG are registered")
    require(type(config.get("_seed")) is int and config["_seed"] in train.get("seeds", []), "Seed is outside declared inventory")
    require(train.get("seeds") == [1, 2, 3], "Registered protocol requires seeds 1,2,3")
    require(data.get("shot") == 16 and data.get("base_fraction", 0.5) == 0.5 and data.get("base_rounding", "ceil") == "ceil", "Keep sixteen shots and ceil-half sorted labels")
    require(data.get("role", "source") == "source" and not data.get("eval_only", False), "Training source must not be an eval target")
    require(not data.get("use_val", False), "Full source validation is disabled; explicit registered method subset must be separate")
    require(model.get("backbone") == "ViT-B/16" and model.get("prompt_length") == 16 and model.get("shallow_layers") == 6 and model.get("depth") == 12, "SPCP backbone/prompt geometry differs")
    require(model.get("interaction") == "directional_shared_projection", "SPCP interaction sharing differs")
    require(model.get("alignment_metric") == "cosine" and all(model.get(key) == 1.0 for key in ("alignment_weight", "text_alignment_weight", "image_alignment_weight")), "Keep original cosine alignment weights")
    exact = {"epochs": 50 if data["task"] == "b2n" else 10, "other_task_epochs": 10, "optimizer": "Adam", "learning_rate": 0.0025, "weight_decay": 0.00001,
             "batch_size": 16, "eval_batch_size": 64, "scheduler": "cosine", "warmup_epochs": 1, "warmup_learning_rate": 0.00001,
             "crop_min": 0.08, "horizontal_flip": 0.5, "grad_scaler_initial_scale": 128, "checkpoint_selection": "final_epoch"}
    for key, value in exact.items():
        require(train.get(key) == value, "Published training recipe differs: " + key)
    require(type(train.get("workers")) is int and train["workers"] >= 0, "workers must be a nonnegative integer")
    require(train.get("device") in ("cpu", "cuda") and train.get("precision") in ("fp32", "amp"), "Invalid device/precision")
    minutes = train.get("max_minutes_per_run")
    require(isinstance(minutes, (int, float)) and not isinstance(minutes, bool) and math.isfinite(minutes) and 0 < minutes <= 110, "Invalid per-fit time limit")
    _check_taxonomy(data)
    targets = config.get("targets", [])
    require(isinstance(targets, list), "targets must be a list")
    if data["task"] == "b2n":
        require(not targets and data["dataset"] in ("PatternNet", "RSICD", "NWPU-RESISC45", "MLRSNet"), "B2N source/targets differ")
    else:
        expected_source = "PatternNetv2" if data["task"] == "dg" else "PatternNet"
        expected_targets = ["RSICDv2", "NWPU-RESISC45v2", "MLRSNetv2"] if data["task"] == "dg" else ["RSICD", "NWPU-RESISC45", "MLRSNet"]
        require(data["dataset"] == expected_source and [t.get("dataset") for t in targets] == expected_targets, "Transfer inventory requires one PatternNet source and the three ordered targets")
        for target in targets:
            require(target.get("role") == "target" and target.get("eval_only") is True and not target.get("use_val", False), "Target must be evaluation only without supports/validation")
            _check_taxonomy(target)
    if data["task"] == "dg":
        verify_dg_metadata(config)
    require(config["runtime"].get("phase") == "formal", "Only full-method training/evaluation is released")
    require(config["runtime"].get("experiment_protocol") == "spcp." + data["task"] + ".v1", "Protocol identifier differs")
    require(not config.get("source_validation") and not config["runtime"].get("require_source_selection", False),
            "The released protocol uses fixed settings without a source-validation selection step")
    return {"valid": True, "method": method["name"], "task": data["task"], "dataset": data["dataset"],
            "seed": config["_seed"], "source_fits": 1, "target_evaluations": len(targets), "runtime_loaded": False}


def verify_dg_metadata(config):
    data = config["data"]
    protocol = read_json(data["v2_protocol_manifest"])
    require(protocol.get("metadata_verified") is True and protocol.get("source_commit") == COMMIT,
            "DG pinned APPLeNet v2 protocol is unverified")
    require(protocol.get("common_classnames") == COMMON and protocol.get("v2_label_to_common_class_id") == {str(i): i for i in range(16)}, "DG common-class identity differs")
    for item in [data] + config["targets"]:
        dataset = item["dataset"]
        digest, counts, _ = DG_SPECS[dataset]
        require(item.get("dg_verified") is True and item.get("common_classnames") == COMMON and {str(k): v for k, v in item.get("common_class_map", {}).items()} == {str(i): i for i in range(16)}, "DG verified class map missing")
        require(file_hash(item["split"]) == digest and item.get("split_sha256") == digest, "DG official split SHA differs: " + dataset)
        require(file_hash(item["taxonomy"]) == DG_TAXONOMY_SHA, "DG shared name-only taxonomy differs")
        taxonomy = read_json(item["taxonomy"])
        require(taxonomy.get("official_common_classnames") == COMMON and taxonomy.get("uses_image_pixels_or_scores") is False and set(taxonomy["classes"]) == set(COMMON), "DG taxonomy identity differs")
        split = read_json(item["split"])
        require(set(split) == set(counts) and {key: len(rows) for key, rows in split.items()} == counts, "DG official partition counts differ")
        seen = set()
        for rows in split.values():
            for identifier, label, name in rows:
                require(type(label) is int and 0 <= label < 16 and COMMON[label] == name and identifier not in seen, "DG row/vocabulary identity differs")
                seen.add(identifier)


def verify_dg_pixels(config, bundles):
    """Verify the published DG image ledgers and the pixels used by this run.

    Explicit ledger_root remaps remote receipt paths without relaxing content
    identity. This is called only by a real execution, never by a dry run.
    """
    data = config["data"]
    ready_path = Path(data["v2_data_ready"])
    marker = read_json(ready_path)
    require(marker.get("complete") is True and marker.get("status") == "complete" and marker.get("mode") == "full_prepare" and marker.get("full_dg_images_prepared") is True and marker.get("source_commit") == COMMIT and marker.get("helper_sha256") == DG_PREPARATION_SHA, "DG full image preparation receipt is missing/unverified")
    receipts = marker.get("results", [])
    require(len(receipts) == 4 and {r["dataset"] for r in receipts} == set(DG_SPECS), "DG receipt must cover all four domains")
    receipts = {r["dataset"]: r for r in receipts}
    for item, bundle in zip([data] + config["targets"], bundles):
        dataset = item["dataset"]
        digest, counts, archive_id = DG_SPECS[dataset]
        receipt = receipts[dataset]
        require(receipt.get("source_commit") == COMMIT and receipt.get("split_sha256") == digest and receipt.get("counts") == counts and receipt.get("classnames") == COMMON and receipt.get("images_verified") == sum(counts.values()), "DG pixel receipt coverage differs")
        require(all(receipt.get(k) is True for k in ("rgb_decode_complete", "official_published_json_coverage_verified", "zip_all_members_crc_verified")) and receipt["archive"].get("file_id") == archive_id, "DG RGB/archive/ZIP coverage proof differs")
        ledger = Path(item.get("ledger_root", data.get("ledger_root", ready_path.parent))) / Path(receipt["ledger_path"]).name
        require(file_hash(ledger) == receipt["ledger_sha256"], "DG pixel ledger SHA differs")
        original = {row[0]: {"partition": part, "label": row[1], "classname": row[2]} for part, rows in read_json(item["split"]).items() for row in rows}
        ledger_rows = {}
        with ledger.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                require(row["id"] in original and row["id"] not in ledger_rows and all(row[key] == value for key, value in original[row["id"]].items()), "DG pixel ledger row identity differs")
                require(isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64 and type(row.get("bytes")) is int and row["bytes"] > 0 and row.get("decoded_mode") == "RGB", "DG pixel ledger proof incomplete")
                ledger_rows[row["id"]] = row
        require(set(ledger_rows) == set(original), "DG full ledger coverage differs")
        require(bundle.test_classnames == COMMON and bundle.train_classnames == COMMON and not bundle.val and len(bundle.test) == counts["test"], "DG bundle cohort/vocabulary differs")
        if item.get("role", "source") == "target":
            require(not bundle.train and not bundle.metadata["support_ids"], "DG target generated supports")
        else:
            require(len(bundle.train) == 256, "DG source must have exactly 256 supports")
        for record in bundle.train + bundle.test:
            row = ledger_rows[record["id"]]
            path = Path(record["path"])
            require(path.stat().st_size == row["bytes"] and file_hash(path) == row["sha256"], "DG local image bytes differ: " + record["id"])
        config.setdefault("_dg_pixel_ledgers", {})[dataset] = ledger_rows


def configuration_identity(config):
    public = copy.deepcopy({k: v for k, v in config.items() if not k.startswith("_")})
    validation = public.get("source_validation", {})
    if validation.get("ids_file"):
        validation["ids_file"] = {"sha256": file_hash(validation["ids_file"])}
    # Physical paths may be relocated explicitly; referenced contents are fixed.
    for item in [public["data"]] + public.get("targets", []):
        for key in ("split", "taxonomy", "v2_protocol_manifest", "v2_data_ready"):
            if item.get(key) is not None:
                path = Path(item[key])
                item[key] = {"sha256": file_hash(path)} if path.is_file() else {"unavailable": path.name}
        for key in ("image_root", "ledger_root"):
            item.pop(key, None)
    runtime = public["runtime"]
    for key in ("phase", "candidate", "selection_receipt", "require_source_selection", "require_winner"):
        runtime.pop(key, None)
    checkpoint = runtime.get("clip_checkpoint")
    runtime["clip_checkpoint_identity"] = (file_hash(checkpoint) if checkpoint and Path(checkpoint).is_file()
                                           else runtime.get("clip_checkpoint_sha256", "unavailable_explicit_checkpoint"))
    clip_root = runtime.get("clip_root")
    package = Path(clip_root) / "clip" if clip_root else None
    runtime["clip_source_identity"] = ({p.relative_to(package).as_posix(): file_hash(p) for p in sorted(package.rglob("*"))
                                       if p.is_file() and (p.suffix == ".py" or p.name.endswith(".txt.gz"))}
                                      if package and package.is_dir() else runtime.get("clip_source_hashes", {}))
    for key in ("clip_root", "clip_checkpoint", "output_root"):
        runtime.pop(key, None)
    # YAML class maps have integer keys; JSON checkpoint metadata has string
    # keys. Canonicalize before hashing so serialization/relocation is stable.
    return json.loads(json.dumps(public, ensure_ascii=False, allow_nan=False))


def request_identity(config):
    return {"configuration_sha256": json_hash(configuration_identity(config)), "seed": config["_seed"],
            "source_hashes": config.get("_source_hashes", source_hashes())}


def target_config(config, target):
    value = copy.deepcopy(config)
    value["data"] = copy.deepcopy(target)
    value["data"].update(task="eval", eval_only=True, role="target", use_val=False)
    return value




def apply_asset_binding(config, path):
    binding = read_json(path)
    require(binding.get("schema") == "spcp.external_assets.v1", "Unknown external asset binding schema")
    runtime = binding.get("runtime", {})
    require(runtime.get("clip_source_hashes") == CLIP_SOURCE_SHA and runtime.get("clip_checkpoint_sha256") == CLIP_WEIGHT_SHA,
            "External asset binding original source/weight identity differs")
    require(isinstance(runtime.get("clip_bpe_sha256"), str) and len(runtime["clip_bpe_sha256"]) == 64,
            "External binding BPE identity is missing")
    base = Path(path).absolute().parent
    for key in ("clip_root", "clip_checkpoint"):
        runtime[key] = _resolve(runtime[key], base)
    config["runtime"].update(runtime)
    image_roots = binding.get("image_roots", {})
    for data in [config["data"]] + config.get("targets", []):
        if data["dataset"] in image_roots:
            data["image_root"] = _resolve(image_roots[data["dataset"]], base)
    config["_asset_binding"] = {"path": str(Path(path).absolute()), "sha256": file_hash(path)}
    validate_config(config)
    config["_request_identity"] = request_identity(config)
    return binding
