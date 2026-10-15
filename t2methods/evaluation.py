"""Full held-out evaluation with vocabulary changes and immutable state audit."""
import csv
import io
import json
import math
from pathlib import Path
import time

import torch
import numpy as np
from torch.utils.data import DataLoader

from .checkpoint import file_hash, model_state_hash
from .dataset_loader import SceneDataset
from .config import COMMON, read_json, require


def harmonic_mean(base, new):
    if not all(math.isfinite(x) and 0 <= x <= 100 for x in (base, new)):
        raise ValueError("Accuracy must be a finite percentage")
    return 0.0 if base + new == 0 else 2 * base * new / (base + new)


@torch.no_grad()
def evaluate(model, records, classnames, taxonomy, preprocess, config, prediction_path):
    if not records:
        raise ValueError("Empty evaluation cohort")
    before = model_state_hash(model)
    leaves = [taxonomy["classes"][name]["leaf"] for name in classnames]
    templates = [taxonomy["classes"][name]["template"] for name in classnames]
    model.set_classnames(leaves, templates)
    model.eval()
    device = config["train"]["device"]
    amp = device.startswith("cuda") and config["train"]["precision"] == "amp"
    loader = DataLoader(SceneDataset(records, preprocess), batch_size=config["train"]["eval_batch_size"],
                        shuffle=False, num_workers=config["train"]["workers"], pin_memory=amp)
    prediction_path = Path(prediction_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    correct, count, seen = 0, 0, set()
    expected_ids = {record["id"] for record in records}
    if len(expected_ids) != len(records):
        raise ValueError("Duplicate IDs in evaluation cohort")
    per_class = {index: {"correct": 0, "count": 0} for index in range(len(classnames))}
    start = time.monotonic()
    save_scores = bool(config.get("runtime", {}).get("save_scores", False))
    score_rows, logit_rows, image_rows, label_rows, id_rows = [], [], [], [], []
    with prediction_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label", "prediction"])
        with torch.autocast(device_type="cuda" if amp else "cpu", enabled=amp):
            texts = model.text_features()
            for images, labels, ids in loader:
                result = model(images.to(device, non_blocking=amp), text_features=texts)
                logits = result["logits"].float()
                if not torch.isfinite(logits).all() or logits.shape != (len(labels), len(classnames)):
                    raise FloatingPointError("Invalid/nonfinite evaluation logits")
                predictions = logits.argmax(1).cpu()
                if save_scores:
                    with torch.autocast(device_type=model.device.type, enabled=False):
                        cosine = result["image_features"].float() @ result["text_features"].float().T
                    score_rows.append(cosine.cpu().numpy())
                    logit_rows.append(logits.cpu().numpy())
                    label_rows.extend(labels.tolist()); id_rows.extend(ids)
                    if config["runtime"].get("save_features", False):
                        image_rows.append(result["image_features"].float().cpu().numpy())
                for identifier, label, prediction in zip(ids, labels.tolist(), predictions.tolist()):
                    if identifier in seen or identifier not in expected_ids or not 0 <= label < len(classnames):
                        raise ValueError("Evaluation ID or label exceeds declared cohort")
                    seen.add(identifier)
                    correct += int(label == prediction)
                    count += 1
                    per_class[label]["correct"] += int(label == prediction)
                    per_class[label]["count"] += 1
                    writer.writerow([identifier, label, prediction])
    if seen != expected_ids or count != len(records):
        raise RuntimeError("Incomplete evaluation")
    after = model_state_hash(model)
    if before != after:
        raise RuntimeError("Evaluation modified prompts or fixed method state")
    score_file = None
    if save_scores:
        score_file = prediction_path.with_name(prediction_path.stem + "_scores.npz")
        arrays = dict(ids=np.asarray(id_rows), labels=np.asarray(label_rows, dtype=np.int64),
                      classnames=np.asarray(leaves), cosine=np.concatenate(score_rows), logits=np.concatenate(logit_rows),
                      text_features=texts.detach().float().cpu().numpy(),
                      reference_text_features=model.reference_text().float().cpu().numpy())
        if image_rows: arrays["image_features"] = np.concatenate(image_rows)
        with score_file.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    return {"accuracy": 100 * correct / count, "correct": correct, "count": count,
            "classnames": leaves, "original_classnames": classnames, "per_class": per_class,
            "predictions_file": prediction_path.name, "scores_file": score_file.name if score_file else None,
            "scores_sha256": file_hash(score_file) if score_file else None, "seconds": time.monotonic() - start,
            "state_sha256_before": before, "state_sha256_after": after}


def evaluate_protocol(model, bundle, targets, preprocess, config, output):
    if config["runtime"].get("phase") == "screening":
        if config["runtime"].get("source_selection_version", 1) == 2:
            from .source_evaluation_v2 import evaluate_source_v2
            return evaluate_source_v2(model, bundle, preprocess, config, output)
        return evaluate_source_validation(model, preprocess, config, output)
    output = Path(output)
    taxonomy = json.loads(Path(config["data"]["taxonomy"]).read_text(encoding="utf-8"))
    task = config["data"]["task"]
    if task == "b2n":
        base = evaluate(model, bundle.base_test, bundle.base_classnames, taxonomy, preprocess, config, output / "base_predictions.csv")
        new = evaluate(model, bundle.new_test, bundle.new_classnames, taxonomy, preprocess, config, output / "new_predictions.csv")
        result = {"source": "measured", "task": task, "base": base, "new": new, "hm": harmonic_mean(base["accuracy"], new["accuracy"])}
    else:
        source = evaluate(model, bundle.test, bundle.test_classnames, taxonomy, preprocess, config, output / "source_predictions.csv")
        measured = {}
        for target_data, target in targets:
            if target.train or target.val or target.metadata.get("support_ids") or target.metadata.get("role") != "target":
                raise ValueError("Transfer evaluation target must have no training/validation/supports")
            target_taxonomy = json.loads(Path(target_data["taxonomy"]).read_text(encoding="utf-8"))
            name = target_data["dataset"]
            measured[name] = evaluate(model, target.test, target.test_classnames, target_taxonomy, preprocess, config,
                                      output / (name + "_predictions.csv"))
        if len(measured) != 3:
            raise ValueError("Transfer metric requires all three held-out targets")
        result = {"source": "measured", "task": task, "source_dataset": source, "targets": measured,
                  "target_mean": sum(item["accuracy"] for item in measured.values()) / 3,
                  "target_aggregation": "equal-domain mean; source excluded"}
    # Evaluation changes ephemeral text caches. Training vocabulary is restored.
    model.set_classnames([taxonomy["classes"][name]["leaf"] for name in bundle.train_classnames],
                         [taxonomy["classes"][name]["template"] for name in bundle.train_classnames])
    return result


def source_validation_records(config):
    from .dataset_loader import _ImageResolver
    manifest = read_json(config["source_validation"]["ids_file"])
    rows = manifest["records"]
    require(manifest.get("source_commit") == "42bc4654311da9773b59979dc71fe9f60534ab84" and len(rows) == 64,
            "Source validation manifest is not the predeclared 64 image cohort")
    official = {row[0]: (row[1], row[2]) for row in read_json(config["data"]["split"])["val"]}
    require(len({row["id"] for row in rows}) == 64, "Duplicate source validation IDs")
    counts = {name: 0 for name in COMMON}
    resolver = _ImageResolver(Path(config["data"]["image_root"]), config["data"].get("image_layout", "auto"), "patternnetv2")
    records = []
    for row in rows:
        label, name = official.get(row["id"], (None, None))
        require(type(label) is int and row["label"] == label and row["classname"] == name and COMMON[label] == name,
                "Source validation ID/label is outside official source val")
        counts[name] += 1
        path = resolver.resolve(row["id"])
        ledger = config.get("_dg_pixel_ledgers", {}).get("PatternNetv2", {}).get(row["id"])
        require(ledger is not None and Path(path).stat().st_size == ledger["bytes"] and file_hash(path) == ledger["sha256"],
                "Source validation image differs from official v2 pixel proof")
        records.append(dict(row, original_label=label, relative_path=row["id"], split="val", path=path))
    require(set(counts.values()) == {4}, "Source validation must contain four images per common class")
    return records


class SourceValidationView:
    """Fixed source-only corruption after the ordinary CLIP eval transform."""
    def __init__(self, preprocess, view):
        self.preprocess, self.view = preprocess, view

    def __call__(self, image):
        tensor = self.preprocess(image)
        if self.view == "clean":
            return tensor
        mean = tensor.new_tensor((0.48145466, 0.4578275, 0.40821073))[:, None, None]
        std = tensor.new_tensor((0.26862954, 0.26130258, 0.27577711))[:, None, None]
        rgb = (tensor * std + mean).clamp(0, 1)
        if self.view == "jpeg75":
            from PIL import Image
            import numpy as np
            encoded = io.BytesIO()
            Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(encoded, format="JPEG", quality=75)
            encoded.seek(0)
            with Image.open(encoded) as decoded:
                rgb = torch.from_numpy(np.array(decoded.convert("RGB"), copy=True)).permute(2, 0, 1).float() / 255
        elif self.view == "horizontal_rgb_gain_0.85_1.15":
            rgb = (rgb * torch.linspace(0.85, 1.15, rgb.shape[-1])[None, None, :]).clamp(0, 1)
        else:
            raise ValueError("Unknown registered source validation view")
        return (rgb - mean) / std


def evaluate_source_validation(model, preprocess, config, output):
    records = source_validation_records(config)
    taxonomy = read_json(config["data"]["taxonomy"])
    measured = {view: evaluate(model, records, COMMON, taxonomy, SourceValidationView(preprocess, view), config,
                               Path(output) / ("source_val_" + view + "_predictions.csv"))
                for view in config["source_validation"]["views"]}
    return {"source": "measured", "task": "dg", "phase": "screening", "source_validation": measured,
            "selection_score": sum(item["accuracy"] for item in measured.values()) / 3,
            "selection_metric": "equal-weight clean/JPEG75/horizontal multiplicative RGB gain accuracy",
            "source_validation_ids_sha256": config["source_validation"]["ids_sha256"],
            "source_validation_count": 64, "extra_validation_labels": 64,
            "target_evaluations": 0, "targets_consumed": False}
