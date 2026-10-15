"""Five fixed source views and auditable continuous, class-balanced CE risk."""
from pathlib import Path
import numpy as np
import torch
from .selection_protocol_v2 import VIEWS, RULE, RULE_SHA
from .evaluation import SourceValidationView, source_validation_records, evaluate
from .round2_views import MEAN, STD, resolution, gaussian_blur
from .config import COMMON, read_json, require
from .checkpoint import write_json, file_hash


class SourceViewV2(SourceValidationView):
    def __call__(self, image):
        if self.view in VIEWS[:3]:
            return super().__call__(image)
        tensor = self.preprocess(image)
        if tensor.shape != (3, 224, 224):
            raise ValueError("Source v2 requires unchanged 224-pixel CLIP preprocessing")
        mean, std = tensor.new_tensor(MEAN)[:, None, None], tensor.new_tensor(STD)[:, None, None]
        rgb = (tensor*std + mean).clamp(0, 1).unsqueeze(0)
        with torch.autocast(device_type=tensor.device.type, enabled=False):
            if self.view == VIEWS[3]: rgb = resolution(rgb.float(), 160)
            elif self.view == VIEWS[4]: rgb = gaussian_blur(rgb.float(), .7)
            else: raise ValueError("Unknown v2 view")
        return (rgb.squeeze(0).clamp(0, 1)-mean)/std


def risk_metrics(cosine, labels, ids, classes=16, per_class=4):
    cosine, labels = np.asarray(cosine, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    require(cosine.shape == (5, classes*per_class, classes) and len(ids) == len(labels) == classes*per_class,
            "v2 score shape/cohort differs")
    require(len(set(ids)) == len(ids) and np.isfinite(cosine).all() and np.max(np.abs(cosine)) <= 1.0001,
            "Invalid source cosine scores/IDs")
    require(np.array_equal(np.bincount(labels, minlength=classes), np.full(classes, per_class)),
            "v2 source class balance differs")
    logits = RULE["cosine_scale"] * cosine
    centered = logits - logits.max(-1, keepdims=True)
    ce = np.log(np.exp(centered).sum(-1)) - np.take_along_axis(centered, labels[None, :, None], axis=2).squeeze(-1)
    per_image = .5*ce[0] + .5*ce[1:].max(0)
    true = np.take_along_axis(cosine, labels[None, :, None], axis=2).squeeze(-1)
    others = cosine.copy(); others[:, np.arange(len(labels)), labels] = -np.inf
    margins = true - others.max(-1)
    balance = lambda values: float(np.mean([values[labels == c].mean() for c in range(classes)]))
    return dict(risk=balance(per_image), clean_ce=balance(ce[0]), worst_corruption_ce=balance(ce[1:].max(0)),
                per_view={v: dict(accuracy=float(100*np.mean(cosine[i].argmax(-1) == labels)),
                                  ce=balance(ce[i]), margin=balance(margins[i])) for i, v in enumerate(VIEWS)},
                per_class=[dict(label=c, count=int(np.sum(labels == c)), risk=float(per_image[labels == c].mean())) for c in range(classes)],
                per_image=[dict(id=identifier, label=int(labels[i]), risk=float(per_image[i]),
                                ce=ce[:, i].tolist(), margin=margins[:, i].tolist()) for i, identifier in enumerate(ids)])


def recompute_scores(output, measured, expected_names=None):
    arrays, first_ids, first_labels, first_names = [], None, None, None
    require(list(measured) == VIEWS, "v2 view order differs")
    for view in VIEWS:
        row = measured[view]
        path = Path(output) / row["scores_file"]
        require(path.parent.absolute() == Path(output).absolute() and file_hash(path) == row["scores_sha256"], "Source score file identity differs")
        with np.load(path, allow_pickle=False) as values:
            ids, labels, names = values["ids"].tolist(), values["labels"].copy(), values["classnames"].tolist()
            if first_ids is None: first_ids, first_labels, first_names = ids, labels, names
            require(expected_names is None or names == expected_names, "Source score class-column order differs from the taxonomy")
            require(ids == first_ids and np.array_equal(labels, first_labels) and names == first_names, "Source views have different ID/label/class order")
            arrays.append(values["cosine"].copy())
    return risk_metrics(np.stack(arrays), first_labels, first_ids)


def evaluate_source_v2(model, bundle, preprocess, config, output):
    require(config["runtime"].get("save_scores") is True, "v2 must retain continuous scores")
    records = source_validation_records(config)
    require(not ({r["id"] for r in records} & {r["id"] for r in bundle.train}), "Source validation overlaps training supports")
    taxonomy, output = read_json(config["data"]["taxonomy"]), Path(output)
    measured = {view: evaluate(model, records, COMMON, taxonomy, SourceViewV2(preprocess, view), config,
                              output / ("source_val_"+view+"_predictions.csv")) for view in VIEWS}
    metric = recompute_scores(output, measured, [taxonomy["classes"][c]["leaf"] for c in COMMON])
    pixels = [{"id": r["id"], "label": r["label"], "sha256": file_hash(r["path"])} for r in records]
    write_json(output / "source_validation_pixels.json", pixels)
    write_json(output / "source_validation_risk.json", metric)
    return dict(source="measured", task="dg", phase="screening", selection_schema="t2methods.source_selection.v2",
                source_validation=measured, selection_score=metric["risk"], selection_clean_ce=metric["clean_ce"],
                selection_direction="minimize", selection_rule_sha256=RULE_SHA,
                source_validation_ids_sha256=config["source_validation"]["ids_sha256"], source_validation_count=64,
                source_validation_pixels_sha256=file_hash(output / "source_validation_pixels.json"),
                extra_validation_labels=64, new_validation_labels=0, targets_consumed=False, target_evaluations=0)
