"""Pinned scene splits and source-only supports for SPCP.

Original numeric labels and image IDs are retained. Classifier labels are local
to the returned vocabulary. This module never downloads, resplits, or opens
evaluation pixels while constructing supports; image decoding is in SceneDataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset


SceneRecord = Dict[str, Any]
PINNED_COMMIT = "42bc4654311da9773b59979dc71fe9f60534ab84"
PINNED_SPLITS = {
    "patternnet": ("patternnet.json", "db971a7a485c8c1d2d2e6e962e8bbadb5ec36977b4947c42bdad3c0168db29e4"),
    "rsicd": ("rsicd.json", "9d31238de620ab89152bd9ed2897ff8b3552d7bddf88a9c00b0e342c86a96a97"),
    "nwpu": ("resisc45.json", "9458dfb3e0c5cff5bfc305398d1237f2bb6bf6adb199c09229c3df0417ec1a70"),
    "mlrsnet": ("mlrsnet.json", "057e6764291156799ba85c14bf8835918880c986a6ef33fad28e7eb3ca651f68"),
}
_ALIASES = {"nwpu45": "nwpu", "resisc45": "nwpu", "nwpu-resisc45": "nwpu"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class DatasetBundle:
    """Source supports and separate full held-out classification vocabularies."""

    all_classnames: List[str]
    base_classnames: List[str]
    new_classnames: List[str]
    train_classnames: List[str]
    train: List[SceneRecord]
    val: List[SceneRecord]
    base_test: List[SceneRecord]
    new_test: List[SceneRecord]
    test: List[SceneRecord]
    metadata: Dict[str, Any]
    test_classnames: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.test_classnames is None:
            self.test_classnames = list(self.all_classnames)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_hash(value: Any) -> str:
    return _sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")).encode("utf-8"))


def _dataset_key(value: str) -> str:
    key = value.casefold().replace("_", "-")
    return _ALIASES.get(key, key)


def _configured_path(value: Any, base: Path, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty filesystem path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _safe_id(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Each image ID must be a non-empty relative path")
    # Normalize separators for overlap checks, while retaining the JSON ID below.
    path = PurePosixPath(value.replace("\\", "/"))
    if (path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts
            or path == PurePosixPath(".")):
        raise ValueError(f"Absolute or escaping image ID: {value!r}")
    return path.as_posix()


def _read_split(path: Path, expected_sha: Optional[str]) -> tuple:
    raw = path.read_bytes()
    digest = _sha256(raw)
    if expected_sha is not None:
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError("split_sha256 must be a complete SHA256 string")
        if digest.casefold() != expected_sha.casefold():
            raise ValueError(f"Split SHA256 mismatch for {path}")
    data = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(data, dict) or not all(k in data for k in ("train", "val", "test")):
        raise ValueError("Expected published train/val/test lists")
    partitions: Dict[str, List[SceneRecord]] = {}
    names: Dict[int, str] = {}
    used: Dict[str, str] = {}
    for partition in ("train", "val", "test"):
        if not isinstance(data[partition], list):
            raise ValueError(f"{partition} must be a list")
        records = []
        for row in data[partition]:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ValueError(f"Malformed {partition} row: expected path/label/classname")
            identifier, original_label, classname = row
            normalized_id = _safe_id(identifier)
            if normalized_id in used:
                raise ValueError(f"Duplicate image ID in {used[normalized_id]} and {partition}: {identifier}")
            used[normalized_id] = partition
            if isinstance(original_label, bool) or not isinstance(original_label, int) or original_label < 0:
                raise ValueError("Original labels must be non-negative integers")
            if not isinstance(classname, str) or not classname.strip():
                raise ValueError("Class names must be non-empty strings")
            if original_label in names and names[original_label] != classname:
                raise ValueError(f"Inconsistent classname for original label {original_label}")
            names[original_label] = classname
            records.append({"id": identifier, "relative_path": normalized_id,
                            "original_label": original_label, "classname": classname,
                            "split": partition, "path": None})
        partitions[partition] = records
    if len(names) < 2 or not partitions["test"]:
        raise ValueError("Expected at least two classes and a non-empty held-out test partition")
    # CLIP lowercases class text, so distinct labels cannot share the same name.
    seen_names = {}
    for label, name in names.items():
        key = " ".join(name.replace("_", " ").split()).casefold()
        if key in seen_names:
            raise ValueError(f"Classname collision between labels {seen_names[key]} and {label}")
        seen_names[key] = label
    return partitions, names, digest


def _sample_train(records: Sequence[SceneRecord], shot: int, seed: int) -> List[SceneRecord]:
    if isinstance(shot, bool) or not isinstance(shot, int) or shot < 1:
        raise ValueError("shot must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    groups: Dict[int, List[SceneRecord]] = {}
    for record in records:
        groups.setdefault(record["original_label"], []).append(record)
    rng = random.Random(seed)
    result = []
    # Matches Dassl/CoOp first-appearance group ordering. B2N filtering occurs
    # after this draw, preserving the source random stream without new supports.
    for label, candidates in groups.items():
        if len(candidates) < shot:
            raise ValueError(f"Original class {label} has {len(candidates)} train images, requires {shot}")
        result.extend(rng.sample(candidates, shot))
    return result


class _ImageResolver:
    def __init__(self, root: Path, layout: str, dataset_key: str):
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError(f"image_root must be a directory: {root}")
        if layout not in ("auto", "class_folders", "rsicd_flat"):
            raise ValueError(f"Unsupported image layout {layout!r}")
        self.layout = layout
        self.flat = layout == "rsicd_flat" or (layout == "auto" and dataset_key == "rsicd")
        self.basename_index: Optional[Dict[str, List[Path]]] = None

    def _inside(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"Image path escapes image_root: {path}") from error
        return resolved

    def resolve(self, identifier: str) -> str:
        relative = PurePosixPath(_safe_id(identifier))
        candidates = [self.root.joinpath(*relative.parts),
                      self.root.joinpath("images", *relative.parts)]
        if self.flat:
            candidates.extend([self.root / relative.name,
                               self.root / "images" / relative.name,
                               self.root / "RSICD_images" / relative.name])
        matches = {self._inside(p) for p in candidates if p.is_file()}
        if len(matches) > 1:
            raise ValueError(f"Ambiguous image candidates for {identifier}: {sorted(map(str, matches))}")
        if matches:
            return str(next(iter(matches)))
        if not self.flat:
            raise FileNotFoundError(f"Missing published image: {identifier} under {self.root}")
        if self.basename_index is None:
            index: Dict[str, List[Path]] = {}
            for path in sorted(self.root.rglob("*"), key=lambda p: p.as_posix()):
                if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES:
                    index.setdefault(path.name, []).append(self._inside(path))
            self.basename_index = index
        matches = set(self.basename_index.get(relative.name, []))
        if len(matches) != 1:
            if not matches:
                raise FileNotFoundError(f"Missing published RSICD image basename: {identifier}")
            raise ValueError(f"Ambiguous RSICD basename: {relative.name}")
        return str(next(iter(matches)))


def _remap(records: Sequence[SceneRecord], label_map: Mapping[int, int],
           paths: Mapping[str, Optional[str]]) -> List[SceneRecord]:
    return [dict(record, label=label_map[record["original_label"]],
                 path=paths.get(record["relative_path"])) for record in records
            if record["original_label"] in label_map]


def _task_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("task must be a string")
    value = value.casefold().replace("-", "_")
    aliases = {"base_to_new": "b2n", "few_shot": "fewshot", "cross_dataset": "cd",
               "dg_v2": "dg", "domain_generalization": "dg", "eval_only": "eval",
               "evaluation": "eval", "all": "eval"}
    value = aliases.get(value, value)
    if value not in ("b2n", "fewshot", "cd", "dg", "eval"):
        raise ValueError(f"Unsupported task {value!r}")
    return value


def _dg_inventory(data: dict, names: Mapping[int, str], dataset_name: str) -> tuple:
    if data.get("dg_verified") is not True or not dataset_name.casefold().endswith("v2"):
        raise ValueError("DG requires verified v2 split identity; raw datasets cannot be renamed to enable DG")
    common_names = data.get("common_classnames")
    mapping = data.get("common_class_map")
    if (not isinstance(common_names, list) or not common_names or
            not all(isinstance(x, str) and x.strip() for x in common_names) or not isinstance(mapping, dict)):
        raise ValueError("DG requires common_classnames and original-label-to-common-index common_class_map")
    if len({x.casefold() for x in common_names}) != len(common_names):
        raise ValueError("DG common_classnames must be unique")
    label_map = {}
    for key, value in mapping.items():
        try:
            original = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError("DG mapping keys must be original integer labels") from error
        if isinstance(key, bool) or str(original) != str(key) or original not in names:
            raise ValueError(f"Invalid DG original label {key!r}")
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(common_names):
            raise ValueError("DG mapping values must be valid common-vocabulary indices")
        if original in label_map:
            raise ValueError("Duplicate original labels in DG mapping")
        label_map[original] = value
    if sorted(label_map.values()) != list(range(len(common_names))):
        raise ValueError("DG mapping must provide exactly one original label for each shared class")
    return list(common_names), label_map


def load_dataset(config: dict, task: Optional[str] = None,
                 seed: Optional[int] = None) -> DatasetBundle:
    """Load one source bundle, or an explicitly eval-only target bundle.

    Relative split/image paths use ``config['_config_dir']`` or this module's
    directory. Set ``data.metadata_only`` for metadata checks before images are
    prepared; these records have ``path=None`` and cannot decode pixels. Targets
    must use task ``eval``/``all`` or ``data.eval_only=True``. ``cd`` otherwise
    samples only this source split and never accesses ``target_split``.
    """
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("Expected configuration with a data mapping")
    data = config["data"]
    dataset_name = data.get("dataset")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("data.dataset must be a non-empty name")
    key = _dataset_key(dataset_name)
    actual_task = _task_name(task if task is not None else data.get("task", "b2n"))
    eval_only = bool(data.get("eval_only", False)) or actual_task == "eval"
    if data.get("role", "source") == "target" and not eval_only:
        raise ValueError("Target data must be eval-only; target supports are prohibited")
    metadata_only = bool(data.get("metadata_only", False))
    base = Path(config.get("_config_dir", Path(__file__).resolve().parent))
    split_value = data.get("split")
    if split_value is None and key in PINNED_SPLITS:
        split_value = Path(__file__).resolve().parent.parent / "rccp_experiment" / "references" / "rs_benchmarks" / "applenet_42bc4654311d" / PINNED_SPLITS[key][0]
    split_path = _configured_path(split_value, base, "data.split").resolve(strict=True)
    expected_sha = data.get("split_sha256")
    if expected_sha is None and key in PINNED_SPLITS:
        expected_sha = PINNED_SPLITS[key][1]
    partitions, names, split_sha = _read_split(split_path, expected_sha)
    labels = sorted(names)
    if data.get("base_fraction", 0.5) != 0.5 or data.get("base_rounding", "ceil") != "ceil":
        raise ValueError("B2N uses ceil-half sorted integer class labels")
    cut = math.ceil(len(labels) / 2)
    base_ids, new_ids = labels[:cut], labels[cut:]
    maps = {"all": {label: i for i, label in enumerate(labels)},
            "base": {label: i for i, label in enumerate(base_ids)},
            "new": {label: i for i, label in enumerate(new_ids)}}
    vocabularies = {"all": [names[x] for x in labels], "base": [names[x] for x in base_ids],
                    "new": [names[x] for x in new_ids]}
    train_map = maps["base"] if actual_task == "b2n" and not eval_only else maps["all"]
    train_names = vocabularies["base"] if actual_task == "b2n" and not eval_only else vocabularies["all"]
    test_map, test_names = maps["all"], vocabularies["all"]
    if actual_task == "dg" or data.get("dg_verified"):
        if split_sha in {entry[1] for entry in PINNED_SPLITS.values()}:
            raise ValueError("Raw pinned split cannot be renamed to a verified v2 DG split")
        train_names, train_map = _dg_inventory(data, names, dataset_name)
        test_map, test_names = train_map, train_names
    maps["train"], maps["test"] = train_map, test_map
    chosen_seed = seed if seed is not None else data.get("seed", config.get("seed"))
    if chosen_seed is None:
        configured_seeds = config.get("train", {}).get("seeds", [])
        if len(configured_seeds) != 1 and not eval_only:
            raise ValueError("Supply an explicit seed when a multi-seed training configuration is used")
        chosen_seed = configured_seeds[0] if configured_seeds else 1
    if isinstance(chosen_seed, bool) or not isinstance(chosen_seed, int):
        raise ValueError("seed must be an integer")
    shot = data.get("shot")
    train_labels = {x["original_label"] for x in partitions["train"]}
    test_labels = {x["original_label"] for x in partitions["test"]}
    if not set(test_map).issubset(test_labels):
        raise ValueError("The held-out test partition does not cover every evaluation class")
    if eval_only:
        supports = []
    else:
        if not set(train_map).issubset(train_labels):
            raise ValueError("Every training class needs source training records")
        support_candidates = partitions["train"]
        # DG only uses the already verified common-class source split.
        if actual_task == "dg":
            support_candidates = [r for r in support_candidates if r["original_label"] in train_map]
        supports = [r for r in _sample_train(support_candidates, shot, chosen_seed)
                    if r["original_label"] in train_map]
    use_val = bool(data.get("use_val", False)) and not eval_only
    validation = [r for r in partitions["val"] if r["original_label"] in train_map] if use_val else []
    used_records = supports + validation + partitions["test"]
    paths: Dict[str, Optional[str]] = {}
    if not metadata_only:
        root = _configured_path(data.get("image_root"), base, "data.image_root")
        resolver = _ImageResolver(root, data.get("image_layout", "auto"), key)
        physical_ids: Dict[str, str] = {}
        for record in used_records:
            relative = record["relative_path"]
            if relative in paths:
                continue
            resolved = resolver.resolve(relative)
            if resolved in physical_ids:
                raise ValueError(f"Different image IDs resolve to the same physical path: {physical_ids[resolved]} and {record['id']}")
            physical_ids[resolved] = record["id"]
            paths[relative] = resolved
    train = _remap(supports, train_map, paths)
    val = _remap(validation, train_map, paths)
    base_test = _remap(partitions["test"], maps["base"], paths)
    new_test = _remap(partitions["test"], maps["new"], paths)
    test = _remap(partitions["test"], test_map, paths)
    support_ids = [r["id"] for r in train]
    metadata = {
        "dataset": dataset_name, "task": actual_task, "role": "target" if eval_only else "source",
        "seed": chosen_seed, "shot": shot if not eval_only else None,
        "split_path": str(split_path), "split_sha256": split_sha,
        "split_commit": PINNED_COMMIT if key in PINNED_SPLITS and split_sha == PINNED_SPLITS[key][1] else None,
        "paper_exact_split_identity_confirmed": False, "metadata_only": metadata_only,
        "image_paths_verified": not metadata_only, "image_pixels_verified": False,
        "image_root": str(data.get("image_root")) if data.get("image_root") is not None else None,
        "support_ids": support_ids, "support_sha256": _stable_hash(support_ids),
        "support_record_sha256": _stable_hash([[r["id"], r["original_label"], r["label"]] for r in train]),
        "class_ids": {"all": labels, "base": base_ids, "new": new_ids,
                      "train": [x for x, _ in sorted(train_map.items(), key=lambda item: item[1])],
                      "test": [x for x, _ in sorted(test_map.items(), key=lambda item: item[1])]},
        "class_maps": maps,
        "vocabularies": dict(vocabularies, train=list(train_names), test=list(test_names)),
        "original_classnames": dict(names),
        "partition_counts": {name: len(records) for name, records in partitions.items()},
        "returned_counts": {"train": len(train), "val": len(val), "base_test": len(base_test),
                            "new_test": len(new_test), "test": len(test)},
        "sampling": "local random.Random(seed), JSON first-appearance classes, without replacement; sample all train classes before B2N filtering",
        "base_new_split": "sorted original integer labels; first ceil(C/2) base",
        "validation_policy": "complete original source validation partition, no selection/tuning" if use_val else "disabled",
        "target_supports_generated": False, "target_config_consumed": False,
        "evaluation_partition": "complete original test for the declared evaluation vocabulary",
        "dg_verified": bool(data.get("dg_verified", False)),
    }
    return DatasetBundle(list(vocabularies["all"]), list(vocabularies["base"]),
                         list(vocabularies["new"]), list(train_names), train, val,
                         base_test, new_test, test, metadata, list(test_names))


class SceneDataset(Dataset):
    """Decode RGB images from a prepared record list, retaining original IDs."""

    def __init__(self, records: Sequence[SceneRecord],
                 transform: Optional[Callable] = None):
        self.records = list(records)
        self.transform = transform
        for record in self.records:
            if record.get("path") is None:
                raise ValueError("Metadata-only records cannot decode images; resolve image_root first")
            if isinstance(record.get("label"), bool) or not isinstance(record.get("label"), int):
                raise ValueError("Each record needs a mapped integer label")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple:
        record = self.records[index]
        with Image.open(record["path"]) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record["label"], record["id"]


class DatasetLoader:
    """Convenience namespace for loading a scene-classification dataset."""

    load_dataset = staticmethod(load_dataset)
