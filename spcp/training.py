"""SPCP Adam/AMP training with strict epoch-boundary resume."""
import json
import math
from pathlib import Path
import random
import time
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .checkpoint import (SCHEMA, absolute_path, backbone_hash, capture_rng, completed_run, file_hash, finish_run, json_hash,
                         load_checkpoint, model_state_hash, restore_model, restore_rng,
                         save_checkpoint, validate_epoch_resume, write_json)
from .dataset_loader import SceneDataset
from .evaluation import evaluate_protocol


def seed_everything(seed, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed touches CUDA generators even for CPU; keep CPU local.
    torch.random.default_generator.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(_index):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def source_identity(bundle):
    keys = ("dataset", "task", "role", "seed", "shot", "split_sha256", "support_ids", "support_sha256",
            "support_record_sha256", "class_ids", "class_maps", "vocabularies", "partition_counts", "dg_verified")
    return {key: bundle.metadata.get(key) for key in keys}


class Trainer:
    def __init__(self, model, bundle, config, output, targets=None):
        self.model, self.bundle, self.config, self.output = model, bundle, config, Path(output)
        self.targets = targets or []
        self.preprocess = config["_eval_preprocess"]
        self.taxonomy = json.loads(Path(config["data"]["taxonomy"]).read_text(encoding="utf-8"))
        self._training_classnames()
        if bundle.metadata.get("role") != "source" or not bundle.train or bundle.val:
            raise ValueError("Training requires source supports with no full validation partition")
        self.parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not self.parameters or any(name.startswith("clip_model.") for name, _ in self.parameters):
            raise ValueError("Frozen CLIP must not be optimized")
        self.initial_backbone = backbone_hash(model)
        self.metadata = {"schema": SCHEMA, "model_name": config["method"]["name"], "ours": True,
                         "task": config["data"]["task"], "dataset": config["data"]["dataset"], "seed": config["_seed"],
                         "request_identity": config["_request_identity"], "source_identity": source_identity(bundle),
                         "clip_source_hashes": config["_clip_source_hashes"], "clip_checkpoint_sha256": config["_clip_checkpoint_sha256"],
                         "taxonomy_sha256": file_hash(config["data"]["taxonomy"]),
                         "frozen_backbone_before": self.initial_backbone,
                         "trainable_parameters": {name: int(p.numel()) for name, p in self.parameters},
                         "config": {k: v for k, v in config.items() if not k.startswith("_")},
                         "checkpoint_selection": "final_epoch",
                         "resume_contract": "completed epoch only; same configuration/device/worker recipe; no mid-batch cursor"}
        import torchvision
        import PIL
        self.metadata["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__, "pillow": PIL.__version__,
            "cuda_runtime": torch.version.cuda, "device": config["train"]["device"],
            "source_view_contract": config.get("source_validation", {}).get("rule_sha256")}

    def _training_classnames(self):
        self.model.set_classnames([self.taxonomy["classes"][name]["leaf"] for name in self.bundle.train_classnames],
                                 [self.taxonomy["classes"][name]["template"] for name in self.bundle.train_classnames])

    def _store_preparation(self, saved=None):
        self.metadata["method_preparation"] = self.model.method_state_dict().get("receipt", {})

    def _validate_saved(self, saved, final=False):
        if saved.get("run_kind") != "training":
            raise ValueError("Engineering/foreign checkpoints cannot resume a registered fit")
        if not 0 <= saved["completed_epochs"] <= self.config["train"]["epochs"]:
            raise ValueError("Checkpoint epoch is outside registered recipe")
        if final and saved["completed_epochs"] != self.config["train"]["epochs"]:
            raise ValueError("Only fixed-final-epoch checkpoints can be evaluated")
        previous = saved["metadata"]
        for key in ("request_identity", "source_identity", "clip_source_hashes", "clip_checkpoint_sha256",
                    "taxonomy_sha256", "frozen_backbone_before", "trainable_parameters"):
            if previous.get(key) != self.metadata[key]:
                raise ValueError("Checkpoint provenance differs: " + key)

    def _recovery_audit(self, saved, origin):
        """Recheck frozen weights; absent historical measurements remain null."""
        final_backbone = backbone_hash(self.model)
        audit = {"all_backbone_weights_frozen": all(not p.requires_grad for p in self.model.clip_model.parameters()),
                 "frozen_backbone_before": self.initial_backbone, "frozen_backbone_after": final_backbone,
                 "backbone_unchanged": self.initial_backbone == final_backbone,
                 "first_step_gradient_norms": None, "steps": saved.get("steps"),
                 "median_step_seconds": None, "peak_memory_bytes": None,
                 "completed_epochs": saved["completed_epochs"], "training_seconds": saved.get("training_seconds"),
                 "state_sha256": model_state_hash(self.model), "recovery_without_optimizer_steps": True,
                 "recovery_origin": origin,
                 "missing_historical_measurements": ["first_step_gradient_norms", "median_step_seconds", "peak_memory_bytes"]}
        write_json(self.output / "audit.json", audit)
        if not audit["backbone_unchanged"] or not audit["all_backbone_weights_frozen"]:
            raise RuntimeError("Frozen CLIP audit failed during checkpoint finalization")
        return audit



    def _validate_recovery_checkpoint(self, checkpoint, saved, final=False):
        self._validate_saved(saved, final=final)

    def evaluate_checkpoint(self, checkpoint, recover_final=False, recovery_origin="final_checkpoint"):
        saved = load_checkpoint(checkpoint)
        self._validate_recovery_checkpoint(checkpoint, saved, final=True)
        if recover_final and absolute_path(checkpoint) != absolute_path(self.output / "final.pt"):
            raise ValueError("Recovery must use this output's final checkpoint")
        restore_model(self.model, saved)
        self._store_preparation(saved)
        if recover_final:
            self._recovery_audit(saved, recovery_origin)
        result = evaluate_protocol(self.model, self.bundle, self.targets, self.preprocess, self.config, self.output)
        result.update(completed_epochs=saved["completed_epochs"], model_name=self.config["method"]["name"],
                      state_sha256=saved["state_sha256"], training_seconds=saved.get("training_seconds"))
        if recover_final:
            result.update(recovery_without_optimizer_steps=True, recovery_origin=recovery_origin,
                          missing_historical_measurements=["first_step_gradient_norms", "median_step_seconds", "peak_memory_bytes"])
        write_json(self.output / "metadata.json", self.metadata)
        write_json(self.output / "results.json", result)
        if recover_final:
            finish_run(self.output, self.config["_request_identity"], saved["state_sha256"])
        write_json(self.output / "status.json", {"status": "evaluation_complete", "checkpoint": str(absolute_path(checkpoint))})
        return result

    def run(self, resume=None):
        train, device = self.config["train"], self.config["train"]["device"]
        if (self.output / "final.pt").exists() or (self.output / "complete.json").exists():
            raise FileExistsError("Final fit exists; never retrain or overwrite it. Replay evaluation into a fresh output if necessary")
        saved = None
        if resume is not None:
            validate_epoch_resume(self.output, resume)
            saved = load_checkpoint(resume)
            self._validate_saved(saved)
            if saved["completed_epochs"] == train["epochs"]:
                # The last epoch committed before final.pt/complete.json.
                # No loader, optimizer, scaler, calibration, or training step
                # is needed to audit and finalize this already trained state.
                restore_model(self.model, saved)
                self._recovery_audit(saved, "full_epoch_last")
                (self.output / "last.pt").replace(self.output / "final.pt")
                return self.evaluate_checkpoint(self.output / "final.pt", recover_final=True, recovery_origin="full_epoch_last")
        amp = device.startswith("cuda") and train["precision"] == "amp"
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(train["crop_min"], 1), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=train["horizontal_flip"]), transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        generator = torch.Generator().manual_seed(self.config["_seed"])
        loader = DataLoader(SceneDataset(self.bundle.train, transform), batch_size=train["batch_size"],
                            shuffle=True, num_workers=train["workers"], worker_init_fn=seed_worker,
                            generator=generator, pin_memory=amp, drop_last=False, persistent_workers=False)
        optimizer = torch.optim.Adam([p for _, p in self.parameters], lr=train["learning_rate"], weight_decay=train["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=train["grad_scaler_initial_scale"])
        records, completed, steps, prior_seconds = [], 0, 0, 0.0
        if resume is not None:
            restore_model(self.model, saved)
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            records, completed, steps = saved["history"], saved["completed_epochs"], saved["steps"]
            prior_seconds = saved["training_seconds"]
            restore_rng(saved["rng"], generator, device)
        else:
            if (self.output / "history.jsonl").exists() or (self.output / "last.pt").exists():
                raise FileExistsError("Partial run exists; explicit --resume is required")
            before = capture_rng(generator, device)
            self.model.prepare(self.bundle, self.preprocess, self.config["_seed"], self.output)
            if self.config["method"]["name"] == "SPCP":
                self.model.method_receipt["source_assets"] = {"taxonomy_sha256": self.metadata["taxonomy_sha256"],
                    "clip_source_hashes": self.metadata["clip_source_hashes"],
                    "clip_checkpoint_sha256": self.metadata["clip_checkpoint_sha256"],
                    "source_identity_sha256": json_hash(self.metadata["source_identity"])}
            # Calibration must never alter prompt initialization/crops/shuffle.
            restore_rng(before, generator, device)
        self._training_classnames()
        self._store_preparation(saved)
        write_json(self.output / "metadata.json", self.metadata)
        write_json(self.output / "status.json", {"status": "running", "completed_epochs": completed, "resumed": resume is not None})
        start = time.monotonic()
        step_times, gradient_report = [], {}
        if amp:
            torch.cuda.reset_peak_memory_stats()
        history_path = self.output / "history.jsonl"
        # A failed write after an epoch may leave an extra history row; the last
        # committed checkpoint is authoritative for epoch-boundary continuation.
        with history_path.open("w" if resume is not None else "x", encoding="utf-8", buffering=1) as history:
            for record in records:
                history.write(json.dumps(record, allow_nan=False) + "\n")
            for epoch in range(completed, train["epochs"]):
                self.model.train()
                if epoch < train["warmup_epochs"]:
                    lr = train["warmup_learning_rate"]
                else:
                    progress = (epoch - train["warmup_epochs"]) / max(1, train["epochs"] - train["warmup_epochs"])
                    lr = train["learning_rate"] * (1 + math.cos(math.pi * progress)) / 2
                for group in optimizer.param_groups:
                    group["lr"] = lr
                totals = dict(loss=0.0, ce=0.0, text_alignment=0.0, image_alignment=0.0, correct=0, count=0)
                mechanism_batches = []
                for images, labels, ids in loader:
                    if prior_seconds + time.monotonic() - start >= train["max_minutes_per_run"] * 60:
                        raise TimeoutError("Declared cumulative training limit reached; last completed epoch remains recoverable")
                    tick = time.monotonic()
                    images, labels = images.to(device, non_blocking=amp), labels.to(device, non_blocking=amp)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda" if amp else "cpu", enabled=amp):
                        terms = self.model.loss_terms(images, labels, ids, epoch)
                        loss, result = terms["loss"], terms["output"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    gradient_comparison = None
                    if len(mechanism_batches) < self.config.get("runtime", {}).get("diagnostic_gradient_batches", 0) and "diagnostic_aux_loss" in terms:
                        diag_start = time.monotonic()
                        tensors = [p for _, p in self.parameters]
                        ga = torch.autograd.grad(terms["diagnostic_base_loss"], tensors, retain_graph=True, allow_unused=True)
                        gb = torch.autograd.grad(terms["diagnostic_aux_loss"], tensors, retain_graph=True, allow_unused=True)
                        dot = sum((a.float()*b.float()).sum() for a, b in zip(ga, gb) if a is not None and b is not None)
                        na = sum(a.float().square().sum() for a in ga if a is not None).sqrt()
                        nb = sum(b.float().square().sum() for b in gb if b is not None).sqrt()
                        gradient_comparison = {"base_norm": float(na), "aux_norm": float(nb),
                            "cosine": float(dot/(na*nb)) if float(na*nb) > 0 else None,
                            "diagnostic_seconds": time.monotonic()-diag_start, "batch_index": len(mechanism_batches)}
                        components = terms.get("diagnostic_loss_components")
                        if components is not None:
                            measured = {}
                            for name, value in components.items():
                                grads = torch.autograd.grad(value, tensors, retain_graph=True, allow_unused=True) if value.requires_grad else [None]*len(tensors)
                                norm2 = sum(float(g.float().square().sum()) for g in grads if g is not None)
                                inner = sum(float((g.float()*ce.float()).sum()) for g,ce in zip(grads,ga) if g is not None and ce is not None)
                                norm = math.sqrt(norm2)
                                measured[name] = dict(value=float(value.detach()), norm=norm,
                                    cosine_to_ce=inner/(norm*float(na)) if norm*float(na)>0 else None)
                            gradient_comparison["weighted_components"] = measured
                            gradient_comparison["diagnostic_seconds"] = time.monotonic()-diag_start
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    missing = [name for name, p in self.parameters if p.grad is None]
                    nonfinite = [name for name, p in self.parameters if p.grad is not None and not torch.isfinite(p.grad).all()]
                    if missing or nonfinite:
                        raise FloatingPointError("Missing gradients: " + str(missing) + "; nonfinite gradients: " + str(nonfinite))
                    if not gradient_report:
                        gradient_report = {name: float(p.grad.norm().detach().cpu()) for name, p in self.parameters}
                    scaler.step(optimizer)
                    scaler.update()
                    if hasattr(self.model, "mechanism_diagnostics"):
                        diagnostic = self.model.mechanism_diagnostics()
                        diagnostic["batch_index"] = len(mechanism_batches)
                        diagnostic["labels"] = labels.detach().cpu().tolist()
                        diagnostic["gradient_comparison"] = gradient_comparison
                        diagnostic["trainable_gradient_norm"] = math.sqrt(sum(float(p.grad.float().square().sum()) for _, p in self.parameters))
                        mechanism_batches.append(diagnostic)
                    if amp:
                        torch.cuda.synchronize()
                    step_times.append(time.monotonic() - tick)
                    steps += 1
                    count = images.shape[0]
                    for key in ("loss", "ce", "text_alignment", "image_alignment"):
                        totals[key] += float(terms[key].detach().cpu()) * count
                    totals["correct"] += int((result["logits"].argmax(1) == labels).sum().detach().cpu())
                    totals["count"] += count
                if totals["count"] != len(self.bundle.train):
                    raise RuntimeError("Incomplete source support epoch")
                completed = epoch + 1
                elapsed = prior_seconds + time.monotonic() - start
                record = {"epoch": completed, "lr": lr, "seconds": elapsed,
                          **{key: totals[key] / totals["count"] for key in ("loss", "ce", "text_alignment", "image_alignment")},
                          "train_accuracy": 100 * totals["correct"] / totals["count"], "support_count": totals["count"]}
                if mechanism_batches:
                    record["mechanism_diagnostics"] = {"batches": mechanism_batches,
                        "sampling": "all source batches; auxiliary gradient comparison on registered first batch per epoch"}
                records.append(record)
                history.write(json.dumps(record, allow_nan=False) + "\n")
                with (self.output / "mechanism_diagnostics.jsonl").open("w", encoding="utf-8") as diagnostic_log:
                    for item in records:
                        diagnostic_log.write(json.dumps({"epoch": item["epoch"], **item.get("mechanism_diagnostics", {})}, allow_nan=False) + "\n")
                print(json.dumps({k: v for k, v in record.items() if k != "mechanism_diagnostics"}), flush=True)
                checkpoint = {"schema": SCHEMA, "run_kind": "training", "completed_epochs": completed,
                              "prompts": self.model.prompt_state_dict(), "method_state": self.model.method_state_dict(),
                              "state_sha256": model_state_hash(self.model), "optimizer": optimizer.state_dict(),
                              "scaler": scaler.state_dict(), "metadata": self.metadata,
                              "rng": capture_rng(generator, device), "history": records, "steps": steps, "training_seconds": elapsed}
                save_checkpoint(self.output / "last.pt", checkpoint)
        final_backbone = backbone_hash(self.model)
        audit = {"all_backbone_weights_frozen": all(not p.requires_grad for p in self.model.clip_model.parameters()),
                 "frozen_backbone_before": self.initial_backbone, "frozen_backbone_after": final_backbone,
                 "backbone_unchanged": self.initial_backbone == final_backbone, "first_step_gradient_norms": gradient_report,
                 "steps": steps, "median_step_seconds": float(np.median(step_times)), "completed_epochs": completed,
                 "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if amp else None,
                 "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if amp else None,
                 "training_seconds": prior_seconds + time.monotonic() - start, "state_sha256": model_state_hash(self.model)}
        write_json(self.output / "audit.json", audit)
        if not audit["backbone_unchanged"] or not audit["all_backbone_weights_frozen"]:
            raise RuntimeError("Frozen CLIP audit failed")
        if completed != train["epochs"]:
            raise RuntimeError("Partial fit is not a final result")
        (self.output / "last.pt").replace(self.output / "final.pt")
        result = evaluate_protocol(self.model, self.bundle, self.targets, self.preprocess, self.config, self.output)
        result.update(completed_epochs=completed, model_name=self.config["method"]["name"],
                      training_seconds=audit["training_seconds"], elapsed_seconds=prior_seconds + time.monotonic() - start,
                      state_sha256=model_state_hash(self.model))
        write_json(self.output / "results.json", result)
        finish_run(self.output, self.config["_request_identity"], result["state_sha256"])
        write_json(self.output / "status.json", {"status": "completed", "completed_epochs": completed, "seconds": result["elapsed_seconds"]})
        return result
