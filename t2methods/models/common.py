from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Real

import torch
from torch.nn import functional as F

from ..base_model import T2NetModel

def finite_float(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    value = float(value)
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(name + " is out of range")
    return value

def flag(value, name):
    if not isinstance(value, bool):
        raise ValueError(name + " must be boolean")
    return value

def payload_hash(payload):
    """Hash tensors and settings independently of runtime vocabulary/device."""
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    visit(payload)
    return digest.hexdigest()

class MethodModel(T2NetModel):
    method_name = "UNSET"

    def __init__(self, clip_model, clip_module, config):
        requested = copy.deepcopy(config.get("method", {}))
        if requested.get("name", self.method_name) != self.method_name:
            raise ValueError("Model/config method mismatch")
        # This constructor is byte-preserving original-v3 initialization.
        super().__init__(clip_model, clip_module, config)
        self.method_config = requested
        self.method_config["name"] = self.method_name
        self.enabled = flag(requested.get("enabled", True), "method.enabled")
        self.method_config["enabled"] = self.enabled
        self.method_receipt = {}
        self.inference_off = False
        self.set_loss_settings(config)

    def prepare(self, bundle, eval_preprocess, seed, output_dir):
        self.method_receipt = {"method": self.method_name, "calibration": "not required",
                               "extra_images": 0, "extra_labels": 0}

    def teacher_target(self, images, output, ids=None, epoch=0):
        cached = output.get("reference_image_features")
        return self.reference_image(images) if cached is None else cached.detach()

    def loss_terms(self, images, labels, ids=None, epoch=0):
        output = self(images)
        raw = output.get("raw_image_features", output["image_features"])
        teacher = self.teacher_target(images, output, ids, epoch)
        metric = getattr(self, "alignment_metric", "cosine")
        if metric != "cosine":
            raise ValueError("The fixed-v3 default and method comparisons require cosine alignment")
        ce = F.cross_entropy(output["logits"].float(), labels)
        text = (1 - F.cosine_similarity(output["text_features"].float(), self.reference_text().float(), dim=-1)).mean()
        image = (1 - F.cosine_similarity(raw.float(), teacher.detach().float(), dim=-1)).mean()
        loss = ce + self.alignment_weight * (self.text_alignment_weight * text + self.image_alignment_weight * image)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite method loss")
        return {"loss": loss, "ce": ce, "text_alignment": text, "image_alignment": image, "output": output}

    def set_loss_settings(self, config):
        settings = config.get("model", config)
        self.alignment_metric = settings.get("alignment_metric", "cosine")
        self.alignment_weight = finite_float(settings.get("alignment_weight", 1.0), "alignment_weight", 0)
        self.text_alignment_weight = finite_float(settings.get("text_alignment_weight", 1.0), "text_alignment_weight", 0)
        self.image_alignment_weight = finite_float(settings.get("image_alignment_weight", 1.0), "image_alignment_weight", 0)

    def _method_buffers(self):
        return {}

    def method_state_dict(self):
        return {"method": self.method_name, "version": 1, "config": copy.deepcopy(self.method_config),
                "buffers": {k: v.detach().cpu().clone() for k, v in self._method_buffers().items()},
                "receipt": copy.deepcopy(self.method_receipt)}

    def validate_method_state(self, state):
        if not isinstance(state, dict) or state.get("method") != self.method_name or state.get("version") != 1:
            raise ValueError("checkpoint method/version mismatch")
        if state.get("config") != self.method_config:
            raise ValueError("checkpoint method configuration mismatch")
        if not isinstance(state.get("buffers"), dict) or set(state["buffers"]) != set(self._method_buffers()):
            raise ValueError("checkpoint method buffer inventory mismatch")
        if not isinstance(state.get("receipt"), dict):
            raise ValueError("checkpoint method receipt missing")
        for name, current in self._method_buffers().items():
            value = state["buffers"][name]
            if not isinstance(value, torch.Tensor) or value.shape != current.shape or value.dtype != current.dtype:
                raise ValueError("method buffer shape/dtype mismatch: " + name)
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("nonfinite method buffer: " + name)

    @torch.no_grad()
    def load_method_state_dict(self, state):
        self.validate_method_state(state)
        for name, current in self._method_buffers().items():
            current.copy_(state["buffers"][name].to(current.device))
        self.method_receipt = copy.deepcopy(state["receipt"])

    def immutable_state_hash(self):
        return payload_hash({"prompts": self.prompt_state_dict(), "method": self.method_state_dict()})
