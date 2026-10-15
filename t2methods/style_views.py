"""Deterministic source style proxies; no Python/NumPy/Torch RNG consumption.

Views act on the same already-cropped image, before CLIP normalization. Tensor
Gaussian blur is explicitly used instead of the design draft's PIL blur.
"""
from __future__ import annotations
import hashlib

import torch
from torchvision.transforms import functional as TF

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
FAMILIES = ("photometric", "saturation", "blur")

def _choice(seed, identifier, family, index):
    text = "%s\0%s\0%s\0%s" % (seed, identifier, family, index)
    return hashlib.sha256(text.encode("utf-8")).digest()[0] & 1

def style_views(images, ids, seed, families=FAMILIES, strength=1.0):
    if images.ndim != 4 or images.shape[1] != 3 or len(ids) != images.shape[0]:
        raise ValueError("style views require Bx3xHxW images and one stable ID per image")
    if not torch.isfinite(images).all():
        raise FloatingPointError("nonfinite input to source style proxies")
    if any(f not in FAMILIES for f in families) or len(set(families)) != len(families):
        raise ValueError("unsupported/repeated style family")
    if not isinstance(strength, (int, float)) or isinstance(strength, bool) or not 0 <= strength <= 2:
        raise ValueError("style strength must be finite and between zero and two")
    if strength == 0:
        return [images.detach().clone() for _ in families]
    mean = images.new_tensor(MEAN).reshape(1, 3, 1, 1)
    std = images.new_tensor(STD).reshape(1, 3, 1, 1)
    rgb = (images.detach().float() * std + mean).clamp(0, 1)
    views = []
    for family in families:
        transformed = []
        for index, identifier in enumerate(ids):
            image = rgb[index]
            if family == "photometric":
                brightness = 1 + strength * ((0.85, 1.15)[_choice(seed, identifier, family, 0)] - 1)
                contrast = 1 + strength * ((0.85, 1.15)[_choice(seed, identifier, family, 1)] - 1)
                image = TF.adjust_contrast(TF.adjust_brightness(image, brightness), contrast)
            elif family == "saturation":
                saturation = 1 + strength * ((0.75, 1.25)[_choice(seed, identifier, family, 0)] - 1)
                image = TF.adjust_saturation(image, saturation)
            else:
                if min(image.shape[-2:]) <= 4:
                    raise ValueError("blur calibration requires spatial dimensions >4")
                sigma = strength * (0.5, 0.8)[_choice(seed, identifier, family, 0)]
                image = TF.gaussian_blur(image, [9, 9], [sigma, sigma])
            transformed.append(image.clamp(0, 1))
        view = (torch.stack(transformed) - mean) / std
        views.append(view)
    return views
