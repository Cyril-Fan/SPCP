"""Registered deterministic RGB operations; never touch global RNG streams."""
import hashlib
import torch
from torch.nn import functional as F

MEAN = (.48145466, .4578275, .40821073)
STD = (.26862954, .26130258, .27577711)


def gaussian_blur(rgb, sigma):
    axis = torch.arange(-4, 5, device=rgb.device, dtype=torch.float32)
    kernel = torch.exp(-axis.square() / (2*sigma*sigma))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :]).expand(3, 1, 9, 9)
    return F.conv2d(F.pad(rgb, (4, 4, 4, 4), mode="reflect"), kernel, groups=3)


def resolution(rgb, size):
    low = F.interpolate(rgb, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    return F.interpolate(low, size=rgb.shape[-2:], mode="bicubic", align_corners=False, antialias=True)


def bmc_choices(identifier, seed, epoch):
    if not isinstance(identifier, str) or not identifier or identifier.startswith(("/", "\\")) or ":" in identifier or ".." in identifier.replace("\\", "/").split("/"):
        raise ValueError("BMC needs a stable relative source image ID")
    key = "BMC-v1|%d|%d|%s" % (seed, epoch, identifier.replace("\\", "/"))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % 3, 1 if digest[4] & 1 else -1, 1 if digest[4] & 2 else -1


def bmc_view(images, ids, seed, epoch):
    if images.ndim != 4 or images.shape[1:] != (3, 224, 224) or ids is None or len(ids) != len(images):
        raise ValueError("BMC view requires normalized Bx3x224x224 crops and stable image IDs")
    with torch.autocast(device_type=images.device.type, enabled=False):
        x = images.float()
        mean, std = x.new_tensor(MEAN)[None, :, None, None], x.new_tensor(STD)[None, :, None, None]
        rgb = (x * std + mean).clamp(0, 1)
        output = []
        for i, identifier in enumerate(ids):
            family, sign_b, sign_c = bmc_choices(identifier, seed, epoch)
            original = rgb[i:i+1]
            if family == 0:
                changed = resolution(original, 112)
            elif family == 1:
                changed = original * (1 + .075*sign_b)
                center = changed.mean(dim=(1, 2, 3), keepdim=True)
                changed = (changed-center) * (1+.075*sign_c) + center
            else:
                changed = gaussian_blur(original, .4)
            output.append(changed.clamp(0, 1))
        return (torch.cat(output) - mean) / std
