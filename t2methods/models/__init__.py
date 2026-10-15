"""Lazy registry for the full SPCP method; historical ID is checkpoint-stable."""
def build_model(method, clip_model, clip_module, config):
    if method != "CPTA":
        raise ValueError("This repository releases full SPCP only (historical ID: CPTA)")
    from .cpta import CPTAModel
    return CPTAModel(clip_model, clip_module, config)
