"""SPCP model construction."""
from .spcp import SPCPModel


def build_model(method, clip_model, clip_module, config):
    if method != "SPCP":
        raise ValueError("Only the full SPCP method is registered")
    return SPCPModel(clip_model, clip_module, config)
