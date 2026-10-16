"""Validated settings for the full SPCP objective."""
import math

DEFAULTS = dict(name="SPCP", enabled=True, eta=0.50, neighbors=3,
                geometry_epsilon=0.20, duplicate_epsilon=1e-6,
                control="difference", version=1)


def settings(method):
    if method.get("name") != "SPCP":
        raise ValueError("Unknown method")
    if set(method) - set(DEFAULTS):
        raise ValueError("Unknown SPCP settings: " + str(sorted(set(method) - set(DEFAULTS))))
    value = {**DEFAULTS, **method}
    if type(value["enabled"]) is not bool or type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("Invalid method enabled/version")
    for key in ("eta", "geometry_epsilon", "duplicate_epsilon"):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError("Finite numeric method setting required: " + key)
    if type(value["neighbors"]) is not int or value["neighbors"] != 3:
        raise ValueError("SPCP requires three candidate neighbors")
    if not 0 <= value["eta"] <= 1 or value["geometry_epsilon"] != .20 or value["duplicate_epsilon"] != 1e-6:
        raise ValueError("SPCP graph or mixture settings differ")
    if value["control"] != "difference":
        raise ValueError("Only the full coordinate-difference objective is released")
    return value
