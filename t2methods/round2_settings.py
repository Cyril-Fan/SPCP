"""Runtime-free, versioned method settings; each method is independent v3."""
import math

METHODS = ("PTR", "BMC", "CPTA")
DEFAULTS = {
    "PTR": dict(name="PTR", enabled=True, rho=0.5, scale_floor=1e-4,
                mode="bounded", fixed_h=0.25, version=1),
    "BMC": dict(name="BMC", enabled=True, weight=0.10, neighbors=3,
                reliability_scale=0.05, target_fraction=0.5, teacher_margin_cap=0.10,
                loss_scale=0.10, mask_epsilon=1e-6, augmentation_version="BMC-v1",
                control="margin_two_views", version=1),
    "CPTA": dict(name="CPTA", enabled=True, eta=0.25, neighbors=3,
                 geometry_epsilon=0.20, duplicate_epsilon=1e-6, control="difference", version=1),
}


def settings(method):
    name = method.get("name")
    if name not in DEFAULTS:
        raise ValueError("Unregistered round2 method")
    value = dict(DEFAULTS[name])
    if set(method) - set(value):
        raise ValueError("Unknown " + name + " settings: " + str(sorted(set(method) - set(value))))
    value.update(method)
    if type(value["enabled"]) is not bool or value["version"] != 1:
        raise ValueError("Invalid method enabled/version")
    for key, item in value.items():
        expected = DEFAULTS[name][key]
        if isinstance(expected, float) and (isinstance(item, bool) or not isinstance(item, (int, float))):
            raise ValueError("Numeric method setting required: " + key)
        if type(expected) is int and (type(item) is not int):
            raise ValueError("Integer method setting required: " + key)
        if isinstance(item, (int, float)) and not isinstance(item, bool) and not math.isfinite(item):
            raise ValueError("Nonfinite method setting: " + key)
    if name == "PTR":
        if not 0 < value["rho"] or value["scale_floor"] != 1e-4:
            raise ValueError("PTR requires positive rho and the registered scale floor")
        if value["mode"] not in ("bounded", "fixed_h", "dynamic_radius") or value["fixed_h"] != .25:
            raise ValueError("Unknown PTR control")
    if name == "BMC":
        if not 0 <= value["weight"] or type(value["neighbors"]) is not int or value["neighbors"] != 3:
            raise ValueError("Invalid BMC weight/neighbors")
        for key in ("reliability_scale", "target_fraction", "teacher_margin_cap", "loss_scale", "mask_epsilon", "augmentation_version"):
            if value[key] != DEFAULTS[name][key]:
                raise ValueError("BMC fixed protocol differs: " + key)
        if value["control"] not in ("margin_two_views", "original_only", "augmented_ce"):
            raise ValueError("Unknown BMC control")
    if name == "CPTA":
        if not 0 <= value["eta"] <= 1 or type(value["neighbors"]) is not int or value["neighbors"] != 3:
            raise ValueError("Invalid CPTA eta/neighbors")
        if value["geometry_epsilon"] != .20 or value["duplicate_epsilon"] != 1e-6:
            raise ValueError("CPTA fixed graph constants differ")
        if value["control"] not in ("difference", "gram", "unnormalized"):
            raise ValueError("Unknown CPTA control")
    return value
