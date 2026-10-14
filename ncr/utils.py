import copy
import json
import os
import random
import numpy as np
import torch
import yaml


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("_base_", None)
    if base:
        base_cfg = load_config(os.path.join(os.path.dirname(path), base))
        cfg = deep_update(base_cfg, cfg)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(val))
    return cfg


def deep_update(base, new):
    """Recursive dict merge. A sub-dict that carries a different 'type' than the base replaces it entirely
    (so switching e.g. the activation type does not inherit the keyword arguments of the previous type)."""
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            if "type" in v and v.get("type") != out[k].get("type"):
                out[k] = copy.deepcopy(v)
            else:
                out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg, key, val):
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = val


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(x, y, data_range=1.0):
    """Per-image PSNR (B,) of x against reference y (both clamped to [0, data_range] beforehand by caller)."""
    mse = ((x - y) ** 2).flatten(1).mean(dim=1)
    return 10 * torch.log10(data_range ** 2 / mse.clamp_min(1e-12))


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=float)


class CSVLogger:
    def __init__(self, path):
        self.path = path
        self._keys = None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path):
            with open(path) as f:
                head = f.readline().strip()
            self._keys = head.split(",") if head else None

    def log(self, row):
        if self._keys is None:
            self._keys = list(row.keys())
            with open(self.path, "w") as f:
                f.write(",".join(self._keys) + "\n")
        with open(self.path, "a") as f:
            f.write(",".join(str(row.get(k, "")) for k in self._keys) + "\n")
