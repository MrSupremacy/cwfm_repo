from __future__ import annotations

import numpy as np


def layer_summary(values, worst=None):
    # Undefined layers propagate, never disappear through nanmean.
    if not values or any(v is None for v in values):
        return {"mean": None, "layer_std": None, "worst": None}
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite layer metric")
    return {"mean": float(x.mean()), "layer_std": float(x.std(ddof=0)),
            "worst": None if worst is None else float(x.max() if worst == "max" else x.min())}


def seed_summary(values, deterministic=False):
    if any(v is None for v in values):
        return {"mean": None, "std": None, "n": len(values), "deterministic": deterministic}
    if not values or (not deterministic and len(values) < 2):
        if len(values) == 1:
            return {"mean": float(values[0]), "std": None, "n": 1, "deterministic": False}
        raise ValueError("No seed observations")
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "std": None if deterministic else float(x.std(ddof=1)),
            "n": len(values), "deterministic": deterministic}
