import numpy as np


def intersections(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("Routing sets must have identical token population and k")
    return (a[:, :, None] == b[:, None, :]).any(-1).sum(-1)


def churn(a, b):
    shared = intersections(a, b)
    return 1 - shared.astype(np.float64) / a.shape[1], (shared != a.shape[1]).astype(np.float64)
