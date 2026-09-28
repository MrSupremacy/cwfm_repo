import numpy as np


def load_metrics(counts, tokens, k):
    counts = np.asarray(counts)
    if tokens <= 0 or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("Invalid load population/counts")
    if counts.sum() != tokens * k or np.any(counts > tokens):
        raise ValueError("Load assignments must sum to T*k with each expert selected at most once/token")
    p = counts.astype(np.float64) / (tokens * k)
    E = len(p)
    return {"cv": float(p.std(ddof=0) / p.mean()),
            "gini": float(np.abs(p[:, None] - p[None, :]).sum() / (2 * E * p.sum())),
            "maximum_share": float(p.max()), "valid_token_count": int(tokens)}
