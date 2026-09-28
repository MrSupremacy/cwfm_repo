import numpy as np


def load_metrics(counts, tokens, k):
    counts = np.asarray(counts)
    if tokens <= 0 or counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("Invalid load population/counts")
    if np.any(counts < 0) or counts.sum() != tokens * k or np.any(counts > tokens):
        raise ValueError("Assignments must sum to T*k with each expert used at most once per token")
    values = counts.astype(np.float64)
    return {"cv": float(values.std(ddof=0) / values.mean()), "valid_token_count": int(tokens)}
