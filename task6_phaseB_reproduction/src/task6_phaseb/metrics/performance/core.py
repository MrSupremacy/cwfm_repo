from __future__ import annotations

import numpy as np


def performance(records, dense_accuracy=None, expected_ids=None):
    ids = np.asarray(records["sample_id"], dtype=np.int64)
    right = np.asarray(records["prediction_right"], dtype=bool)
    valid = np.asarray(records["prediction_valid"], dtype=bool)
    if len(ids) == 0 or len(set(ids.tolist())) != len(ids) or len(ids) != len(right) or len(ids) != len(valid):
        raise ValueError("Incomplete/duplicate A records")
    if np.any(right & ~valid):
        raise ValueError("An invalid prediction cannot be correct")
    if expected_ids is not None and not np.array_equal(ids, expected_ids):
        raise ValueError("A sample IDs differ from the fixed validation population")
    accuracy = float(right.mean())
    return {"count": len(ids), "correct": int(right.sum()), "accuracy": accuracy,
            "invalid_rate": float((~valid).mean()),
            "relative_performance": None if dense_accuracy is None or dense_accuracy <= 0 else 100 * accuracy / dense_accuracy}


def choose_best(candidates):
    if not candidates or len({item["count"] for item in candidates}) != 1:
        raise ValueError("Best candidates must cover the same complete validation population")
    return min(candidates, key=lambda item: (-item["correct"], item["state"]["step"]))
