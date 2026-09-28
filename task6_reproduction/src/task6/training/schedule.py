def scheduler_scale(step, total, warmup):
    """Factor at the current successful-update count; step zero uses LR zero."""
    if total < 1 or not 0 <= warmup <= total or step < 0:
        raise ValueError("Invalid scheduler bounds")
    return step / max(1, warmup) if step < warmup else max(0.0, (total - step) / max(1, total - warmup))
