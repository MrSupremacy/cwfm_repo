from __future__ import annotations

from task6_phased.common.config import Condition, budgets_for
from task6_phased.substrate.assets import inspect_assets


def validate_substrate(config, experts):
    import torch
    from task6_phased.dense.model import load_t5
    from task6_phased.substrate.model import attach
    experts = int(experts)
    paths, labels, centroids, _ = inspect_assets(config, experts)
    dense, tokenizer = load_t5(config, paths["dense_best"], trainable=False)
    routed, _ = load_t5(config, paths["dense_best"], trainable=False)
    budgets = budgets_for(config, experts)
    controller = attach(config, Condition(experts, "R2", budgets[0], None), routed, labels, centroids)
    for wrapper in controller.wrappers.values():
        wrapper.force_all = True
    dense.eval()
    routed.eval()
    batch = tokenizer(
        ["sst2 sentence: a good movie", "qnli question: What is this? sentence: This is an example."],
        return_tensors="pt", padding=True,
    )
    labels_tensor = tokenizer(text_target=["positive", "entailment"], return_tensors="pt", padding=True)["input_ids"]
    labels_tensor[labels_tensor == tokenizer.pad_token_id] = -100
    batch = {key: value.to(config["execution"]["device"]) for key, value in batch.items()}
    labels_tensor = labels_tensor.to(config["execution"]["device"])
    with torch.no_grad():
        left = dense(**batch, labels=labels_tensor, use_cache=False).logits
        right = routed(**batch, labels=labels_tensor, use_cache=False).logits
    max_abs = float((left - right).abs().max())
    if max_abs >= 1.0e-5:
        raise ValueError(f"Force-all equivalence failed: max_abs={max_abs}")

    routed_trainable, _ = load_t5(config, paths["dense_best"], trainable=False)
    trained = attach(config, Condition(experts, "R4d", budgets[1], 0), routed_trainable, labels, centroids)
    trainable = {name for name, value in routed_trainable.named_parameters() if value.requires_grad}
    expected = {
        name for name, value in routed_trainable.named_parameters()
        if ".router." in name and value.requires_grad
    }
    if not trainable or trainable != expected:
        raise ValueError("F0 trainable scope is not exactly router parameters")
    return {
        "experts": experts, "force_all_max_abs": max_abs,
        "trainable_parameter_tensors": len(trainable),
    }
