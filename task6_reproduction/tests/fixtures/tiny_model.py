"""Small offline T5 fixture using the production wrappers and routers."""


def tiny_model(arm="R4o", variant="default", seed=0):
    import numpy as np
    import torch
    from transformers import T5Config, T5ForConditionalGeneration

    from task6.common.config import Condition, load_config
    from task6.routing.routers import raw_centroids
    from task6.substrate.model import attach, ffn_layers

    config = load_config()
    config["execution"]["device"] = "cpu"
    config["data"].update(loader_workers=0, pin_memory=False)
    config["model"].update(
        d_model=16, d_ff=32, num_experts=4, expert_size=8,
        encoder_layers=1, decoder_layers=1)
    torch.manual_seed(123)
    model = T5ForConditionalGeneration(T5Config(
        vocab_size=32, d_model=16, d_ff=32, d_kv=4, num_heads=4,
        num_layers=1, num_decoder_layers=1, feed_forward_proj="relu",
        dropout_rate=0.0, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    model.requires_grad_(True)
    labels = {key: np.arange(32) % 4 for key, _, _, _ in ffn_layers(model)}
    centroids = {key: raw_centroids(module.wi.weight, torch.from_numpy(labels[key]), 4).numpy()
                 for key, _, _, module in ffn_layers(model)}
    condition = Condition("sst2", "dense") if arm == "dense" else Condition(
        "sst2", arm, variant, 0 if arm == "dense-ft" else 2, seed)
    controller = attach(config, condition, model, labels, centroids)
    batch = {
        "input_ids": torch.tensor([[2, 3, 1], [4, 1, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "labels": torch.tensor([[5, 1], [6, -100]]),
    }
    return model, controller, batch, config, condition
