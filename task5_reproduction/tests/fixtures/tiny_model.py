"""Synthetic model fixtures. Never download or impersonate the remote research assets."""


def tiny_model(arm="R4", variant="default", seed=0):
    import numpy as np
    import torch
    from transformers import T5Config, T5ForConditionalGeneration
    from task5.common.config import Condition, load_config
    from task5.substrate.model import attach, ffn_layers
    from task5.routing.routers import make_hash_table, raw_centroids
    config = load_config()
    if arm in ("R2-soft", "R4-hard"):
        config["variants"].append({"arm": arm, "name": "default", "trainable": arm == "R4-hard"})
    config["execution"]["device"] = "cpu"
    config["data"]["loader_workers"] = 0
    config["data"]["pin_memory"] = False
    config["model"].update(d_model=16, d_ff=32, num_experts=4, expert_size=8, encoder_layers=1, decoder_layers=1)
    torch.manual_seed(123)
    model = T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=16, d_ff=32, d_kv=4, num_heads=4,
                                              num_layers=1, num_decoder_layers=1, feed_forward_proj="relu",
                                              dropout_rate=.1, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    model.requires_grad_(False)
    labels = {key: np.arange(32) % 4 for key, _, _, _ in ffn_layers(model)}
    centroids = {key: raw_centroids(module.wi.weight, torch.from_numpy(labels[key]), 4).numpy()
                 for key, _, _, module in ffn_layers(model)}
    table = make_hash_table(32, 4, seed) if arm == "G0" else None
    controller = attach(config, Condition("sst2", arm, variant, 2, seed), model, labels, centroids, table)
    batch = {"input_ids": torch.tensor([[2, 3, 1], [4, 1, 0]]), "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
             "labels": torch.tensor([[5, 1], [6, -100]])}
    return model, controller, batch, config
