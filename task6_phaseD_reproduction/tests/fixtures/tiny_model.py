def tiny_model(arm="G1", seed=0):
    import numpy as np
    import torch
    from transformers import T5Config, T5ForConditionalGeneration

    from task6_phased.common.config import Condition, load_config
    from task6_phased.routing.routers import raw_centroids
    from task6_phased.substrate.model import attach, ffn_layers

    config = load_config()
    config["execution"]["device"] = "cpu"
    config["model"].update(d_model=16, d_ff=32, encoder_layers=1, decoder_layers=1)
    torch.manual_seed(123)
    model = T5ForConditionalGeneration(T5Config(
        vocab_size=32, d_model=16, d_ff=32, d_kv=4, num_heads=4,
        num_layers=1, num_decoder_layers=1, feed_forward_proj="relu",
        dropout_rate=.1, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    ))
    model.requires_grad_(False)
    labels = {key: (np.arange(32) % 4).astype(np.int64) for key, _, _, _ in ffn_layers(model)}
    centroids = {
        key: raw_centroids(module.wi.weight, torch.from_numpy(labels[key]), 4).numpy()
        for key, _, _, module in ffn_layers(model)
    }
    condition = Condition("sst2", 4, arm, "default", 2, None if arm == "R2" else seed)
    controller = attach(config, condition, model, labels, centroids)
    batch = {
        "input_ids": torch.tensor([[2, 3, 1], [4, 1, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "labels": torch.tensor([[5, 1], [6, -100]]),
    }
    return model, controller, batch

