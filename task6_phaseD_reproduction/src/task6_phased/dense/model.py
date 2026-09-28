from __future__ import annotations

from pathlib import Path

from task6_phased.common.config import repository_root


def resolve_asset(config, name):
    try:
        path = Path(config["assets"][name]).expanduser()
    except KeyError as exc:
        raise ValueError(f"Missing assets.{name} in local config") from exc
    return (path if path.is_absolute() else repository_root() / path).resolve()


def load_t5(config, source, *, trainable):
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    source = Path(source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(
        str(source), local_files_only=True, dtype=torch.float32
    )
    expected, actual = config["model"], model.config
    identity = (actual.d_model, actual.d_ff, actual.num_layers, actual.num_decoder_layers, actual.feed_forward_proj)
    wanted = (
        expected["d_model"], expected["d_ff"], expected["encoder_layers"],
        expected["decoder_layers"], expected["feed_forward_proj"],
    )
    if identity != wanted:
        raise ValueError(f"Unexpected T5 architecture: {identity}, expected {wanted}")
    model.requires_grad_(bool(trainable))
    model.to(config["execution"]["device"])
    return model, tokenizer


def sample_token_domain_loss(logits, labels, domain_ids):
    import torch
    import torch.nn.functional as functional
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logit/label dimensions are not aligned")
    token_loss = functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
        reduction="none", ignore_index=-100,
    ).reshape_as(labels)
    valid = labels != -100
    per_sample = token_loss.sum(1) / valid.sum(1).clamp_min(1)
    domains = []
    for domain in range(4):
        selected = domain_ids == domain
        if int(selected.sum()) == 0:
            raise ValueError("Every mixed batch must contain all four domains")
        domains.append(per_sample[selected].mean())
    return torch.stack(domains).mean(), torch.stack(domains), per_sample
