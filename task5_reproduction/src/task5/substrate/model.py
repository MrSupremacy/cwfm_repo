from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from task5.common.config import root_for, variant_config
from task5.routing.routers import Router


def ffn_layers(model):
    if model.config.feed_forward_proj != "relu":
        raise ValueError("Only non-gated ReLU T5 FFNs are in scope")
    for stack in ("encoder", "decoder"):
        for i, block in enumerate(getattr(model, stack).block):
            parent = block.layer[-1]
            module = parent.DenseReluDense
            if not all(hasattr(module, attr) for attr in ("wi", "wo", "act", "dropout")):
                raise ValueError("Unsupported T5 FFN structure")
            yield f"{stack}_layer_{i:02d}", stack, parent, module


def load_dense(config, directory):
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(str(directory), local_files_only=True, torch_dtype=torch.float32)
    m, actual = config["model"], model.config
    if (actual.d_model, actual.d_ff, actual.num_layers, actual.num_decoder_layers) != (
            m["d_model"], m["d_ff"], m["encoder_layers"], m["decoder_layers"]):
        raise ValueError("Checkpoint architecture differs from the experiment substrate")
    if actual.feed_forward_proj != "relu":
        raise ValueError("Expected ReLU T5")
    model.requires_grad_(False)
    model.to(config["execution"]["device"])
    return model, tokenizer


class MaskedFFN(nn.Module):
    """Keep original neuron order, norm/residual and dropout; mask only the intermediate."""

    def __init__(self, original, labels, router, key, stack, k):
        super().__init__()
        self.wi, self.wo = original.wi, original.wo
        self.act, self.dropout = original.act, original.dropout
        self.router = router
        self.key, self.stack, self.k = key, stack, k
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.wi.weight.device)
        self.register_buffer("labels", labels, persistent=False)
        self.register_buffer("group_order", torch.argsort(labels, stable=True), persistent=False)
        self.experts = int(labels.max().item()) + 1
        self.cap = len(labels) // self.experts
        self.valid = self.token_ids = None
        self.observer = None
        self.capture_q = False
        self.force_all = router is None

    def forward(self, hidden):
        shape = hidden.shape[:-1]
        flat = hidden.reshape(-1, hidden.shape[-1])
        valid = torch.ones(len(flat), dtype=torch.bool, device=flat.device) if self.valid is None else self.valid.reshape(-1)
        if valid.numel() != len(flat):
            raise ValueError(f"{self.key}: valid mask is not aligned with current hidden input")
        activation = self.act(self.wi(hidden))
        a = activation.reshape(-1, activation.shape[-1])
        q = None
        if self.capture_q or (self.router is not None and self.router.arm == "R1"):
            q = a.detach().float().index_select(1, self.group_order).reshape(-1, self.experts, self.cap).sum(-1)
        selected = None
        if not self.force_all:
            ids = None if self.token_ids is None else self.token_ids.reshape(-1)
            selected, weights = self.router(flat, self.k, q=q, valid=valid, token_ids=ids)
            if weights is None:
                weights = torch.ones_like(selected, dtype=torch.float32)
            expert_weights = a.new_zeros((len(a), self.experts)).scatter(1, selected, weights)
            a = a * expert_weights.index_select(1, self.labels)
        if self.observer is not None:
            self.observer(self, shape, valid, selected, q, activation.reshape(-1, activation.shape[-1]).detach())
        return self.wo(self.dropout(a.reshape_as(activation)))


class Controller:
    def __init__(self, model, wrappers):
        self.wrappers = wrappers
        self.teacher_masks = None
        self.sample_ids = None
        self.handles = []
        for stack in ("encoder", "decoder"):
            self.handles.append(getattr(model, stack).register_forward_pre_hook(self._hook(stack), with_kwargs=True))

    def _hook(self, stack):
        def before(module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            mask = kwargs.get("attention_mask")
            if self.teacher_masks is not None:
                mask = self.teacher_masks[stack]
            elif ids is not None:
                # Generation has no capture or aux statistics. Cached decoder IDs may be length one.
                mask = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask[..., -ids.shape[-1]:].bool()
            for wrapper in self.wrappers.values():
                if wrapper.stack == stack:
                    wrapper.token_ids = ids
                    wrapper.valid = None if mask is None else mask.bool()
        return before

    def teacher_batch(self, batch):
        self.teacher_masks = {"encoder": batch["attention_mask"].bool(), "decoder": batch["labels"] != -100}

    def generation(self):
        self.teacher_masks = None
        self.observe(None)

    def observe(self, callback, with_q=False):
        for wrapper in self.wrappers.values():
            wrapper.observer, wrapper.capture_q = callback, with_q

    def aux_loss(self):
        terms = [w.router.aux for w in self.wrappers.values() if w.router is not None and w.router.aux is not None]
        value = sum(terms) if terms else next(iter(self.wrappers.values())).wi.weight.new_zeros(())
        for w in self.wrappers.values():
            if w.router is not None:
                w.router.aux = None
        return value

    def after_step(self):
        for wrapper in self.wrappers.values():
            if wrapper.router is not None:
                wrapper.router.after_step()

    def state(self):
        return {key: w.router.state_dict() for key, w in self.wrappers.items() if w.router is not None}

    def load_state(self, states):
        if set(states) != {key for key, w in self.wrappers.items() if w.router is not None}:
            raise ValueError("Checkpoint router layer set differs")
        for key, value in states.items():
            self.wrappers[key].router.load_state_dict(value, strict=True)

    def noise_states(self):
        return {key: w.router.noise_state() for key, w in self.wrappers.items() if w.router is not None}

    def load_noise_states(self, states):
        for key, value in states.items():
            self.wrappers[key].router.load_noise_state(value)


def attach(config, condition, model, labels, centroids, hash_table=None):
    wrappers = {}
    variant = {} if condition.arm == "dense" else variant_config(config, condition)
    if hash_table is not None:
        hash_table = hash_table.to(next(model.parameters()).device)
    for key, stack, parent, original in list(ffn_layers(model)):
        router = None if condition.arm == "dense" else Router(
            condition.arm, torch.as_tensor(centroids[key], device=original.wi.weight.device),
            config["routing"], variant, condition.seed, key, hash_table)
        wrapper = MaskedFFN(original, labels[key], router, key, stack, condition.k)
        wrapper.to(original.wi.weight.device)
        parent.DenseReluDense = wrapper
        wrappers[key] = wrapper
    return Controller(model, wrappers)
