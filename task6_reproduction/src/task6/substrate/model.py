from __future__ import annotations

import torch
from torch import nn

from task6.common.config import variant_config
from task6.routing.routers import Router


def ffn_layers(model):
    if model.config.feed_forward_proj != "relu":
        raise ValueError("Only non-gated ReLU T5 FFNs are in scope")
    for stack in ("encoder", "decoder"):
        for index, block in enumerate(getattr(model, stack).block):
            parent = block.layer[-1]
            module = parent.DenseReluDense
            if not all(hasattr(module, attr) for attr in ("wi", "wo", "act", "dropout")):
                raise ValueError("Unsupported T5 FFN structure")
            yield f"{stack}_layer_{index:02d}", stack, parent, module


def load_dense(config, directory):
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(
        str(directory), local_files_only=True, dtype=torch.float32)
    expected, actual = config["model"], model.config
    if (actual.d_model, actual.d_ff, actual.num_layers, actual.num_decoder_layers) != (
            expected["d_model"], expected["d_ff"], expected["encoder_layers"], expected["decoder_layers"]):
        raise ValueError("Checkpoint architecture differs from the Phase A substrate")
    if actual.feed_forward_proj != "relu":
        raise ValueError("Expected non-gated ReLU T5")
    model.requires_grad_(True)
    model.to(config["execution"]["device"])
    return model, tokenizer


class MaskedFFN(nn.Module):
    """Preserve T5 neuron order and apply the selected expert mask before dropout/W2."""

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
        self.capacity = len(labels) // self.experts
        self.valid = None
        self.observer = None
        self.capture_q = False
        self.force_all = router is None

    def forward(self, hidden):
        shape = hidden.shape[:-1]
        flat = hidden.reshape(-1, hidden.shape[-1])
        valid = torch.ones(len(flat), dtype=torch.bool, device=flat.device) if self.valid is None else self.valid.reshape(-1)
        if valid.numel() != len(flat):
            raise ValueError(f"{self.key}: valid mask is not aligned with hidden input")
        activation = self.act(self.wi(hidden))
        flat_activation = activation.reshape(-1, activation.shape[-1])
        q = None
        if self.capture_q:
            grouped = flat_activation.detach().float().index_select(1, self.group_order)
            q = grouped.reshape(-1, self.experts, self.capacity).sum(-1)
        selected = None
        if not self.force_all:
            selected, weights = self.router(flat, self.k, valid=valid)
            if weights is None:
                weights = torch.ones_like(selected, dtype=torch.float32)
            expert_weights = flat_activation.new_zeros((len(flat_activation), self.experts)).scatter(1, selected, weights)
            flat_activation = flat_activation * expert_weights.index_select(1, self.labels)
        if self.observer is not None:
            self.observer(self, shape, valid, selected, q)
        return self.wo(self.dropout(flat_activation.reshape_as(activation)))


class Controller:
    def __init__(self, model, wrappers):
        self.model, self.wrappers = model, wrappers
        self.teacher_masks = None
        self.handles = []
        for stack in ("encoder", "decoder"):
            self.handles.append(getattr(model, stack).register_forward_pre_hook(
                self._hook(stack), with_kwargs=True))

    def _hook(self, stack):
        def before(module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            mask = kwargs.get("attention_mask")
            if self.teacher_masks is not None:
                mask = self.teacher_masks[stack]
            elif ids is not None:
                mask = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask[..., -ids.shape[-1]:].bool()
            for wrapper in self.wrappers.values():
                if wrapper.stack == stack:
                    wrapper.valid = None if mask is None else mask.bool()
        return before

    def teacher_batch(self, batch):
        self.teacher_masks = {
            "encoder": batch["attention_mask"].bool(),
            "decoder": batch["labels"] != -100,
        }

    def generation(self):
        self.teacher_masks = None
        self.observe(None)

    def observe(self, callback, with_q=False):
        for wrapper in self.wrappers.values():
            wrapper.observer, wrapper.capture_q = callback, with_q

    def aux_losses(self):
        terms = {key: wrapper.router.aux for key, wrapper in self.wrappers.items()
                 if wrapper.router is not None and wrapper.router.aux is not None}
        for wrapper in self.wrappers.values():
            if wrapper.router is not None:
                wrapper.router.aux = None
        return terms

    def aux_loss(self):
        terms = self.aux_losses()
        return sum(terms.values()) if terms else next(iter(self.wrappers.values())).wi.weight.new_zeros(())

    def after_step(self):
        for wrapper in self.wrappers.values():
            if wrapper.router is not None:
                wrapper.router.after_step()

    def clear_pending(self):
        for wrapper in self.wrappers.values():
            if wrapper.router is not None:
                wrapper.router.clear_pending()


def attach(config, condition, model, labels, centroids):
    wrappers = {}
    variant = variant_config(config, condition)
    for key, stack, parent, original in list(ffn_layers(model)):
        router = None if not condition.is_routed else Router(
            condition.arm,
            torch.as_tensor(centroids[key], device=original.wi.weight.device),
            config["routing"], variant, condition.seed, key,
        )
        wrapper = MaskedFFN(original, labels[key], router, key, stack, condition.k)
        wrapper.to(original.wi.weight.device)
        parent.DenseReluDense = wrapper
        wrappers[key] = wrapper
    return Controller(model, wrappers)
