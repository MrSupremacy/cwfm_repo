from __future__ import annotations

import torch
from torch import nn

from task8_p01.routing.interventions import MODES, route


def ffn_layers(model):
    if model.config.feed_forward_proj != "relu":
        raise ValueError("Only the frozen non-gated ReLU T5 FFNs are supported")
    for stack in ("encoder", "decoder"):
        for index, block in enumerate(getattr(model, stack).block):
            parent = block.layer[-1]
            module = parent.DenseReluDense
            if not all(hasattr(module, name) for name in ("wi", "wo", "act", "dropout")):
                raise ValueError("Unsupported T5 FFN structure")
            yield f"{stack}_layer_{index:02d}", stack, parent, module


class InterventionFFN(nn.Module):
    """Frozen T5 FFN with a runtime-switchable P1 routing intervention."""

    def __init__(self, original, labels, c0, learned, key, stack, k, routing):
        super().__init__()
        self.wi, self.wo = original.wi, original.wo
        self.act, self.dropout = original.act, original.dropout
        self.key, self.stack, self.k = key, stack, int(k)
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.wi.weight.device)
        c0 = torch.as_tensor(c0, dtype=torch.float32, device=self.wi.weight.device)
        learned = torch.as_tensor(learned, dtype=torch.float32, device=self.wi.weight.device)
        if labels.shape != (self.wi.out_features,):
            raise ValueError(f"{key}: labels do not cover all FFN neurons")
        experts = int(labels.max()) + 1
        if c0.shape != learned.shape or c0.shape != (experts, self.wi.in_features):
            raise ValueError(f"{key}: summary shape mismatch")
        self.register_buffer("labels", labels, persistent=False)
        self.register_buffer("c0", c0, persistent=False)
        self.register_buffer("learned", learned, persistent=False)
        self.mode = "M11"
        self.routing = dict(routing)
        self.valid = None
        self.observer = None

    def forward(self, hidden):
        shape = hidden.shape[:-1]
        flat = hidden.reshape(-1, hidden.shape[-1])
        indices, weights = route(
            self.mode,
            flat,
            self.c0,
            self.learned,
            self.k,
            l2_epsilon=self.routing["l2_epsilon"],
            rms_epsilon=self.routing["rms_epsilon"],
            temperature=self.routing["temperature"],
        )
        activation = self.act(self.wi(hidden))
        flat_activation = activation.reshape(-1, activation.shape[-1])
        expert_weights = flat_activation.new_zeros((len(flat_activation), len(self.c0)))
        expert_weights.scatter_(1, indices, weights.to(flat_activation.dtype))
        masked = flat_activation * expert_weights.index_select(1, self.labels)
        output = self.wo(self.dropout(masked.reshape_as(activation)))
        if self.observer is not None:
            valid = (
                torch.ones(len(flat), dtype=torch.bool, device=flat.device)
                if self.valid is None
                else self.valid.reshape(-1)
            )
            self.observer(self, flat, valid, indices, weights, output)
        return output


class InterventionController:
    def __init__(self, model, wrappers):
        self.wrappers = wrappers
        self.teacher_masks = None
        self.handles = []
        for stack in ("encoder", "decoder"):
            self.handles.append(
                getattr(model, stack).register_forward_pre_hook(self._hook(stack), with_kwargs=True)
            )

    def _hook(self, stack):
        def before(module, args, kwargs):
            del module
            ids = kwargs.get("input_ids", args[0] if args else None)
            mask = kwargs.get("attention_mask")
            if self.teacher_masks is not None:
                mask = self.teacher_masks[stack]
            elif ids is not None:
                mask = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask[..., -ids.shape[-1] :].bool()
            for wrapper in self.wrappers.values():
                if wrapper.stack == stack:
                    wrapper.valid = None if mask is None else mask.bool()
        return before

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(mode)
        for wrapper in self.wrappers.values():
            wrapper.mode = mode

    def teacher_batch(self, batch):
        self.teacher_masks = {
            "encoder": batch["attention_mask"].bool(),
            "decoder": batch["labels"] != -100,
        }

    def generation(self):
        self.teacher_masks = None
        self.observe(None)

    def observe(self, callback):
        for wrapper in self.wrappers.values():
            wrapper.observer = callback


def attach(config, model, labels, c0, learned, k):
    wrappers = {}
    for key, stack, parent, original in list(ffn_layers(model)):
        if key not in labels or key not in c0 or key not in learned:
            raise ValueError(f"Missing routing tensor for {key}")
        wrapper = InterventionFFN(
            original,
            labels[key],
            c0[key],
            learned[key],
            key,
            stack,
            k,
            config["routing"],
        )
        wrapper.to(original.wi.weight.device)
        parent.DenseReluDense = wrapper
        wrappers[key] = wrapper
    if len(wrappers) != 12:
        raise ValueError(f"Expected 12 FFNs, found {len(wrappers)}")
    return InterventionController(model, wrappers)

