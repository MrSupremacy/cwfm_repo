from __future__ import annotations

import math


def parameter_groups(model, controller, condition, training):
    router_ids = {
        id(parameter)
        for wrapper in controller.wrappers.values()
        if wrapper.router is not None
        for parameter in wrapper.router.parameters()
    }
    named = list(model.named_parameters())
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    frozen = [(name, parameter) for name, parameter in named if not parameter.requires_grad]
    if frozen:
        raise ValueError(f"F1 requires every model Parameter to be trainable; found {len(frozen)} frozen")
    backbone = [(name, parameter) for name, parameter in trainable if id(parameter) not in router_ids]
    router = [(name, parameter) for name, parameter in trainable if id(parameter) in router_ids]
    observed_router_ids = {id(parameter) for _, parameter in router}
    if observed_router_ids != router_ids:
        raise ValueError("Router parameters are missing from the wrapped model")
    if not backbone:
        raise ValueError("F1 backbone parameter group is empty")
    if condition.has_router_parameters != bool(router):
        raise ValueError("Condition/router parameter ownership mismatch")

    groups = [{"name": "backbone", "params": [parameter for _, parameter in backbone],
               "lr": training["backbone_lr"]}]
    if router:
        groups.append({"name": "router", "params": [parameter for _, parameter in router],
                       "lr": training["router_lr"]})

    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    names_by_group = {
        "backbone": [name for name, _ in backbone],
        "router": [name for name, _ in router],
    }
    manifest = {
        "groups": [{"name": group["name"], "lr": group["lr"],
                    "parameter_names": names_by_group[group["name"]],
                    "parameter_count": sum(parameter.numel() for parameter in group["params"])}
                   for group in groups],
        "unique_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "aliases": {names[0]: names[1:] for names in aliases.values() if len(names) > 1},
    }
    if sum(item["parameter_count"] for item in manifest["groups"]) != manifest["unique_parameter_count"]:
        raise ValueError("Optimizer groups do not partition trainable parameters exactly once")
    return groups, manifest


def gradient_norm(parameters):
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            value = parameter.grad.detach().float().norm(2)
            squares.append(value.square())
    if not squares:
        return 0.0
    total = sum(squares).sqrt().item()
    if not math.isfinite(total):
        raise FloatingPointError("Non-finite gradient norm")
    return total
