from __future__ import annotations

import hashlib
import os
import random
import numpy as np


def stream_seed(seed, *parts):
    text = ":".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**63 - 1)


def configure_torch(config):
    # Must execute before a CUDA context/BLAS handle is initialized.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("DDP is not part of this protocol")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(config["execution"]["deterministic"])


def seed_dropout(seed):
    import torch
    derived = stream_seed(seed, "dropout")
    random.seed(derived)
    np.random.seed(derived % 2**32)
    torch.manual_seed(derived)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derived)


def rng_state():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
