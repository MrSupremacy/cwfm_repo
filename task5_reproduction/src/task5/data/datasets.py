from __future__ import annotations

import hashlib
import importlib.metadata
import json
import numpy as np

from task5.common.config import digest, root_for


def load_raw(config, task, paths):
    from datasets import load_dataset, load_from_disk
    spec = config["tasks"][task]
    if paths["source"] == "local_files":
        cache = root_for(config) / "tmp/preprocessing/raw"
        cache.mkdir(parents=True, exist_ok=True)
        raw = load_dataset("parquet", data_files={"train": str(paths["train"]), "validation": str(paths["validation"])},
                           cache_dir=str(cache))
        train, validation = raw["train"], raw["validation"]
    elif paths["source"] == "load_from_disk":
        raw = load_from_disk(str(paths["dataset"]))
        train, validation = raw["train"], raw[spec["validation_split"]]
    else:
        raise ValueError("Only local_files or load_from_disk is supported")
    for name, ds in (("train", train), ("validation", validation)):
        if len(ds) != spec[f"{name}_count"]:
            raise ValueError(f"{task}.{name} has {len(ds)} samples; expected {spec[f'{name}_count']}. Do not silently truncate.")
        lab = np.asarray(ds["label"])
        if not np.issubdtype(lab.dtype, np.integer) or np.any(lab < 0) or np.any(lab >= len(spec["labels"])):
            raise ValueError("Invalid dataset label; no samples may be silently removed")
    limits = config["suite"]
    if "train_limit" in limits:
        train = train.select(range(limits["train_limit"]))
        validation = validation.select(range(limits["validation_limit"]))
    return train, validation


def cache_declaration(config, task, dataset, tokenizer, input_identity, population):
    """Content identity, not machine path, decides whether tokenized data may be reused."""
    if not input_identity or population not in ("train", "validation"):
        raise ValueError("Tokenization requires verified input identity and population")
    versions = {}
    for name in ("datasets", "transformers", "tokenizers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # These are transient call settings; the actual protocol is recorded below.
    backend.pop("truncation", None)
    backend.pop("padding", None)
    return {"format": "task5_tokenization_v1", "task": task, "population": population,
            "inputs": input_identity, "dataset_fingerprint": dataset._fingerprint,
            "sample_count": len(dataset), "template": config["tasks"][task]["template"],
            "labels": config["tasks"][task]["labels"],
            "source_length": config["data"]["max_source_length"],
            "target_length": config["data"]["max_target_length"],
            "workers": config["data"]["preprocess_workers"],
            "tokenizer_class": type(tokenizer).__name__,
            "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side,
            "tokenizer_backend": digest(backend),
            "special_tokens": tokenizer.special_tokens_map, "versions": versions}


def tokenize(config, task, dataset, tokenizer, input_identity, population):
    spec, data = config["tasks"][task], config["data"]
    from task5.common.io import read_json, write_json
    declaration = cache_declaration(config, task, dataset, tokenizer, input_identity, population)
    key = digest(declaration)
    cache = root_for(config) / "tmp/preprocessing/tokenized" / task / population / key
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "declaration.json"
    if manifest.exists() and read_json(manifest) != declaration:
        raise ValueError("Tokenization cache identity mismatch")
    write_json(manifest, declaration)

    def encode(batch, indices):
        inputs = []
        for i in range(len(indices)):
            fields = {key: value[i] for key, value in batch.items()}
            for key in (("sentence",) if task == "sst2" else ("premise", "hypothesis")):
                if not isinstance(fields[key], str):
                    raise ValueError(f"Non-text field {key} in sample {indices[i]}")
            inputs.append(spec["template"].format(**fields))
        out = tokenizer(inputs, truncation=True, max_length=data["max_source_length"], padding=False)
        targets = tokenizer(text_target=[spec["labels"][int(v)] for v in batch["label"]],
                            truncation=True, max_length=data["max_target_length"], padding=False)
        out["labels"] = targets["input_ids"]
        out["sample_id"] = indices
        out["class_id"] = [int(v) for v in batch["label"]]
        return out

    # datasets appends a worker suffix to this internal fingerprint (64-character
    # limit). The cache path and declaration retain the complete SHA256 identity.
    return dataset.map(encode, batched=True, with_indices=True, remove_columns=dataset.column_names,
                       num_proc=data["preprocess_workers"], load_from_cache_file=True,
                       cache_file_name=str(cache / "tokens.arrow"), new_fingerprint=key[:32])


def probe_members(labels, count, seed=0):
    labels = np.asarray(labels)
    if not 0 < count <= len(labels):
        raise ValueError("Invalid probe size")
    if count == len(labels):
        return np.arange(len(labels), dtype=np.int64)
    classes, frequencies = np.unique(labels, return_counts=True)
    allocation = count * frequencies / len(labels)
    quotas = np.floor(allocation).astype(int)
    order = sorted(range(len(classes)), key=lambda i: (-(allocation[i] - quotas[i]), int(classes[i])))
    for i in order[:count - int(quotas.sum())]:
        quotas[i] += 1
    rng = np.random.Generator(np.random.PCG64(seed))
    chosen = [rng.choice(np.flatnonzero(labels == label), size=n, replace=False) for label, n in zip(classes, quotas)]
    return np.sort(np.concatenate(chosen)).astype(np.int64)


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, records):
        import torch
        inputs = [{k: row[k] for k in ("input_ids", "attention_mask")} for row in records]
        batch = self.tokenizer.pad(inputs, return_tensors="pt", padding=True)
        width = max(len(row["labels"]) for row in records)
        target = torch.full((len(records), width), -100, dtype=torch.long)
        for i, row in enumerate(records):
            values = torch.tensor(row["labels"], dtype=torch.long)
            offset = width - len(values) if self.tokenizer.padding_side == "left" else 0
            target[i, offset:offset + len(values)] = values
        batch["labels"] = target
        batch["sample_id"] = torch.tensor([row["sample_id"] for row in records], dtype=torch.long)
        batch["class_id"] = torch.tensor([row["class_id"] for row in records], dtype=torch.long)
        return batch


def make_loader(config, dataset, tokenizer, batch_size, shuffle_generator=None):
    import torch
    from torch.utils.data import DataLoader, RandomSampler
    workers = config["data"]["loader_workers"]
    # Worker base-seed creation must not consume the shuffle stream on resume.
    worker_rng = torch.Generator().manual_seed(1701)
    sampler = RandomSampler(dataset, generator=shuffle_generator) if shuffle_generator is not None else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=False,
                      collate_fn=Collator(tokenizer), num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=config["data"]["pin_memory"], generator=worker_rng)


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items() if k not in ("sample_id", "class_id")}


def expected_probe_keys(dataset, batch_size, padding_side):
    result = {}
    for stack, field in (("encoder", "input_ids"), ("decoder", "labels")):
        h, count = hashlib.sha256(), 0
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            width = max(len(row[field]) for row in rows)
            for row in rows:
                length = len(row[field])
                offset = width - length if padding_side == "left" else 0
                keys = np.column_stack((np.full(length, row["sample_id"], dtype=np.int64), np.arange(offset, offset + length)))
                h.update(keys.astype("<i8").tobytes())
                count += length
        result[stack] = {"count": count, "sha256": h.hexdigest()}
    return result
