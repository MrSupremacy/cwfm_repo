from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
import re
import numpy as np

from task6_phased.common.config import TASKS, digest, repository_root, tmp_root


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repository_root() / path).resolve()


def dataset_paths(config, task):
    try:
        raw = config["assets"]["datasets"][task]
    except KeyError as exc:
        raise ValueError(f"Missing assets.datasets.{task}") from exc
    paths = {key: resolve_path(value) for key, value in raw.items()}
    for population in ("train", "validation"):
        if population not in paths or not paths[population].is_file():
            raise FileNotFoundError(f"Missing {task} {population} parquet: {paths.get(population)}")
        expected = config["data"]["parquet_sha256"][task][population]
        actual = parquet_sha256(paths[population])
        if actual != expected:
            raise ValueError(f"{task}.{population}: parquet SHA256 mismatch ({actual} != {expected})")
    return paths


def load_raw(config, task):
    from datasets import load_dataset
    paths = dataset_paths(config, task)
    cache = tmp_root(config) / "datasets_cache" / task
    cache.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(
        "parquet",
        data_files={name: str(path) for name, path in paths.items()},
        cache_dir=str(cache),
    )
    spec = config["tasks"][task]
    train, validation = raw["train"], raw["validation"]
    formal = config["suite"]["name"] != "smoke"
    if formal:
        for name, dataset in (("train", train), ("validation", validation)):
            expected = int(spec[f"{name}_count"])
            if len(dataset) != expected:
                raise ValueError(f"{task}.{name}: expected {expected}, found {len(dataset)}")
    else:
        train = train.select(range(min(len(train), int(config["suite"]["train_limit"]))))
        validation = validation.select(range(min(len(validation), int(config["suite"]["validation_limit"]))))
    for name, dataset in (("train", train), ("validation", validation)):
        labels = np.asarray(dataset["label"])
        if len(labels) == 0 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{task}.{name}: invalid labels")
        if np.any(labels < 0) or np.any(labels >= len(spec["labels"])):
            raise ValueError(f"{task}.{name}: label outside verbalizer")
    return train, validation


def tokenize(config, task, dataset, tokenizer, population, input_identity):
    spec, data = config["tasks"][task], config["data"]
    declaration = {
        "format": "task6_phased_tokenization_v1",
        "task": task,
        "population": population,
        "dataset_fingerprint": dataset._fingerprint,
        "sample_count": len(dataset),
        "input_identity": input_identity,
        "template": spec["template"],
        "labels": spec["labels"],
        "max_source_length": data["max_source_length"],
        "max_target_length": data["max_target_length"],
        "tokenizer": tokenizer.name_or_path,
        "special_tokens": tokenizer.special_tokens_map,
    }
    key = digest(declaration)
    cache = tmp_root(config) / "tokenized" / task / population / key
    cache.mkdir(parents=True, exist_ok=True)

    def encode(batch, indices):
        inputs = []
        spans_by_input = []
        for row in range(len(indices)):
            fields = {field: batch[field][row] for field in spec["fields"]}
            if any(not isinstance(value, str) for value in fields.values()):
                raise ValueError(f"{task}: non-text input at source index {indices[row]}")
            pieces, spans, cursor = [], [], 0
            for match in re.finditer(r"\{([A-Za-z0-9_]+)\}", spec["template"]):
                literal = spec["template"][cursor:match.start()]
                pieces.append(literal)
                value = fields[match.group(1)]
                start = sum(len(piece) for piece in pieces)
                pieces.append(value)
                spans.append((start, start + len(value)))
                cursor = match.end()
            pieces.append(spec["template"][cursor:])
            inputs.append("".join(pieces))
            spans_by_input.append(spans)
        encoded = tokenizer(
            inputs, max_length=data["max_source_length"], truncation=True, padding=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        encoded["content_mask"] = [
            [
                bool(start < end and any(start >= left and end <= right for left, right in spans))
                for start, end in sample_offsets
            ]
            for sample_offsets, spans in zip(offsets, spans_by_input)
        ]
        targets = tokenizer(
            text_target=[spec["labels"][int(label)] for label in batch["label"]],
            max_length=data["max_target_length"], truncation=True, padding=False,
            add_special_tokens=True,
        )
        encoded["labels"] = targets["input_ids"]
        encoded["source_index"] = indices
        encoded["class_id"] = [int(value) for value in batch["label"]]
        encoded["domain_id"] = [TASKS.index(task)] * len(indices)
        return encoded

    return dataset.map(
        encode, batched=True, with_indices=True, remove_columns=dataset.column_names,
        num_proc=int(data["preprocess_workers"]), load_from_cache_file=True,
        cache_file_name=str(cache / "tokens.arrow"), new_fingerprint=key[:32],
    )


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, records):
        import torch
        inputs = [{key: row[key] for key in ("input_ids", "attention_mask")} for row in records]
        batch = self.tokenizer.pad(inputs, return_tensors="pt", padding=True)
        width = max(len(row["labels"]) for row in records)
        labels = torch.full((len(records), width), -100, dtype=torch.long)
        for index, row in enumerate(records):
            value = torch.tensor(row["labels"], dtype=torch.long)
            offset = width - len(value) if self.tokenizer.padding_side == "left" else 0
            labels[index, offset:offset + len(value)] = value
        batch["labels"] = labels
        source_width = batch["input_ids"].shape[1]
        content = torch.zeros((len(records), source_width), dtype=torch.bool)
        for index, row in enumerate(records):
            value = torch.tensor(row["content_mask"], dtype=torch.bool)
            offset = source_width - len(value) if self.tokenizer.padding_side == "left" else 0
            content[index, offset:offset + len(value)] = value
        batch["content_mask"] = content
        for key in ("source_index", "class_id", "domain_id"):
            batch[key] = torch.tensor([int(row[key]) for row in records], dtype=torch.long)
        return batch


def move_model_batch(batch, device):
    metadata = {"source_index", "class_id", "domain_id", "content_mask"}
    return {key: value.to(device) for key, value in batch.items() if key not in metadata}


def make_loader(config, dataset, tokenizer, batch_size):
    from torch.utils.data import DataLoader
    workers = int(config["data"]["loader_workers"])
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        collate_fn=Collator(tokenizer), num_workers=workers,
        persistent_workers=workers > 0, pin_memory=bool(config["data"]["pin_memory"]),
    )


class CyclicDomainSampler:
    """Four independent shuffled cyclic streams with exact resumable state."""

    def __init__(self, lengths, seed):
        import torch
        self.lengths = {task: int(lengths[task]) for task in TASKS}
        self.generators = {}
        self.permutations = {}
        self.cursors = {}
        self.cycles = {}
        for domain, task in enumerate(TASKS):
            generator = torch.Generator().manual_seed(int(seed) * 1009 + domain)
            self.generators[task] = generator
            self.permutations[task] = torch.randperm(self.lengths[task], generator=generator).tolist()
            self.cursors[task] = 0
            self.cycles[task] = 0

    def _refill(self, task):
        import torch
        self.permutations[task] = torch.randperm(
            self.lengths[task], generator=self.generators[task]
        ).tolist()
        self.cursors[task] = 0
        self.cycles[task] += 1

    def take(self, task, count):
        result = []
        while len(result) < count:
            if self.cursors[task] == self.lengths[task]:
                self._refill(task)
            available = min(count - len(result), self.lengths[task] - self.cursors[task])
            begin = self.cursors[task]
            result.extend(self.permutations[task][begin:begin + available])
            self.cursors[task] += available
        return result

    def state_dict(self):
        return {
            "lengths": dict(self.lengths),
            "permutations": {task: list(value) for task, value in self.permutations.items()},
            "cursors": dict(self.cursors), "cycles": dict(self.cycles),
            "generator_states": {task: gen.get_state().clone() for task, gen in self.generators.items()},
        }

    def load_state_dict(self, state):
        if state["lengths"] != self.lengths:
            raise ValueError("Sampler dataset lengths changed")
        self.permutations = {task: list(state["permutations"][task]) for task in TASKS}
        self.cursors = {task: int(state["cursors"][task]) for task in TASKS}
        self.cycles = {task: int(state["cycles"][task]) for task in TASKS}
        for task in TASKS:
            self.generators[task].set_state(state["generator_states"][task])


class MixedBatchStream:
    def __init__(self, datasets, tokenizer, domain_batch_size, seed):
        self.datasets = datasets
        self.collator = Collator(tokenizer)
        self.domain_batch_size = int(domain_batch_size)
        self.sampler = CyclicDomainSampler({task: len(datasets[task]) for task in TASKS}, seed)

    def next(self):
        records = []
        for task in TASKS:
            records.extend(self.datasets[task][index] for index in self.sampler.take(task, self.domain_batch_size))
        return self.collator(records)

    def state_dict(self):
        return self.sampler.state_dict()

    def load_state_dict(self, state):
        self.sampler.load_state_dict(state)


def probe_members(labels, count, seed=0):
    labels = np.asarray(labels, dtype=np.int64)
    if count == len(labels):
        return np.arange(len(labels), dtype=np.int64)
    classes, frequencies = np.unique(labels, return_counts=True)
    allocation = count * frequencies / len(labels)
    quotas = np.floor(allocation).astype(int)
    order = sorted(range(len(classes)), key=lambda i: (-(allocation[i] - quotas[i]), int(classes[i])))
    for index in order[:count - int(quotas.sum())]:
        quotas[index] += 1
    rng = np.random.Generator(np.random.PCG64(seed))
    selected = [
        rng.choice(np.flatnonzero(labels == label), size=int(quota), replace=False)
        for label, quota in zip(classes, quotas)
    ]
    return np.sort(np.concatenate(selected)).astype(np.int64)


@lru_cache(maxsize=None)
def parquet_sha256(path):
    digest_value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()
