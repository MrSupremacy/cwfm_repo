"""Real datasets.map multiprocessing/cache regression, with no downloads or model."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
import importlib.util

HAS_DATA = all(importlib.util.find_spec(name) for name in ("datasets", "transformers", "tokenizers", "pyarrow"))


@skipUnless(HAS_DATA, "datasets/Transformers/tokenizers/PyArrow are required for real multiprocessing")
class PreprocessingTests(TestCase):
    def test_four_workers_preserve_full_cache_identity_and_reuse_shards(self):
        from datasets import Dataset
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from task5.common.config import digest, load_config
        from task5.common.io import read_json
        from task5.data.datasets import cache_declaration, tokenize

        backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "sst2": 2, "sentence": 3,
                                       ":": 4, "sample": 5, "positive": 6, "negative": 7}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
        dataset = Dataset.from_dict({"sentence": [f"sample {i}" for i in range(12)], "label": [i % 2 for i in range(12)]})
        identity = {"data": "synthetic-preprocessing-regression"}
        config = load_config()
        config["data"]["preprocess_workers"] = 4
        with TemporaryDirectory() as temp:
            config["execution"]["output_root"] = temp
            declaration = cache_declaration(config, "sst2", dataset, tokenizer, identity, "validation")
            full_key = digest(declaration)
            cache = Path(temp).resolve() / "tmp/preprocessing/tokenized/sst2/validation" / full_key
            encoded = tokenize(config, "sst2", dataset, tokenizer, identity, "validation")
            self.assertEqual(encoded["sample_id"], list(range(12)))
            self.assertEqual(encoded["class_id"], dataset["label"])
            self.assertEqual(encoded["labels"], [[7 if i % 2 == 0 else 6] for i in range(12)])
            self.assertEqual(len(full_key), 64)
            self.assertEqual(read_json(cache / "declaration.json"), declaration)
            self.assertEqual(encoded._fingerprint, full_key[:32])
            shards = sorted(cache.glob("tokens_*.arrow"))
            self.assertEqual(len(shards), 4)
            before = {p.name: p.stat().st_mtime_ns for p in shards}
            again = tokenize(config, "sst2", dataset, tokenizer, identity, "validation")
            self.assertEqual(again.to_dict(), encoded.to_dict())
            self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in shards})
