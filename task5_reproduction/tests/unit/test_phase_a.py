from copy import deepcopy
from unittest import TestCase

from task5.cli import main, parser, select_conditions
from task5.common.config import Condition, load_config, protocol_id, recorded_protocol, repository_root
from task5.aggregation.phase_a import compatible_config


def phase_config():
    root = repository_root()
    return load_config(root / "configs/suites/phaseA_f0.yaml", root / "configs/extensions/phaseA_f0.yaml")


class PhaseAConfigTests(TestCase):
    def test_two_new_arms_only_and_eight_shards(self):
        c = phase_config()
        args = parser().parse_args(["capture", "--part", "A"])
        selected = select_conditions(c, args)
        self.assertEqual(len(selected), 32)
        self.assertEqual(sum(x.trainable for x in selected), 24)
        self.assertTrue(all(x.seed is None for x in selected if x.arm == "R2-soft"))
        self.assertEqual(sum(11 if x.trainable else 1 for x in selected), 272)
        partitions = [select_conditions(c, parser().parse_args(["train", "--shard-count", "8", "--shard-index", str(i)])) for i in range(8)]
        self.assertEqual([len(p) for p in partitions], [3] * 8)
        self.assertEqual(len({x for p in partitions for x in p}), 24)
        self.assertTrue(all(x.arm == "R4-hard" for p in partitions for x in p))

    def test_base_config_same_but_protocol_not_impersonated(self):
        c = phase_config()
        compatible_config(load_config(), c)
        self.assertEqual(recorded_protocol(c, Condition("sst2", "dense"), "formal20260830a"), c["extension"]["base_protocol"])
        for arm in ("R2-soft", "R4-hard"):
            self.assertEqual(recorded_protocol(c, Condition("sst2", arm), "formal20260830a"), protocol_id(c))
        for section, key in (("training", "lr"), ("routing", "temperature"), ("capture", "probe_seed"), ("metrics", "random_seed")):
            wrong = deepcopy(c)
            wrong[section][key] += 1
            with self.assertRaises(ValueError):
                compatible_config(c, wrong)

    def test_no_old_arm_writes_or_existing_result_commands(self):
        root = repository_root()
        common = ["--suite", str(root / "configs/suites/phaseA_f0.yaml"), "--local", str(root / "configs/extensions/phaseA_f0.yaml"), "--run-id", "formal20260830a"]
        for cmd in (["train", "--arm", "R4"], ["capture", "--part", "E"], ["aggregate"], ["tables"], ["figures"], ["prepare"]):
            with self.subTest(cmd=cmd), self.assertRaises(ValueError):
                main([*cmd, *common])
