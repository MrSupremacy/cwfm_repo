from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import unittest


HAS_MODEL = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))


@unittest.skipUnless(HAS_MODEL, "PyTorch/Transformers unavailable")
class FullFTTinyT5Tests(unittest.TestCase):
    def test_dense_fullft_has_only_the_backbone_group(self):
        import torch

        from task6.training.parameters import parameter_groups
        from tests.fixtures.tiny_model import tiny_model

        model, controller, batch, config, condition = tiny_model("dense-ft")
        dense, dense_controller, dense_batch, _, _ = tiny_model("dense")
        model.eval()
        dense.eval()
        controller.teacher_batch(batch)
        dense_controller.teacher_batch(dense_batch)
        with torch.no_grad():
            self.assertLess((model(**batch, use_cache=False).logits
                             - dense(**dense_batch, use_cache=False).logits).abs().max().item(), 1e-6)
        model.train()
        controller.teacher_batch(batch)
        model(**batch, use_cache=False).loss.backward()
        groups, _ = parameter_groups(model, controller, condition, config["training"])
        self.assertEqual([group["name"] for group in groups], ["backbone"])
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in groups[0]["params"]))

    def test_eight_arm_forward_backward_and_parameter_partition(self):
        from task6.common.config import ARMS
        from task6.training.parameters import parameter_groups
        from tests.fixtures.tiny_model import tiny_model

        for arm in ARMS:
            with self.subTest(arm=arm):
                variant = "aux_0.001" if arm == "G2" else "default"
                model, controller, batch, config, condition = tiny_model(arm, variant)
                model.train()
                controller.teacher_batch(batch)
                task_loss = model(**batch, use_cache=False).loss
                aux = sum(controller.aux_losses().values(), task_loss.new_zeros(()))
                (task_loss + aux).backward()
                groups, manifest = parameter_groups(model, controller, condition, config["training"])
                group = {item["name"]: item for item in groups}
                self.assertEqual(group["backbone"]["lr"], 1e-5)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in group["backbone"]["params"]))
                self.assertEqual("router" in group, condition.has_router_parameters)
                if condition.has_router_parameters:
                    self.assertEqual(group["router"]["lr"], 3e-4)
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in group["router"]["params"]))
                self.assertEqual(sum(item["parameter_count"] for item in manifest["groups"]),
                                 manifest["unique_parameter_count"])
                controller.clear_pending()

    def test_force_all_matches_dense(self):
        import torch
        from task6.common.config import ARMS
        from tests.fixtures.tiny_model import tiny_model

        dense, dense_controller, batch, _, _ = tiny_model("dense")
        dense.eval()
        dense_controller.teacher_batch(batch)
        with torch.no_grad():
            expected = dense(**batch, use_cache=False).logits
        for arm in ARMS:
            variant = "aux_0.001" if arm == "G2" else "default"
            model, controller, batch, _, _ = tiny_model(arm, variant)
            model.eval()
            for wrapper in controller.wrappers.values():
                wrapper.force_all = True
            controller.teacher_batch(batch)
            with torch.no_grad():
                actual = model(**batch, use_cache=False).logits
            self.assertLess((actual - expected).abs().max().item(), 1e-6, arm)


if __name__ == "__main__":
    unittest.main()
