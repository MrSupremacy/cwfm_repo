import importlib.util
from unittest import TestCase, skipUnless


HAS_MODEL = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))


@skipUnless(HAS_MODEL, "PyTorch/Transformers unavailable")
class PhaseBTinyT5Tests(TestCase):
    def test_force_all_matches_original_dense_for_all_six_arms(self):
        import torch
        from tests.fixtures.tiny_model import tiny_model

        reference, _, batch = tiny_model("R2")
        # The wrapped model with force_all is the unchanged dense computation.
        reference.eval()
        with torch.no_grad():
            for wrapper in reference.encoder.block[0].layer[-1].DenseReluDense, reference.decoder.block[0].layer[-1].DenseReluDense:
                wrapper.force_all = True
            expected = reference(**batch, use_cache=False).logits
        for arm in ("R2", "R4o", "R4d", "G1", "G2-0.001", "G4"):
            model, controller, batch = tiny_model(arm)
            model.eval()
            controller.teacher_batch(batch)
            for wrapper in controller.wrappers.values():
                wrapper.force_all = True
            with torch.no_grad():
                actual = model(**batch, use_cache=False).logits
            self.assertLess((actual - expected).abs().max().item(), 1e-5, arm)

    def test_f0_updates_only_router_parameters(self):
        import torch
        from tests.fixtures.tiny_model import tiny_model

        for arm in ("R4o", "R4d", "G1", "G2-0.001", "G4"):
            model, controller, batch = tiny_model(arm)
            before = {name: value.detach().clone() for name, value in model.named_parameters()}
            trainable = [value for value in model.parameters() if value.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=1e-3)
            model.train()
            controller.teacher_batch(batch)
            loss = model(**batch, use_cache=False).loss + controller.aux_loss()
            loss.backward()
            optimizer.step()
            controller.after_step()
            for name, value in model.named_parameters():
                if value.requires_grad:
                    self.assertIsNotNone(value.grad, (arm, name))
                else:
                    self.assertIsNone(value.grad, (arm, name))
                    self.assertTrue(torch.equal(value, before[name]), (arm, name))
