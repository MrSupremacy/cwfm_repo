from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
import importlib.util

HAS_MODEL = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))


@skipUnless(HAS_MODEL, "PyTorch/Transformers unavailable; tiny-T5 integration awaits model environment")
class TinyT5Tests(TestCase):
    def setUp(self):
        from task5.common.config import load_config
        from task5.common.randomness import configure_torch
        configure_torch(load_config())

    def test_force_all_matches_dense_logits(self):
        import torch
        from tests.fixtures.tiny_model import tiny_model
        dense, dense_ctrl, batch, _ = tiny_model("dense")
        dense.eval()
        dense_ctrl.teacher_batch(batch)
        with torch.no_grad():
            expected = dense(**batch, use_cache=False).logits
        for arm, variant in (("R1", "default"), ("R2", "default"), ("R3", "default"), ("R4", "default"),
                             ("R4-R2Init", "default"), ("R2-soft", "default"), ("R4-hard", "default"),
                             ("G0", "default"), ("G1", "default"), ("G2", "aux_0.01"), ("G3", "default"), ("G4", "default")):
            model, ctrl, batch, _ = tiny_model(arm, variant)
            model.eval()
            ctrl.teacher_batch(batch)
            for w in ctrl.wrappers.values():
                w.force_all = True
            with torch.no_grad():
                actual = model(**batch, use_cache=False).logits
            self.assertLess((actual-expected).abs().max().item(), 1e-5, arm)

    def test_teacher_masks_local_q_and_router_gradients(self):
        import torch
        from tests.fixtures.tiny_model import tiny_model
        model, ctrl, batch, _ = tiny_model("G2", "aux_0.01")
        seen = {}

        def observe(wrapper, shape, valid, selected, q, activation):
            seen[wrapper.key] = int(valid.sum())
            self.assertEqual(q.shape, (len(activation), 4))
            expected = activation.detach().index_select(1, wrapper.group_order).reshape(-1, 4, 8).sum(-1)
            self.assertTrue(torch.equal(q, expected))

        model.train()
        ctrl.teacher_batch(batch)
        ctrl.observe(observe, with_q=True)
        loss = model(**batch, use_cache=False).loss + ctrl.aux_loss()
        loss.backward()
        self.assertEqual(seen, {"encoder_layer_00": 5, "decoder_layer_00": 3})
        for w in ctrl.wrappers.values():
            self.assertFalse(w.wi.weight.requires_grad)
            self.assertIsNone(w.wi.weight.grad)
            self.assertIsNotNone(w.router.weight.grad)
        self.assertGreater(ctrl.wrappers["encoder_layer_00"].router.weight.grad.abs().sum().item(), 0)

    def test_hash_generation_uses_current_decoder_ids_with_cache(self):
        import torch
        from tests.fixtures.tiny_model import tiny_model
        model, ctrl, batch, _ = tiny_model("G0")
        model.eval()
        ctrl.generation()
        decoder_ids = []

        def watch(router, args, kwargs, output):
            ids = kwargs["token_ids"]
            expected = router.table[ids, :2].long()
            self.assertTrue(torch.equal(expected, output[0]))
            decoder_ids.append(ids.clone())

        ctrl.wrappers["decoder_layer_00"].router.register_forward_hook(watch, with_kwargs=True)
        tables = [w.router.table for w in ctrl.wrappers.values()]
        self.assertEqual(tables[0].data_ptr(), tables[1].data_ptr())
        with torch.no_grad():
            generated = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                       max_new_tokens=3, min_new_tokens=3, do_sample=False, num_beams=1, use_cache=True)
        self.assertEqual(len(decoder_ids), 3)
        self.assertTrue(torch.equal(decoder_ids[0], torch.zeros(2, dtype=torch.long)))
        self.assertTrue(torch.equal(decoder_ids[1], generated[:, 1]))
        self.assertTrue(torch.equal(decoder_ids[2], generated[:, 2]))

    def test_checkpoint_roundtrip_and_uninterrupted_resume(self):
        import torch
        from task5.common.randomness import seed_dropout
        from task5.training.checkpoints import restore_checkpoint, save_checkpoint
        from task5.training.schedule import scheduler_scale
        from tests.fixtures.tiny_model import tiny_model

        def setup(arm):
            model, ctrl, batch, _ = tiny_model(arm)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4,
                                         foreach=False, fused=False, weight_decay=0)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: scheduler_scale(step, 6, 1))
            shuffle = torch.Generator().manual_seed(0)
            return model, ctrl, batch, optimizer, scheduler, shuffle

        def step(items):
            model, ctrl, batch, optimizer, scheduler, shuffle = items
            model.train()
            order = torch.randperm(5, generator=shuffle)
            ctrl.teacher_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, use_cache=False).loss + ctrl.aux_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            ctrl.after_step()
            return loss.detach().clone(), order

        for arm in ("R4", "R4-R2Init", "R4-hard", "G3", "G4"):
            with self.subTest(arm=arm), TemporaryDirectory() as temp:
                first = setup(arm)
                seed_dropout(0)
                step(first)
                step(first)
                _, ctrl, _, optimizer, scheduler, shuffle = first
                path = Path(temp) / "step_2"
                save_checkpoint(path, ctrl, optimizer, scheduler, shuffle, {"step": 2, "epoch": 2, "name": "step_2"})
                expected_steps = [step(first), step(first)]
                expected_state = {layer: {key: value.clone() for key, value in state.items()} for layer, state in ctrl.state().items()}
                second = setup(arm)
                restore_checkpoint(path, second[1], second[3], second[4], second[5])
                actual_steps = [step(second), step(second)]
                for (loss_a, order_a), (loss_b, order_b) in zip(expected_steps, actual_steps):
                    self.assertTrue(torch.allclose(loss_a, loss_b, atol=1e-6, rtol=1e-5))
                    self.assertTrue(torch.equal(order_a, order_b))
                for layer, values in second[1].state().items():
                    for key, value in values.items():
                        if key == "beta" or not value.is_floating_point():
                            self.assertTrue(torch.equal(value, expected_state[layer][key]))
                        else:
                            self.assertTrue(torch.allclose(value, expected_state[layer][key], atol=1e-6, rtol=1e-5))
                self.assertEqual(first[4].state_dict()["last_epoch"], second[4].state_dict()["last_epoch"])

    def test_f0_hard_updates_routers_but_no_backbone_parameters(self):
        import torch
        from tests.fixtures.tiny_model import tiny_model
        model, ctrl, batch, _ = tiny_model("R4-hard")
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        parameters = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(parameters), len(ctrl.wrappers))
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        model.train()
        ctrl.teacher_batch(batch)
        model(**batch, use_cache=False).loss.backward()
        for wrapper in ctrl.wrappers.values():
            self.assertGreater(wrapper.router.summary.grad.abs().sum().item(), 0)
        optimizer.step()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertFalse(torch.equal(p, before[name]), name)
            else:
                self.assertIsNone(p.grad, name)
                self.assertTrue(torch.equal(p, before[name]), name)
        static, _, _, _ = tiny_model("R2-soft")
        self.assertFalse(any(p.requires_grad for p in static.parameters()))
