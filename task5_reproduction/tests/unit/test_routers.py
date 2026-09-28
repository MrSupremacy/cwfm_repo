import importlib.util
from unittest import TestCase, skipUnless


@skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; router execution tests await model environment")
class RouterTests(TestCase):
    def setUp(self):
        import torch
        from task5.common.config import load_config
        self.torch = torch
        self.spec = load_config()["routing"]
        self.cents = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32 + .1
        self.x = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8.], [0., 1., 2., 3., 4., 5., 6., 7.]])

    def router(self, arm, aux=0.0, seed=0):
        from task5.routing.routers import Router
        return Router(arm, self.cents, self.spec, {"aux_weight": aux}, seed, "encoder_layer_00")

    def test_r4_r2init_uses_exact_centroid_but_remains_trainable(self):
        t = self.torch
        first = self.router("R4-R2Init", seed=0)
        second = self.router("R4-R2Init", seed=2)
        frozen = self.router("R3")
        self.assertTrue(t.equal(first.summary, self.cents))
        self.assertTrue(t.equal(second.summary, self.cents))
        self.assertTrue(first.summary.requires_grad)
        selected, weights = first(self.x, 2, q=None, valid=t.ones(2, dtype=t.bool))
        expected, hard_weights = frozen(self.x, 2, q=None, valid=t.ones(2, dtype=t.bool))
        self.assertTrue(t.equal(selected, expected))
        self.assertIsNone(hard_weights)
        self.assertTrue(t.allclose(weights.sum(-1), t.full((2,), 2.0)))
        weights[:, 0].sum().backward()
        self.assertIsNotNone(first.summary.grad)

    def test_r2_soft_is_fixed_cosine_with_unscaled_selected_softmax(self):
        t = self.torch
        soft, hard = self.router("R2-soft"), self.router("R2")
        self.assertFalse(any(p.requires_grad for p in soft.parameters()))
        self.assertTrue(t.equal(soft.centroids, hard.centroids))
        x = self.x.clone().requires_grad_()
        kwargs = {"q": None, "valid": t.ones(2, dtype=t.bool)}
        indices, weights = soft(x, 2, **kwargs)
        self.assertTrue(t.equal(indices, hard(x, 2, **kwargs)[0]))
        normalize = t.nn.functional.normalize
        scores = normalize(x, dim=-1, eps=1e-12) @ normalize(self.cents, dim=-1, eps=1e-12).T
        self.assertTrue(t.equal(weights, 2 * t.softmax(scores.gather(-1, indices), dim=-1)))
        weights[:, 0].sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_r4_hard_coefficient_st_forward_and_derivatives(self):
        t = self.torch
        hard, soft = self.router("R4-hard", seed=2), self.router("R4", seed=2)
        self.assertTrue(t.equal(hard.summary, soft.summary))
        xh, xs = (self.x.clone().requires_grad_() for _ in range(2))
        kwargs = {"q": None, "valid": t.ones(2, dtype=t.bool)}
        ih, wh = hard(xh, 2, **kwargs)
        iss, ws = soft(xs, 2, **kwargs)
        self.assertTrue(t.equal(ih, iss))
        self.assertTrue(t.equal(wh, t.ones_like(wh)))
        expert_output = t.tensor([[2., 7.], [3., 1.]], requires_grad=True)
        (expert_output * wh).sum().backward()
        (expert_output.detach() * ws).sum().backward()
        self.assertTrue(t.equal(expert_output.grad, t.ones_like(expert_output)))
        self.assertTrue(t.equal(hard.summary.grad, soft.summary.grad))
        self.assertGreater(hard.summary.grad.abs().sum().item(), 0)
        self.assertTrue(t.equal(xh.grad, xs.grad))
        self.assertGreater(xh.grad.abs().sum().item(), 0)
        with t.no_grad():
            self.assertTrue(t.equal(hard.eval()(self.x, 2, **kwargs)[1], t.ones_like(wh)))

    def test_common_gate_initialization_is_paired(self):
        t = self.torch
        gates = [self.router("G1"), self.router("G2", .1), self.router("G3"), self.router("G4")]
        for gate in gates[1:]:
            self.assertTrue(t.equal(gates[0].weight, gate.weight))
            self.assertTrue(t.equal(gates[0].gate_bias, gate.gate_bias))

    def test_noise_does_not_consume_global_rng(self):
        t = self.torch
        gate = self.router("G3").train()
        state = t.get_rng_state().clone()
        gate(self.x, 2, q=None, valid=t.tensor([True, True]))
        self.assertTrue(t.equal(state, t.get_rng_state()))

    def test_g3_noise_state_reload_and_eval(self):
        t = self.torch
        first, second = self.router("G3"), self.router("G3")
        first(self.x, 2, q=None, valid=t.ones(2, dtype=t.bool))
        second.load_noise_state(first.noise_state())
        a = first(self.x, 2, q=None, valid=t.ones(2, dtype=t.bool))
        b = second(self.x, 2, q=None, valid=t.ones(2, dtype=t.bool))
        for x, y in zip(a, b):
            self.assertTrue(t.equal(x, y))
        first.eval()
        a = first(self.x, 2, q=None, valid=t.ones(2, dtype=t.bool))
        b = first(self.x, 2, q=None, valid=t.ones(2, dtype=t.bool))
        self.assertTrue(t.equal(a[0], b[0]))

    def test_g2_aux_excludes_padding_and_uses_full_probs(self):
        t = self.torch
        gate = self.router("G2", .01).train()
        selected, _ = gate(self.x, 2, q=None, valid=t.tensor([True, False]))
        p = t.softmax(t.nn.functional.linear(self.x[:1], gate.weight, gate.gate_bias), -1)[0]
        f = t.bincount(selected[:1].flatten(), minlength=4).float() / 2
        self.assertTrue(t.allclose(gate.aux, .01 * 4 * (f*p).sum()))

    def test_g4_bias_selects_but_does_not_reweight(self):
        t = self.torch
        gate = self.router("G4").eval()
        with t.no_grad():
            gate.beta.copy_(t.tensor([100., 100., 0., 0.]))
        selected, weights = gate(self.x, 2, q=None, valid=t.ones(2, dtype=t.bool))
        logits = t.nn.functional.linear(self.x, gate.weight, gate.gate_bias)
        expected = 2 * t.softmax(logits.gather(-1, selected), -1)
        self.assertTrue(t.allclose(weights, expected))
        self.assertTrue(t.equal(gate.pending, t.zeros(4, dtype=t.long)))

    def test_g4_updates_once_after_step(self):
        t = self.torch
        gate = self.router("G4").train()
        gate(self.x, 2, q=None, valid=t.tensor([True, False]))
        counts = gate.pending.clone().float()
        expected = .001 * t.sign(counts.mean() - counts)
        gate.after_step()
        self.assertTrue(t.equal(gate.beta, expected))
        self.assertEqual(gate.pending.sum().item(), 0)
        gate.after_step()
        self.assertTrue(t.equal(gate.beta, expected))

    def test_soft_weights_sum_to_k_and_backpropagate(self):
        t = self.torch
        for arm in ("R4", "R4-R2Init", "G1", "G2", "G3", "G4"):
            gate = self.router(arm, .01 if arm == "G2" else 0)
            _, weights = gate(self.x, 2, q=None, valid=t.ones(2, dtype=t.bool))
            self.assertTrue(t.allclose(weights.sum(-1), t.full((2,), 2.)))
            weights[:, 0].sum().backward()
            self.assertTrue(any(p.grad is not None for p in gate.parameters()))

    def test_ties_and_hash_budget_prefix(self):
        from task5.routing.routers import Router, make_hash_table, stable_topk
        t = self.torch
        self.assertEqual(stable_topk(t.ones(1, 4), 2).tolist(), [[0, 1]])
        table = make_hash_table(10, 4, 0)
        self.assertEqual(table.sort(-1).values.tolist(), [list(range(4))] * 10)
        gate = Router("G0", self.cents, self.spec, {}, 0, "unused", table)
        a, _ = gate(self.x, 2, q=None, valid=t.ones(2, dtype=t.bool), token_ids=t.tensor([0, 7]))
        b, _ = gate(self.x, 3, q=None, valid=t.ones(2, dtype=t.bool), token_ids=t.tensor([0, 7]))
        self.assertTrue(t.equal(a, b[:, :2]))
