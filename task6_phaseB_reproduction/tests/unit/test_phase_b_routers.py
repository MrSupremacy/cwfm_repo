import importlib.util
from unittest import TestCase, skipUnless


@skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class PhaseBRouterTests(TestCase):
    def setUp(self):
        import torch
        from task6_phaseb.common.config import load_config

        self.torch = torch
        self.spec = load_config()["routing"]
        self.centroids = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32 + .1
        self.hidden = torch.tensor([
            [1., 2., 3., 4., 5., 6., 7., 8.],
            [0., 1., 2., 3., 4., 5., 6., 7.],
        ])

    def router(self, arm, seed=0):
        from task6_phaseb.routing.routers import Router

        variant = {"aux_weight": .001} if arm == "G2-0.001" else {}
        return Router(arm, self.centroids, self.spec, variant, seed, "encoder_layer_00")

    def test_only_six_arms_construct(self):
        for arm in ("R2", "R4o", "R4d", "G1", "G2-0.001", "G4"):
            self.router(arm)
        with self.assertRaises(ValueError):
            self.router("G3")

    def test_r4d_exact_centroid_and_r4o_reproducible_orthogonal(self):
        t = self.torch
        first, second = self.router("R4d", 0), self.router("R4d", 2)
        self.assertTrue(t.equal(first.summary, self.centroids))
        self.assertTrue(t.equal(second.summary, self.centroids))
        self.assertTrue(first.summary.requires_grad)
        r4a, r4b = self.router("R4o", 1), self.router("R4o", 1)
        self.assertTrue(t.equal(r4a.summary, r4b.summary))
        self.assertFalse(t.equal(r4a.summary, self.centroids))

    def test_gate_initialization_is_paired(self):
        t = self.torch
        gates = [self.router(arm, 2) for arm in ("G1", "G2-0.001", "G4")]
        for gate in gates[1:]:
            self.assertTrue(t.equal(gates[0].weight, gate.weight))
            self.assertTrue(t.equal(gates[0].gate_bias, gate.gate_bias))

    def test_soft_weights_sum_to_k_and_backpropagate(self):
        t = self.torch
        valid = t.ones(2, dtype=t.bool)
        for arm in ("R4o", "R4d", "G1", "G2-0.001", "G4"):
            router = self.router(arm).train()
            _, weights = router(self.hidden, 2, valid=valid)
            self.assertTrue(t.allclose(weights.sum(-1), t.full((2,), 2.)))
            loss = weights[:, 0].sum() + (router.aux if router.aux is not None else 0)
            loss.backward()
            self.assertTrue(all(p.grad is not None for p in router.parameters() if p.requires_grad))

    def test_g2_uses_valid_tokens_and_full_probabilities(self):
        t = self.torch
        gate = self.router("G2-0.001").train()
        selected, _ = gate(self.hidden, 2, valid=t.tensor([True, False]))
        probabilities = t.softmax(t.nn.functional.linear(self.hidden[:1], gate.weight, gate.gate_bias), -1)[0]
        fractions = t.bincount(selected[:1].flatten(), minlength=4).float() / 2
        self.assertTrue(t.allclose(gate.aux, .001 * 4 * (fractions * probabilities).sum()))

    def test_g4_bias_selects_without_reweighting_and_updates_once(self):
        t = self.torch
        gate = self.router("G4").train()
        with t.no_grad():
            gate.beta.copy_(t.tensor([100., 100., 0., 0.]))
        selected, weights = gate(self.hidden, 2, valid=t.tensor([True, False]))
        logits = t.nn.functional.linear(self.hidden, gate.weight, gate.gate_bias)
        self.assertTrue(t.allclose(weights, 2 * t.softmax(logits.gather(-1, selected), -1)))
        counts = gate.pending.clone().float()
        expected = gate.beta + .001 * t.sign(counts.mean() - counts)
        gate.after_step()
        self.assertTrue(t.equal(gate.beta, expected))
        self.assertEqual(gate.pending.sum().item(), 0)
        gate.after_step()
        self.assertTrue(t.equal(gate.beta, expected))
