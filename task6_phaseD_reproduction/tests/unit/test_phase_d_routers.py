import importlib.util
import json
from unittest import TestCase, skipUnless


@skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class PhaseDRouterTests(TestCase):
    def setUp(self):
        import torch
        from task6_phased.common.config import load_config
        self.torch = torch
        self.spec = load_config()["routing"]
        self.centroids = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32 + .1
        self.hidden = torch.tensor([[1.,2.,3.,4.,5.,6.,7.,8.],[0.,1.,2.,3.,4.,5.,6.,7.]])

    def router(self, arm, seed=0):
        from task6_phased.routing.routers import Router
        variant = {"aux_weight": .001} if arm == "G2-0.001" else {}
        return Router(arm, self.centroids, self.spec, variant, seed, "encoder_layer_00")

    def test_six_arms_and_paired_gate_initialization(self):
        for arm in ("R2", "R4o", "R4d", "G1", "G2-0.001", "G4"):
            self.router(arm)
        gates = [self.router(arm, 2) for arm in ("G1", "G2-0.001", "G4")]
        for gate in gates[1:]:
            self.assertTrue(self.torch.equal(gates[0].weight, gate.weight))
            self.assertTrue(self.torch.equal(gates[0].gate_bias, gate.gate_bias))

    def test_weights_sum_to_k_and_g4_updates_once(self):
        torch = self.torch
        valid = torch.ones(2, dtype=torch.bool)
        for arm in ("R4o", "R4d", "G1", "G2-0.001", "G4"):
            router = self.router(arm).train()
            _, weights = router(self.hidden, 2, valid=valid)
            self.assertTrue(torch.allclose(weights.sum(-1), torch.full((2,), 2.)))
        g4 = self.router("G4").train()
        g4(self.hidden, 2, valid=valid)
        counts = g4.pending.clone().float()
        expected = g4.beta + .001 * torch.sign(counts.mean() - counts)
        g4.after_step()
        self.assertTrue(torch.equal(g4.beta, expected))
        self.assertEqual(g4.last_update["pending_counts_after_update"], 0)
        json.dumps(g4.last_update, allow_nan=False)
        g4.after_step()
        self.assertTrue(torch.equal(g4.beta, expected))

        g2 = self.router("G2-0.001").train()
        g2(self.hidden, 2, valid=valid)
        self.assertAlmostEqual(g2.last_stats["fraction_sum"], 1.0)
        self.assertAlmostEqual(g2.last_stats["probability_sum"], 1.0)
        json.dumps(g2.last_stats, allow_nan=False)
