import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch unavailable")
class RouterTests(unittest.TestCase):
    def setUp(self):
        import torch
        from task6.common.config import load_config
        self.torch = torch
        self.config = load_config()
        self.centroids = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32
        self.x = torch.randn(7, 8, generator=torch.Generator().manual_seed(7), requires_grad=True)
        self.valid = torch.tensor([1, 1, 1, 1, 1, 0, 0], dtype=torch.bool)

    def router(self, arm):
        from task6.routing.routers import Router
        variant = {"aux_weight": 0.001} if arm == "G2" else {}
        return Router(arm, self.centroids, self.config["routing"], variant, 0, "encoder_layer_00")

    def test_all_eight_arms_have_valid_routes(self):
        from task6.common.config import ARMS
        for arm in ARMS:
            with self.subTest(arm=arm):
                router = self.router(arm)
                router.train()
                selected, weights = router(self.x, 2, valid=self.valid)
                self.assertEqual(tuple(selected.shape), (7, 2))
                self.assertTrue(self.torch.all(selected[:, 0] != selected[:, 1]))
                if weights is not None:
                    self.torch.testing.assert_close(weights.sum(-1), self.torch.full((7,), 2.0))
                router.clear_pending()

    def test_stable_ties_choose_low_ids(self):
        from task6.routing.routers import stable_topk
        selected = stable_topk(self.torch.zeros(3, 4), 2)
        self.assertTrue(self.torch.equal(selected, self.torch.tensor([[0, 1], [0, 1], [0, 1]])))

    def test_r2_soft_backpropagates_to_hidden_but_centroid_is_fixed(self):
        router = self.router("R2-soft")
        _, weights = router(self.x, 2, valid=self.valid)
        coefficients = self.torch.arange(1, weights.numel() + 1, dtype=weights.dtype).reshape_as(weights)
        (weights * coefficients).sum().backward()
        self.assertGreater(self.x.grad.abs().sum().item(), 0)
        self.assertFalse(any(True for _ in router.parameters()))

    def test_r4o_hard_forward_is_one_and_straight_through_has_gradient(self):
        router = self.router("R4o-hard")
        _, weights = router(self.x, 2, valid=self.valid)
        self.assertTrue(self.torch.equal(weights, self.torch.ones_like(weights)))
        coefficients = self.torch.arange(1, weights.numel() + 1, dtype=weights.dtype).reshape_as(weights)
        (weights * coefficients).sum().backward()
        self.assertGreater(router.summary.grad.abs().sum().item(), 0)
        self.assertGreater(self.x.grad.abs().sum().item(), 0)

    def test_g2_aux_and_g4_successful_step_state(self):
        g2 = self.router("G2").train()
        g2(self.x, 2, valid=self.valid)
        self.assertIsNotNone(g2.aux)
        g2.aux.backward(retain_graph=True)
        self.assertGreater(g2.weight.grad.abs().sum().item(), 0)
        g4 = self.router("G4").train()
        g4(self.x.detach(), 2, valid=self.valid)
        before = g4.beta.clone()
        self.assertGreater(g4.pending.sum().item(), 0)
        g4.after_step()
        self.assertFalse(self.torch.equal(before, g4.beta))
        self.assertEqual(g4.pending.sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
