import importlib.util
from unittest import TestCase, skipUnless


@skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class RoutingTests(TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(1)
        self.hidden = torch.randn(5, 8)
        self.c0 = torch.randn(4, 8)
        self.learned = self.c0 + 0.2 * torch.randn(4, 8)

    def call(self, mode, k=2, learned=None):
        from task8_p01.routing.interventions import route
        return route(mode, self.hidden, self.c0, self.learned if learned is None else learned, k)

    def test_four_cells_have_expected_support_and_weight_contracts(self):
        torch = self.torch
        m00_i, m00_w = self.call("M00")
        m01_i, m01_w = self.call("M01")
        m10_i, m10_w = self.call("M10")
        m11_i, m11_w = self.call("M11")
        self.assertTrue(torch.equal(m00_i, m01_i))
        self.assertTrue(torch.equal(m10_i, m11_i))
        self.assertTrue(torch.equal(m00_w, torch.ones_like(m00_w)))
        self.assertTrue(torch.equal(m10_w, torch.ones_like(m10_w)))
        self.assertTrue(torch.allclose(m01_w.sum(-1), torch.full((5,), 2.0)))
        self.assertTrue(torch.allclose(m11_w.sum(-1), torch.full((5,), 2.0)))

    def test_weighter_cannot_retopk_supplied_support(self):
        from task8_p01.routing.interventions import rms_logits, weighter
        torch = self.torch
        supplied = torch.tensor([[3, 1]]).expand(5, -1).contiguous()
        state = {"kind": "r4d_soft", "summaries": self.learned, "epsilon": 1.0e-6, "temperature": 1.0}
        weights = weighter(self.hidden, state, supplied)
        expected = 2 * torch.softmax(rms_logits(self.hidden, self.learned).gather(-1, supplied), -1)
        self.assertTrue(torch.equal(weights, expected))

    def test_zero_tiny_tie_and_full_budget(self):
        from task8_p01.routing.interventions import route
        torch = self.torch
        for hidden in (torch.zeros(2, 8), torch.full((2, 8), 1.0e-30)):
            indices, weights = route("M11", hidden, self.c0, self.learned, 2)
            self.assertTrue(torch.isfinite(weights).all())
        tied = torch.ones_like(self.c0)
        indices, _ = route("M11", torch.ones(2, 8), self.c0, tied, 2)
        self.assertTrue(torch.equal(indices, torch.tensor([[0, 1], [0, 1]])))
        indices, weights = route("M00", self.hidden, self.c0, self.learned, 4)
        self.assertTrue(torch.equal(torch.sort(indices, -1).values, torch.arange(4).expand(5, 4)))
        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))

    def test_k1_and_kgt1_gradients(self):
        torch = self.torch
        for k in (1, 3):
            learned = self.learned.clone().requires_grad_(True)
            _, weights = self.call("M11", k, learned)
            # k=1 softmax is constant; zero gradient is correct and must remain finite.
            (weights.square().sum()).backward()
            self.assertIsNotNone(learned.grad)
            self.assertTrue(torch.isfinite(learned.grad).all())
            if k > 1:
                self.assertGreater(float(learned.grad.abs().sum()), 0)

