import importlib.util
from unittest import TestCase, skipUnless


@skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class R3RMSRegression(TestCase):
    def test_r3_preserves_rms_epsilon_not_forced_r2_equivalence(self):
        import torch
        from task5.common.config import load_config
        from task5.routing.routers import Router
        D, E = 512, 64
        x = torch.zeros(1, D)
        x[0, 0] = 1
        s = torch.zeros(E, D)
        s[0, 0] = 1e-4
        cosine = torch.linspace(.95, -.95, E-1)
        s[1:, 0], s[1:, 1] = cosine, (1-cosine.square()).sqrt()
        spec = load_config()["routing"]
        r2 = Router("R2", s, spec, {}, None, "test")
        r3 = Router("R3", s, spec, {}, None, "test")
        a, _ = r2(x, 6, q=None, valid=torch.ones(1, dtype=torch.bool))
        b, weights = r3(x, 6, q=None, valid=torch.ones(1, dtype=torch.bool))
        self.assertEqual(a.tolist(), [[0, 1, 2, 3, 4, 5]])
        self.assertEqual(b.tolist(), [[1, 2, 3, 4, 5, 6]])
        self.assertIsNone(weights)
        self.assertFalse(r3.summary.requires_grad)
