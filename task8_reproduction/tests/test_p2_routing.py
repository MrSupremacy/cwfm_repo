import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch unavailable locally; required on server")
class P2RoutingTests(unittest.TestCase):
    def test_fixed_selector_and_original_endpoints(self):
        import torch
        from task8_p01.routing.interventions import route
        from task8_p2.routing import temperature_route
        torch.manual_seed(11)
        hidden = torch.randn(31,8)
        summary = torch.randn(7,8)
        routing = {"l2_epsilon":1e-12,"rms_epsilon":1e-6,"temperature":1.0}
        ids0,weights0 = route("M11",hidden,summary,summary,3)
        for m in (1,2,4,8,"uniform"):
            ids,weights = temperature_route(hidden,summary,3,routing,m)
            self.assertTrue(torch.equal(ids,ids0))
            self.assertTrue(torch.allclose(weights.sum(-1),torch.full((31,),3.)))
            if m==1:
                self.assertTrue(torch.equal(weights,weights0))
            if m=="uniform":
                uniform_ids,uniform_weights = route("M10",hidden,summary,summary,3)
                self.assertTrue(torch.equal(ids,uniform_ids))
                self.assertTrue(torch.equal(weights,uniform_weights))

    def test_geometry_zero_tiny_and_eta_decomposition(self):
        import numpy as np
        from task8_p2.routing import router_arrays
        from task8_p2.numerics import chain
        routing = {"l2_epsilon":1e-12,"rms_epsilon":1e-6,"temperature":1.0}
        rng = np.random.default_rng(5)
        hidden = np.vstack([np.zeros((1,8)),np.full((1,8),1e-30),rng.normal(size=(3,8))]).astype(np.float32)
        summary = rng.normal(size=(7,8)).astype(np.float32)
        summary[0] = 0
        r = router_arrays(hidden,summary,summary,3,routing,"cpu")
        states = chain(r["q_St"],r["z_actual"],r["eta_x"],r["eta_St"],r["ids_R4d"],np.sqrt(8))
        self.assertTrue(np.isfinite(states["N4"][1]).all())

    def test_trace_alignment_rejects_reordered_tokens(self):
        import numpy as np
        from task8_p2.traces import align
        a = {"sample_id":np.array([1,2]),"token_position":np.array([0,0]),"token_group":np.array([1,1])}
        b = {**a,"sample_id":np.array([2,1])}
        with self.assertRaises(ValueError):
            align(a,b)
