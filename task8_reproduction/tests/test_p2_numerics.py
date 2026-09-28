import unittest

import numpy as np

from task8_p2.numerics import Moments, chain, probabilities, support_metrics, weighting_metrics


class P2NumericsTests(unittest.TestCase):
    def test_probability_alpha_and_effective_width_contracts(self):
        z = np.array([[0., 0., 0.], [3., 1., -2.]])
        p = probabilities(z)
        m = weighting_metrics(z, p)
        np.testing.assert_allclose(p.sum(-1), 1)
        np.testing.assert_allclose(m["n_eff_simpson_over_k"], 1/(1+m["weight_variance"]), rtol=1e-13)
        self.assertAlmostEqual(m["weight_variance"][0], 0)
        self.assertAlmostEqual(m["entropy_norm"][0], 1)
        np.testing.assert_allclose(m["alpha_max"], 3*m["p_max"])

    def test_k1_entropy_is_missing_and_logits_are_stable(self):
        p = probabilities(np.array([[10000.], [-10000.]]))
        m = weighting_metrics(np.array([[10000.], [-10000.]]), p)
        self.assertTrue(np.isnan(m["entropy_norm"]).all())
        np.testing.assert_equal(m["n_eff_simpson_over_k"], [1., 1.])
        np.testing.assert_equal(probabilities(np.array([[10000., 9999.]])), probabilities(np.array([[1., 0.]])))

    def test_support_distance_and_tie_aware_maximum(self):
        a = np.array([[0, 1], [0, 1]])
        b = np.array([[1, 2], [2, 3]])
        scores2 = np.array([[3., 2., 1., 0.], [1., 1., 1., 1.]])
        scores4 = np.array([[1., 3., 2., 0.], [1., 1., 1., 1.]])
        m = support_metrics(a, b, scores2, scores4)
        np.testing.assert_allclose(m["overlap_at_k"], [.5, 0])
        np.testing.assert_allclose(m["jaccard"], [1/3, 0])
        np.testing.assert_allclose(m["tie_aware_overlap_at_k"], [.5, 1])
        np.testing.assert_equal(m["replacement_count"], [1, 2])

    def test_tie_aware_overlap_matches_exhaustive_feasible_supports(self):
        from itertools import combinations
        rng = np.random.default_rng(19)
        for k in (1,2,3):
            for _ in range(20):
                scores = [rng.integers(0,3,size=5).astype(float) for _ in range(2)]
                selected = [np.argsort(-s,kind="stable")[:k] for s in scores]
                allowed = []
                for s in scores:
                    edge = sorted(s,reverse=True)[k-1]
                    mandatory = set(np.flatnonzero(s>edge))
                    candidates = set(np.flatnonzero(s>=edge))
                    allowed.append([set(c) for c in combinations(range(5),k) if mandatory<=set(c)<=candidates])
                maximum = max(len(a&b) for a in allowed[0] for b in allowed[1])/k
                value = support_metrics(selected[0][None,:],selected[1][None,:],scores[0][None,:],scores[1][None,:])
                self.assertAlmostEqual(value["tie_aware_overlap_at_k"][0],maximum)

    def test_numeric_chain_only_weights_externally_supplied_support(self):
        q = np.array([[.1, .9, -.5, .4]])
        ex = np.array([.8])
        es = np.array([[.1, .2, .4, .9]])
        actual = 10*q*ex[:,None]*es
        ids = np.array([[0, 2]])  # deliberately not actual top-k
        states = chain(q, actual, ex, es, ids, 10)
        self.assertEqual(list(states), ["N0", "N1", "N2", "N3", "N4"])
        np.testing.assert_allclose(states["N4"][1], probabilities(np.take_along_axis(actual,ids,-1)))
        np.testing.assert_allclose(states["N0"][1], [[.5, .5]])
        self.assertGreater(states["N2"][0]["weight_variance"][0], states["N1"][0]["weight_variance"][0])

    def test_welford_handles_unequal_chunks_and_ddof_one(self):
        values = np.array([[1., 2.], [2., 8.], [4., 3.], [20., 7.]])
        m = Moments()
        m.add(values[:1])
        m.add(values[1:])
        mean,var,std = m.values()
        np.testing.assert_allclose(mean, values.mean(0))
        np.testing.assert_allclose(var, values.var(0,ddof=1))
        np.testing.assert_allclose(std, values.std(0,ddof=1))

    def test_temperature_is_monotone_per_unit(self):
        from task8_p2.local import temperatures
        rng = np.random.default_rng(3)
        states = temperatures(rng.normal(size=(31,51))*10)
        previous = np.full(31,np.inf)
        for metrics,p in states.values():
            self.assertTrue(np.all(metrics["weight_variance"] <= previous+1e-9))
            previous = metrics["weight_variance"]
            np.testing.assert_allclose(p.sum(-1), 1)

    def test_unit_pooling_does_not_average_group_means(self):
        from task8_p2.local import Collector
        c = Collector({"E":64,"k":6})
        meta = {"task":"sst2","layer_id":"encoder_layer_00","stack":"encoder",
                "hidden_hash":"chunk-a","forward_mode":"natural"}
        for values in (np.array([1.]),np.array([3.,3.,3.])):
            tokens = {"sample_id":np.arange(len(values)),"token_group":np.zeros(len(values),int)}
            c.add("weights",meta,tokens,{"v":values})
        row = next(r for r in c.rows("weights") if r["task"]==r["layer_id"]==r["stack"]=="__all__")
        self.assertAlmostEqual(row["v"],2.5)
        self.assertEqual(row["n_units"],4)


if __name__ == "__main__":
    unittest.main()
