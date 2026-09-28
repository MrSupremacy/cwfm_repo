import importlib.util
from unittest import TestCase, skipUnless


@skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class MultiTaskTrainingTests(TestCase):
    def test_sample_token_domain_loss_is_equal_domain(self):
        import torch
        from task6_phased.dense.model import sample_token_domain_loss
        logits = torch.zeros((8, 2, 2), dtype=torch.float32)
        labels = torch.tensor([[0, -100], [0, 0]] * 4)
        domains = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss, per_domain, per_sample = sample_token_domain_loss(logits, labels, domains)
        self.assertTrue(torch.allclose(per_domain, torch.full((4,), torch.log(torch.tensor(2.)))))
        self.assertTrue(torch.allclose(loss, torch.log(torch.tensor(2.))))

    def test_cyclic_sampler_resume(self):
        from task6_phased.common.config import TASKS
        from task6_phased.data.datasets import CyclicDomainSampler
        lengths = dict.fromkeys(TASKS, 5)
        left = CyclicDomainSampler(lengths, 2)
        left.take("sst2", 7)
        state = left.state_dict()
        expected = left.take("sst2", 8)
        right = CyclicDomainSampler(lengths, 2)
        right.load_state_dict(state)
        self.assertEqual(right.take("sst2", 8), expected)
