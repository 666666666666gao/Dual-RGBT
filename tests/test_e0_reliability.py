"""Synthetic CPU engineering checks, not trained tracking evidence."""
import copy
import inspect
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_rgbt.e0_reliability import E0Reliability, localization_loss, localization_targets


class E0ReliabilityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        torch.set_num_threads(1)

    def test_overlap_boundaries(self):
        labels = torch.tensor([0., .099999, .1, .499999, .5, 1.], dtype=torch.float64)
        self.assertEqual(localization_targets(labels).tolist(), [2, 2, 1, 1, 0, 0])

    def test_invalid_overlap_rejected(self):
        for values in ([float('nan')], [-.01], [1.01]):
            with self.assertRaises(AssertionError):
                localization_targets(torch.tensor(values))

    def test_features_only_interface_and_read_only_input(self):
        model = E0Reliability().eval()
        features = torch.randn(6, 914)
        before = features.clone()
        self.assertEqual(list(inspect.signature(model.forward).parameters), ['features'])
        with torch.no_grad():
            logits = model(features)
        self.assertEqual(logits.shape, (6, 3))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 117507)
        self.assertTrue(torch.equal(before, features))
        self.assertTrue(torch.isfinite(logits).all())

    def test_optimizer_updates_only_prototype(self):
        protected = torch.nn.Linear(4, 4).eval()
        for parameter in protected.parameters():
            parameter.requires_grad_(False)
        before = copy.deepcopy(protected.state_dict())
        model = E0Reliability()
        head_before = copy.deepcopy(model.state_dict())
        features = torch.randn(12, 914)
        labels = torch.tensor([.9, .3, .01] * 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        initial = localization_loss(model(features), labels).item()
        for _ in range(10):
            optimizer.zero_grad()
            loss = localization_loss(model(features), labels)
            loss.backward()
            optimizer.step()
        self.assertLess(localization_loss(model(features), labels).item(), initial)
        self.assertTrue(all(torch.equal(value, protected.state_dict()[key])
                            for key, value in before.items()))
        self.assertTrue(all(p.grad is None for p in protected.parameters()))
        self.assertTrue(all(not torch.equal(value, model.state_dict()[key])
                            for key, value in head_before.items()))

    def test_strict_reload(self):
        model = E0Reliability().eval()
        other = E0Reliability().eval()
        other.load_state_dict(model.state_dict(), strict=True)
        features = torch.randn(3, 914)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(features), other(features)))


if __name__ == '__main__':
    unittest.main()
