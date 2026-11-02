"""Behavioral checks for a proposed TRAIN head, not execution admission."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_rgbt.e0_context_advantage import (
    CONTINUOUS, E0ContextAdvantage, advantage_loss, context_features)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from context_e0_train_data import sequence_weights


class ContextAdvantageTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        torch.set_num_threads(1)
        self.head = E0ContextAdvantage()
        self.features = torch.randn(4, 3, 28)
        self.observed = torch.ones(4, 3, dtype=torch.bool)

    def test_initially_keep_and_unobserved_never_selected(self):
        self.assertEqual(sum(p.numel() for p in self.head.parameters()), 961)
        self.observed[1] = False
        scores = self.head(self.features, self.observed)
        self.assertEqual(scores.argmax(1).tolist(), [0] * 4)
        self.assertTrue(torch.isneginf(scores[1, 1:]).all())
        with torch.no_grad():
            self.head.net[-1].bias.fill_(1)
        self.assertEqual(self.head(self.features, self.observed).argmax(1).tolist(),
                         [1, 0, 1, 1])

    def test_fit_normalization_keeps_discrete_fields_unchanged(self):
        self.head.fit_normalization(self.features, self.observed)
        excluded = sorted(set(range(28)) - set(CONTINUOUS))
        self.assertTrue(torch.equal(self.head.mean[excluded], torch.zeros(len(excluded))))
        self.assertTrue(torch.equal(self.head.scale[excluded], torch.ones(len(excluded))))
        self.assertTrue(torch.allclose(self.head.mean[list(CONTINUOUS)],
                                      self.features[:, :, CONTINUOUS].mean((0, 1))))
        mean = self.head.mean.clone()
        self.head(self.features * 100, self.observed)
        self.assertTrue(torch.equal(mean, self.head.mean))

    def test_context_geometry_and_missing_evidence(self):
        previous = torch.tensor([[10., 20., 8., 4.]])
        current = previous.clone()
        candidates = previous[:, None].expand(-1, 3, -1).clone()
        candidates[:, 1, 0] += 8
        scores = torch.tensor([[1., 2., float("-inf"), 3.]])
        coverage = torch.ones(1, 3, 2)
        coverage[:, 1] = 0
        history = torch.arange(8.)[None]
        route = torch.tensor([[1., 4., .7]], dtype=torch.float64)
        features, observed = context_features(
            scores, torch.zeros(1, 3), torch.ones(1, 2), coverage,
            current, candidates, previous, history, route)
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(features.dtype, scores.dtype)
        self.assertEqual(observed.tolist(), [[True, False, True]])
        self.assertEqual(features[0, 0, :3].tolist(), [1., 2., 1.])
        self.assertEqual(features[0, 1, 12:16].tolist(), [1., 0., 0., 0.])
        self.assertEqual(features[0, 0, 16:24].tolist(), history[0].tolist())
        self.assertEqual(features[0, 0, 27].item(), 1.)
        features, observed = context_features(
            torch.full((1, 4), float("-inf")), torch.zeros(1, 3),
            torch.zeros(1, 2), coverage, current, candidates, previous, history, route)
        self.assertFalse(observed.any())
        self.assertTrue(torch.isfinite(features).all())

    def test_training_targets_ambiguous_overlap_and_protection_gradients(self):
        logits = torch.tensor([[0., .2, .4], [0., .2, .4]], requires_grad=True)
        observed = torch.tensor([[True, False], [True, True]])
        current = torch.tensor([.8, .05], dtype=torch.float64)
        candidates = torch.tensor([[.3, .0], [.3, .8]], dtype=torch.float64)
        loss, parts = advantage_loss(logits, observed, current, candidates,
                                     torch.ones(2), torch.ones(2))
        self.assertGreater(parts["protection"].item(), 0)
        loss.backward()
        self.assertGreater(logits.grad[0, 1].item(), 0)
        self.assertEqual(logits.grad[0, 2].item(), 0)
        self.assertLess(logits.grad[1, 1].item(), 0)
        self.assertLess(logits.grad[1, 2].item(), 0)
        self.assertTrue(torch.equal(logits.grad[:, 0], torch.zeros(2)))

    def test_fixed_sequence_weights_apply_without_batch_renormalization(self):
        scores = torch.tensor([[0., .3], [0., -.1]])
        observed = torch.ones(2, 1, dtype=torch.bool)
        current = torch.tensor([.7, .0], dtype=torch.float64)
        candidates = torch.tensor([[.2], [.7]], dtype=torch.float64)
        weights = torch.tensor([.5, 1.5])
        full, _ = advantage_loss(scores, observed, current, candidates, weights, weights)
        separate = [advantage_loss(scores[i:i+1], observed[i:i+1],
                    current[i:i+1], candidates[i:i+1], weights[i:i+1],
                    weights[i:i+1])[0] for i in range(2)]
        self.assertTrue(torch.allclose(full, torch.stack(separate).mean()))

    def test_sequence_weights_equalize_sequences_and_exclude_ineligible_sets(self):
        index = torch.tensor([0, 0, 0, 1, 1, 2])
        eligible = torch.tensor([True, True, False, True, False, False])
        weights = sequence_weights(index, eligible)
        self.assertEqual(weights.tolist(), [1.5, 1.5, 0., 3., 0., 0.])
        self.assertEqual(sequence_weights(index, torch.zeros_like(eligible)).tolist(),
                         [0.] * 6)


if __name__ == "__main__":
    unittest.main()
