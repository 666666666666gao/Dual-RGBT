"""Check recovery weighting and gradient behavior, with no benchmark data."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from context_recovery_objective import recovery_loss, recovery_weights
from dual_rgbt.e0_context_advantage import advantage_loss


class RecoveryObjectiveTest(unittest.TestCase):
    def setUp(self):
        self.observed = torch.tensor([[True, True], [True, True], [True, False]])
        self.current = torch.tensor([.05, .05, .8], dtype=torch.float64)
        self.candidates = torch.tensor([[.8, .1], [.7, .9], [.2, .9]],
                                       dtype=torch.float64)
        self.values = dict(observed=self.observed, current_iou=self.current,
                           candidate_iou=self.candidates,
                           sequence_index=torch.tensor([0, 0, 1]))

    def test_fixed_weights_and_recovery_only_gradient(self):
        weights = recovery_weights(self.values)
        self.assertEqual(weights.tolist(), [1.5, 1.5, 0.])
        logits = torch.zeros(3, 3, requires_grad=True)
        ones = torch.ones(3)
        total, parts = recovery_loss(logits, self.observed, self.current,
                                    self.candidates, ones, ones, weights)
        original, _ = advantage_loss(logits, self.observed, self.current,
                                     self.candidates, ones, ones)
        self.assertTrue(torch.allclose(total, original + parts["recovery"]))
        gradient = torch.autograd.grad(total - original, logits)[0]
        self.assertLess(gradient[0, 1].item(), 0)
        self.assertEqual(gradient[0, 2].item(), 0)
        self.assertLess(gradient[1, 1].item(), 0)
        self.assertLess(gradient[1, 2].item(), 0)
        self.assertTrue(torch.equal(gradient[2], torch.zeros(3)))
        self.assertTrue(torch.equal(gradient[:, 0], torch.zeros(3)))

    def test_batch_partition_does_not_renormalize(self):
        logits = torch.zeros(3, 3)
        weights = recovery_weights(self.values)
        ones = torch.ones(3)
        full, _ = recovery_loss(logits, self.observed, self.current,
                                 self.candidates, ones, ones, weights)
        separate = [
            recovery_loss(logits[i:i+1], self.observed[i:i+1],
                          self.current[i:i+1], self.candidates[i:i+1],
                          ones[i:i+1], ones[i:i+1], weights[i:i+1])[0]
            for i in range(3)]
        self.assertTrue(torch.allclose(full, torch.stack(separate).mean()))

    def test_no_recovering_actions_preserves_original_loss(self):
        current = torch.full_like(self.current, .8)
        values = dict(self.values, current_iou=current)
        weights = recovery_weights(values)
        self.assertTrue(torch.equal(weights, torch.zeros(3)))
        logits = torch.zeros(3, 3)
        ones = torch.ones(3)
        total, parts = recovery_loss(logits, self.observed, current,
                                    self.candidates, ones, ones, weights)
        original, _ = advantage_loss(logits, self.observed, current,
                                     self.candidates, ones, ones)
        self.assertTrue(torch.equal(total, original))
        self.assertEqual(parts["recovery"].item(), 0)


if __name__ == "__main__":
    unittest.main()
