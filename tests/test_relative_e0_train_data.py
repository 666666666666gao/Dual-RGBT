"""Check TRAIN relative supervision and per-sequence protection accounting."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from relative_e0_train_data import action_metrics, qualifies_for_train_acceptance, targets_and_loss


class RelativeTrainDataTest(unittest.TestCase):
    def test_unknown_candidates_do_not_enter_targets_or_loss_denominator(self):
        logits = torch.tensor([[1., 2., 1000.], [1., 2., 1000.]], requires_grad=True)
        current = torch.tensor([.2, .8], dtype=torch.float64)
        candidates = torch.tensor([[.7, .9], [.7, .9]], dtype=torch.float64)
        labels = torch.tensor([[0, 1, -1], [0, 1, -1]], dtype=torch.int8)
        loss, target, trainable = targets_and_loss(
            logits, current, candidates, torch.ones(2, 2), torch.ones(2, 2, 2), labels)
        self.assertEqual(target.tolist(), [1, 0])
        self.assertTrue(trainable.all())
        expected = torch.nn.functional.cross_entropy(logits[:, :2], target)
        self.assertTrue(torch.equal(loss, expected))
        loss.backward()
        self.assertEqual(logits.grad[:, 2].tolist(), [0., 0.])

    def test_uncovered_current_is_excluded_from_loss(self):
        logits = torch.tensor([[0., float("-inf")], [1., 2.]], requires_grad=True)
        loss, target, trainable = targets_and_loss(
            logits, torch.tensor([.05, .8], dtype=torch.float64),
            torch.tensor([[.9], [.7]], dtype=torch.float64),
            torch.tensor([[0., 0.], [1., 1.]]), torch.ones(2, 1, 2),
            torch.ones(2, 2, dtype=torch.int8))
        self.assertEqual(trainable.tolist(), [False, True])
        self.assertEqual(target.tolist(), [0, 0])
        self.assertTrue(torch.isfinite(loss))

    def test_state_damage_recovery_and_sequence_macro(self):
        current = torch.tensor([.9, .6, .05, .05], dtype=torch.float64)
        candidate = torch.tensor([[.4], [.8], [.7], [.2]], dtype=torch.float64)
        chosen = torch.ones(4, dtype=torch.long)
        sequence = torch.tensor([0, 0, 0, 1], dtype=torch.long)
        labels = torch.ones(4, 2, dtype=torch.int8)
        labels[3, 1] = -1
        result = action_metrics(chosen, current, candidate, sequence, labels)
        self.assertEqual(result["reliable_to_below_half"], 1)
        self.assertEqual(result["lost_to_at_least_half"], 1)
        self.assertEqual(result["selected_original_unknown"], 1)
        self.assertAlmostEqual(result["sequence_macro_iou_delta"], ((-.5 + .2 + .65) / 3 + .15) / 2)
        self.assertAlmostEqual(result["sequence_macro_reliable_iou_delta"], -.15)
        keep = action_metrics(torch.zeros(4, dtype=torch.long), current, candidate, sequence, labels)
        self.assertEqual(keep["sequence_macro_iou_delta"], 0)
        self.assertEqual(keep["reliable_to_below_half"], 0)

    def test_predeclared_acceptance_does_not_relax_recovery_or_protection(self):
        metrics = dict(reliable_sets=2899, lost_sets=800, reliable_to_below_half=0,
                       sequence_macro_reliable_iou_delta=0., lost_to_at_least_half=180)
        self.assertTrue(qualifies_for_train_acceptance(metrics))
        for key, value in (("reliable_to_below_half", 1),
                           ("sequence_macro_reliable_iou_delta", -.00001),
                           ("lost_to_at_least_half", 179)):
            self.assertFalse(qualifies_for_train_acceptance(dict(metrics, **{key: value})))


if __name__ == "__main__":
    unittest.main()
