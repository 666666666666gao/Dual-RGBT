"""Exercise real optimizer updates and the fixed component TRAIN selection."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from dual_rgbt.e0_relative_identity import E0RelativeIdentity
from relative_e0_training import acceptance_rank, evaluate, update_epoch


class RelativeTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        torch.set_num_threads(1)
        self.head = E0RelativeIdentity()
        self.data = dict(
            anchor=torch.randn(4, 2, 768), current=torch.randn(4, 2, 768),
            current_coverage=torch.ones(4, 2), candidates=torch.randn(4, 2, 2, 768),
            coverage=torch.ones(4, 2, 2), labels=torch.ones(4, 3, dtype=torch.int8),
            current_iou=torch.tensor([.9, .7, .05, .05], dtype=torch.float64),
            candidate_iou=torch.tensor([[.8, .6], [.8, .5], [.7, .2], [.1, .2]],
                                       dtype=torch.float64),
            sequence_index=torch.tensor([0, 0, 1, 1]))
        self.data["current_coverage"][3] = 0

    def test_real_updates_preserve_frozen_scalar_and_exclude_uncovered_states(self):
        before = {key: value.clone() for key, value in self.head.state_dict().items()}
        optimizer = torch.optim.AdamW(
            [p for p in self.head.parameters() if p.requires_grad], lr=.0003)
        result = update_epoch(self.head, optimizer, self.data, 2)
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertEqual(result["supervised_sets"], 3)
        self.assertTrue(torch.equal(before["no_match"], self.head.no_match))
        self.assertTrue(any(not torch.equal(before[key], value)
                            for key, value in self.head.state_dict().items()))
        metrics = evaluate(self.head, self.data, 2)
        self.assertEqual(metrics["sets"], 4)
        self.assertEqual(metrics["supervised_sets"], 3)
        self.assertEqual(metrics["reliable_sets"], 2)
        self.assertEqual(metrics["lost_sets"], 2)

    def test_eval_keeps_unknown_candidates_visible_and_keep_only_rows_counted(self):
        self.data["candidates"][:, 0] = self.data["current"]
        self.data["coverage"][:, 1] = 0
        self.data["labels"][:, 1] = -1
        result = evaluate(self.head, self.data, 2)
        self.assertEqual(result["keep"], 4)
        self.assertEqual(result["mean_iou_delta"], 0)
        self.assertEqual(result["selected_original_unknown"], 0)
        self.assertEqual(result, evaluate(self.head, self.data, 4))

    def test_acceptance_requires_gate_then_fixed_ranking_and_earliest_tie(self):
        metrics = dict(reliable_sets=2899, lost_sets=800, reliable_to_below_half=0,
                       sequence_macro_reliable_iou_delta=0.,
                       lost_to_at_least_half=180, sequence_macro_iou_delta=.01)
        self.assertGreater(acceptance_rank(metrics, 1), acceptance_rank(metrics, 2))
        self.assertIsNone(acceptance_rank(dict(metrics, reliable_to_below_half=1), 1))
        self.assertIsNone(acceptance_rank(dict(metrics, lost_to_at_least_half=179), 1))


if __name__ == "__main__":
    unittest.main()
