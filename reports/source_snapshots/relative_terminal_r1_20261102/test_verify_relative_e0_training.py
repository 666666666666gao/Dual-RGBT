"""Reject forged terminal histories and replay actual retained checkpoint values."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from dual_rgbt.e0_relative_identity import E0RelativeIdentity
from relative_e0_training import evaluate, update_epoch
from verify_relative_e0_training import compare_metrics, verify_checkpoint, verify_history


class RelativeTerminalTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        torch.set_num_threads(1)

    def history_fixture(self):
        data = {
            "fit": {"current_coverage": torch.ones(14762, 2)},
            "holdout": {"current_coverage": torch.ones(3949, 2)}}
        history = []
        for epoch in range(1, 13):
            metrics = dict(sets=3949, keep=3949, replacements=0,
                           supervised_sets=3949, reliable_sets=2899, lost_sets=800,
                           reliable_to_below_half=0, lost_to_at_least_half=0,
                           sequence_macro_reliable_iou_delta=0.,
                           sequence_macro_iou_delta=0., loss=1.)
            history.append(dict(
                epoch=epoch, optimizer_updates=29 * epoch, epoch_seconds=1.,
                training=dict(optimizer_updates=29, supervised_sets=14762, loss=1.),
                fit=dict(metrics, sets=14762, keep=14762, supervised_sets=14762),
                holdout=dict(metrics), qualifies=False))
        return history, data

    def test_history_requires_actual_update_count_and_fixed_populations(self):
        history, data = self.history_fixture()
        self.assertEqual(verify_history(history, data, 512), (None, 348))
        for key, value in (("optimizer_updates", 0), ("qualifies", True)):
            forged = copy.deepcopy(history)
            forged[4][key] = value
            with self.assertRaises(AssertionError):
                verify_history(forged, data, 512)
        forged = copy.deepcopy(history)
        forged[4]["holdout"]["reliable_sets"] = 2898
        with self.assertRaises(AssertionError):
            verify_history(forged, data, 512)

    def test_ranking_uses_qualifying_held_metrics_and_earliest_tie(self):
        history, data = self.history_fixture()
        for index in (2, 4):
            history[index]["holdout"].update(
                lost_to_at_least_half=180, sequence_macro_iou_delta=.01)
            history[index]["qualifies"] = True
        self.assertEqual(verify_history(history, data, 512), (3, 348))

    def test_cpu_replay_rejects_metric_forgery_and_frozen_scalar_changes(self):
        head = E0RelativeIdentity()
        initial = {key: value.clone() for key, value in head.state_dict().items()}
        fields = dict(
            anchor=torch.randn(4, 2, 768), current=torch.randn(4, 2, 768),
            current_coverage=torch.ones(4, 2), candidates=torch.randn(4, 2, 2, 768),
            coverage=torch.ones(4, 2, 2), labels=torch.ones(4, 3, dtype=torch.int8),
            current_iou=torch.tensor([.9, .7, .05, .05], dtype=torch.float64),
            candidate_iou=torch.tensor([[.8, .6], [.8, .5], [.7, .2], [.1, .2]],
                                       dtype=torch.float64),
            sequence_index=torch.tensor([0, 0, 1, 1]))
        optimizer = torch.optim.AdamW(
            [p for p in head.parameters() if p.requires_grad], lr=.0003)
        update_epoch(head, optimizer, fields, 2)
        data = {"fit": fields, "holdout": fields}
        metrics = {part: evaluate(head, values, 2) for part, values in data.items()}
        checkpoint = dict(net=head.state_dict(), epoch=8, metrics=metrics,
                          provenance={"source": "unit-test"},
                          selection_partition="train_component_holdout",
                          test_eligible=False, total_parameters=116482)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "epoch_008.pth"
            torch.save(checkpoint, path)
            verify_checkpoint(path, 8, metrics, checkpoint["provenance"], initial, data, 2)
            forged = copy.deepcopy(checkpoint)
            forged["metrics"]["holdout"]["keep"] += 1
            torch.save(forged, path)
            with self.assertRaises(AssertionError):
                verify_checkpoint(path, 8, forged["metrics"], checkpoint["provenance"],
                                  initial, data, 2)
            forged = copy.deepcopy(checkpoint)
            forged["net"]["no_match"].add_(1)
            torch.save(forged, path)
            with self.assertRaises(AssertionError):
                verify_checkpoint(path, 8, metrics, checkpoint["provenance"], initial, data, 2)

    def test_metrics_require_exact_counts_and_finite_close_floats(self):
        compare_metrics({"keep": 2, "loss": .1}, {"keep": 2, "loss": .10000001})
        for recorded in ({"keep": 3, "loss": .1}, {"keep": 2, "loss": .2},
                         {"keep": 2, "loss": float("nan")}):
            with self.assertRaises(AssertionError):
                compare_metrics({"keep": 2, "loss": .1}, recorded)


if __name__ == "__main__":
    unittest.main()
