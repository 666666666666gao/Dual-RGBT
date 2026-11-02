"""Actual CPU update/reload and source-admission substitution checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from context_e0_train_data import sequence_weights
from context_recovery_execution import BINDINGS, OBJECTIVE, STAGE, review_sources
from context_recovery_training import train
from current_e0_supplement_contract import sha


class RecoveryExecutionTest(unittest.TestCase):
    def test_actual_five_updates_reload_and_fixed_normalization(self):
        torch.manual_seed(2027)
        data = {}
        for part, count in (("fit", 3000), ("holdout", 3949)):
            current = torch.full((count,), .05, dtype=torch.float64)
            current[:2000 if part == "fit" else 2899] = .8
            if part == "holdout":
                current[3699:] = .3
            candidate = torch.full((count, 24), .2, dtype=torch.float64)
            candidate[:, 0] = .7
            observed = torch.ones(count, 24, dtype=torch.bool)
            index = torch.arange(count) % 3
            data[part] = dict(
                features=torch.randn(count, 24, 28), observed=observed,
                current_iou=current, candidate_iou=candidate,
                sequence_index=index, labels=torch.zeros(count, 25, dtype=torch.int8),
                advantage_weight=sequence_weights(index, observed.any(1)),
                protection_weight=sequence_weights(index, current.ge(.5)))
        plan = dict(phase="smoke", seed=2027, epochs=12, batch_sets=512,
                    device="cpu", lr=.0003, weight_decay=.0001,
                    dataset_sha256="synthetic", objective=OBJECTIVE)
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            with patch("builtins.print"):
                receipt = train(data, plan, run, "bound")
            self.assertEqual(receipt["optimizer_updates"], 5)
            for key in ("parameters_changed", "strict_reload_bit_equal",
                        "initial_all_keep", "normalization_fit_only", "recovery_weights_fit_only"):
                self.assertTrue(receipt[key])
            self.assertFalse(receipt["cuda_used"])
            history = json.loads((run / "history.json").read_bytes())
            self.assertGreater(history[0]["recovery_loss"], 0)
            state = torch.load(run / "latest.pth", map_location="cpu")
            self.assertEqual(state["objective"], OBJECTIVE)
            self.assertEqual(state["plan_sha256"], "bound")
            with self.assertRaises(FileExistsError):
                train(data, plan, run, "bound")

    def test_exact_review_bindings_reject_objective_and_task_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.py").write_text("pass\n")
            draft = dict(stage=STAGE, objective=OBJECTIVE, split="train",
                         test_eligible=False, online_allowed=False,
                         scientific_result_audit_accepted=False,
                         seed=2027, epochs=12, batch_sets=512, lr=.0003, weight_decay=.0001,
                         assigned_gpu="2", target_host="2025", target_root="/snapshot",
                         source_sha256={"source.py": sha(root / "source.py")},
                         source_review="review.json", reviewed_draft="draft.json", task="fixed")
            draft.update({name: None for name in BINDINGS})
            draft["launch_allowed"] = False
            (root / "draft.json").write_text(json.dumps(draft))
            review = dict(status="SOURCE_PASS", blocking_issues=0, stage=STAGE,
                          target_root="/snapshot", source_sha256=draft["source_sha256"],
                          reviewed_drafts={"draft.json": sha(root / "draft.json")})
            (root / "review.json").write_text(json.dumps(review))
            plan = dict(draft, launch_allowed=True, source_review_sha256=sha(root / "review.json"))
            review_sources(root, plan)
            for key, value in (("objective", "changed"), ("seed", 2028),
                               ("task", "changed"), ("split", "test"),
                               ("assigned_gpu", "0"), ("target_root", "/elsewhere")):
                with self.assertRaises(AssertionError, msg=key):
                    review_sources(root, dict(plan, **{key: value}))
            (root / "source.py").write_text("changed\n")
            with self.assertRaises(AssertionError):
                review_sources(root, plan)


if __name__ == "__main__":
    unittest.main()
