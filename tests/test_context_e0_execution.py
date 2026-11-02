"""Exercise actual context update engine and rejection of altered run bindings."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from context_e0_execution import BINDINGS, review_sources
from context_e0_train_data import sequence_weights
from context_e0_training import train
from current_e0_supplement_contract import sha
from verify_context_e0_training import same_checkpoint


class ContextExecutionTest(unittest.TestCase):
    def test_saved_checkpoint_content_matches_across_filenames_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            latest, final = Path(folder) / "latest.pth", Path(folder) / "epoch_012.pth"
            state = dict(epoch=12, metrics={"keep": 4}, net={"weight": torch.ones(3)})
            torch.save(state, latest)
            torch.save(state, final)
            actual = torch.load(latest, map_location="cpu")
            expected = torch.load(final, map_location="cpu")
            same_checkpoint(actual, expected)
            with self.assertRaises(AssertionError):
                same_checkpoint(dict(actual, epoch=11), expected)
            actual["net"]["weight"][0] = 2
            with self.assertRaises(AssertionError):
                same_checkpoint(actual, expected)

    def test_real_five_updates_reload_and_no_normalization_drift(self):
        torch.manual_seed(2027)
        data = {}
        for part, count in (("fit", 3000), ("holdout", 3949)):
            current = torch.full((count,), .3, dtype=torch.float64)
            if part == "holdout":
                current[:2899] = .8
                current[2899:3699] = .05
            else:
                current[:2000] = .8
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
                    device="cpu", lr=.0003, weight_decay=.0001, dataset_sha256="fixed")
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            with patch("builtins.print"):
                receipt = train(data, plan, run, "bound")
            self.assertEqual(receipt["optimizer_updates"], 5)
            self.assertTrue(receipt["parameters_changed"])
            self.assertTrue(receipt["strict_reload_bit_equal"])
            self.assertTrue(receipt["initial_all_keep"])
            self.assertFalse(receipt["cuda_used"])
            self.assertFalse(receipt["online_allowed"])
            state = torch.load(run / "latest.pth", map_location="cpu")
            self.assertEqual(state["plan_sha256"], "bound")
            self.assertEqual(state["dataset_sha256"], "fixed")
            self.assertEqual(len(json.loads((run / "history.json").read_bytes())), 1)
            with self.assertRaises(FileExistsError):
                train(data, plan, run, "bound")

    def test_review_and_draft_exact_bindings_reject_substitutions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.py"
            source.write_text("pass\n")
            draft = dict(stage="M2_context_advantage_execution", split="train",
                         test_eligible=False, online_allowed=False,
                         scientific_result_audit_accepted=False,
                         seed=2027, epochs=12, batch_sets=512, lr=.0003, weight_decay=.0001,
                         assigned_gpu="2", target_host="2025", target_root="/snapshot",
                         export_root="/producer", source_sha256={"source.py": sha(source)},
                         source_review="review.json", reviewed_draft="draft.json", task="fixed")
            draft.update({name: None for name in BINDINGS})
            draft["launch_allowed"] = False
            (root / "draft.json").write_text(json.dumps(draft))
            review = dict(status="SOURCE_PASS", blocking_issues=0,
                          stage=draft["stage"], target_root=draft["target_root"],
                          export_root=draft["export_root"], source_sha256=draft["source_sha256"],
                          reviewed_drafts={"draft.json": sha(root / "draft.json")})
            (root / "review.json").write_text(json.dumps(review))
            plan = dict(draft, launch_allowed=True, source_review_sha256=sha(root / "review.json"))
            review_sources(root, plan)
            for key, value in (("seed", 2028), ("split", "test"),
                               ("online_allowed", True), ("assigned_gpu", "0"),
                               ("target_root", "/elsewhere"), ("task", "substituted"),
                               ("lr", .001)):
                with self.assertRaises(AssertionError, msg=key):
                    review_sources(root, dict(plan, **{key: value}))
            source.write_text("different\n")
            with self.assertRaises(AssertionError):
                review_sources(root, plan)


if __name__ == "__main__":
    unittest.main()
