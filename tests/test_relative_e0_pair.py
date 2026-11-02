"""Reject recipe/test promotion and bind only exact qualifying TRAIN evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from relative_e0_pair_contract import (
    BINDINGS, bind_draft, native_command, qualifying_training, recipe)


class PairContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.draft = json.loads((
            ROOT / "reports/e0_relative_pair_train_r1_20261102/draft.json").read_bytes())

    def bound(self):
        plan = copy.deepcopy(self.draft)
        for key in BINDINGS - {"launch_allowed", "qualifying_epoch"}:
            plan[key] = "a" * 64
        plan.update(launch_allowed=True, qualifying_epoch=5)
        return plan

    def test_only_evidence_fields_bind_and_recipe_cannot_change(self):
        bind_draft(self.bound(), self.draft)
        for key, value in (
                ("seed", 2028), ("cost_weight", .05), ("frames", 93),
                ("assigned_gpu", "2"), ("split", "test"), ("benchmark_allowed", True),
                ("sequence_names", list(reversed(self.draft["sequence_names"]))),
                ("relative_checkpoint", "runs/old/best.pth")):
            plan = self.bound()
            plan[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                bind_draft(plan, self.draft)

    def test_missing_or_malformed_training_binding_rejected(self):
        for key in BINDINGS - {"launch_allowed", "qualifying_epoch"}:
            for value in (None, "z" * 64, "a" * 63):
                plan = self.bound()
                plan[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                    bind_draft(plan, self.draft)
        for epoch in (None, 0, 13, True, 1.5):
            plan = self.bound()
            plan["qualifying_epoch"] = epoch
            with self.subTest(epoch=epoch), self.assertRaises(AssertionError):
                bind_draft(plan, self.draft)

    def test_disabled_draft_is_not_changed_by_binding(self):
        before = copy.deepcopy(self.draft)
        bind_draft(self.bound(), self.draft)
        self.assertEqual(self.draft, before)
        self.assertFalse(self.draft["launch_allowed"])
        recipe(self.draft)

    def test_native_command_uses_absolute_interpreter_and_plan(self):
        root = Path("/data2/gb/Dual-RGBT")
        relative = "reports/e0_relative_pair_train_r1_20261102/plan.json"
        self.assertEqual(native_command(root, relative), [
            str(root / "conda-envs/dual_rgbt/bin/python"),
            str(root / "scripts/check_e0_relative_pair_train.py"),
            "--plan", str(root / relative)])

    def test_rejected_stale_or_rebound_training_head_cannot_be_promoted(self):
        plan = self.bound()
        verified = dict(status="RELATIVE_CURRENT_E0_FULL_TRAINING_VERIFIED",
                        retain_protected_e0=False, best_qualifying_epoch=5,
                        artifacts_sha256={"best.pth": "a" * 64}, plan_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / plan["training_acceptance"]
            output.parent.mkdir(parents=True)
            output.write_text(json.dumps(verified))
            with patch("verify_relative_e0_training.verify_completed", return_value=verified):
                self.assertEqual(qualifying_training(root, plan), verified)
                for key, value in (
                        ("qualifying_epoch", 6), ("relative_checkpoint_sha256", "b" * 64),
                        ("training_plan_sha256", "b" * 64)):
                    changed = dict(plan, **{key: value})
                    with self.subTest(key=key), self.assertRaises(AssertionError):
                        qualifying_training(root, changed)
            rejected = dict(verified, retain_protected_e0=True, best_qualifying_epoch=None,
                            artifacts_sha256={})
            output.write_text(json.dumps(rejected))
            with patch("verify_relative_e0_training.verify_completed", return_value=rejected):
                with self.assertRaises(AssertionError):
                    qualifying_training(root, plan)
            with patch("verify_relative_e0_training.verify_completed", return_value=verified):
                with self.assertRaises(AssertionError):
                    qualifying_training(root, plan)


if __name__ == "__main__":
    unittest.main()
