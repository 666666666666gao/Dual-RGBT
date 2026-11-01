"""Admission must reject TEST and nonreviewed TRAIN witness plans."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_witness",
                                             ROOT / "scripts/check_e0_m1_m2_train.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AdmissionTests(unittest.TestCase):
    def check(self, change=None, alter_plan_only=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {"parent": MODULE.PARENT, "m1": MODULE.M1,
                      "m2": MODULE.M2, "factory": "factoryhash", "manifest": "manifesthash"}
            draft = dict(stage=MODULE.STAGE, split="train", phase="smoke",
                         launch_allowed=False, test_eligible=False, training_allowed=False,
                         exploratory_test=False, target_root=root.as_posix(),
                         target_host="2026", target_hostname="testhost", assigned_gpu="1",
                         source_sha256={"scripts/check_e0_m1_m2_train.py": "sourcehash"},
                         input_sha256=inputs, parent_checkpoint="parent",
                         m1_checkpoint="m1", m2_checkpoint="m2", factory_checkpoint="factory",
                         manifest="manifest", frames_per_sequence=31,
                         sequence_names=["2up", "motocross", "carturn"], reviewed_draft="draft")
            if change and not alter_plan_only:
                change(draft)
            plan = copy.deepcopy(draft)
            plan["launch_allowed"] = True
            if change and alter_plan_only:
                change(plan)
            review = dict(stage=MODULE.STAGE, status="SOURCE_PASS", blocking_issues=0,
                          target_root=root.as_posix(), source_sha256=draft["source_sha256"],
                          input_sha256=draft["input_sha256"],
                          reviewed_drafts={"draft": "drafthash"})
            for name, value in (("draft", draft), ("plan", plan), ("review", review)):
                (root / name).write_text(json.dumps(value))

            def digest(path):
                path = Path(path)
                if path == root / "review":
                    return "reviewhash"
                if path == root / "draft":
                    return "drafthash"
                if path.name == "check_e0_m1_m2_train.py":
                    return "sourcehash"
                return inputs[path.name]

            with patch.object(MODULE, "ROOT", root), patch.object(MODULE, "sha", digest):
                with patch.object(MODULE.socket, "gethostname", return_value="testhost"):
                    return MODULE.admission(root / "plan", root / "review", "reviewhash")

    def test_exact_reviewed_train_witness_is_admitted(self):
        self.assertTrue(self.check()["launch_allowed"])

    def test_test_split_rejected_even_if_reviewed(self):
        with self.assertRaises(AssertionError):
            self.check(lambda plan: plan.update(split="test"))

    def test_plan_change_after_review_rejected(self):
        with self.assertRaises(AssertionError):
            self.check(lambda plan: plan.update(task="changed"), alter_plan_only=True)

    def test_wrong_m1_checkpoint_digest_rejected(self):
        with self.assertRaises(AssertionError):
            self.check(lambda plan: plan["input_sha256"].update(m1="wrong"))


if __name__ == "__main__":
    unittest.main()
