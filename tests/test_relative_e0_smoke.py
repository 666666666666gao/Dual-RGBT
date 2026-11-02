"""Check smoke prefix joins and exact review/native execution bindings."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from relative_e0_smoke_contract import (
    STAGE, admission, load_prefix_data, native_command, sha, verify_native)
from replay_m2_train_states import collected_iou


class RelativeSmokeTest(unittest.TestCase):
    def test_native_requires_exact_command_host_gpu_and_task(self):
        root = Path("/data2/gb/Dual-RGBT")
        plan = dict(task="smoke", assigned_gpu="0", target_hostname=socket.gethostname())
        native = dict(state="RUNNING", task="smoke", physical_gpu="0",
                      hostname=socket.gethostname(), command=native_command(root, "plan.json"))
        verify_native(root, plan, native, "plan.json", "RUNNING")
        for key, value in (("task", "other"), ("physical_gpu", "1"), ("hostname", "other"),
                           ("command", ["python", "script.py"]), ("state", "FAILED")):
            with self.assertRaises(AssertionError):
                verify_native(root, plan, dict(native, **{key: value}), "plan.json", "RUNNING")

    def test_review_cannot_change_recipe_or_impersonate_other_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            draft = dict(stage=STAGE, launch_allowed=False, source_review="review.json",
                         source_review_sha256=None, reviewed_draft="draft.json",
                         source_sha256={"source.py": "fixed"}, lr=.0003)
            (root / "draft.json").write_text(json.dumps(draft))
            review = dict(stage=STAGE, status="SOURCE_PASS", blocking_issues=0,
                          target_host="2026", target_root=str(root),
                          source_sha256=draft["source_sha256"],
                          reviewed_drafts={"draft.json": sha(root / "draft.json")})
            (root / "review.json").write_text(json.dumps(review))
            plan = dict(draft, launch_allowed=True, source_review_sha256=sha(root / "review.json"))
            with patch("relative_e0_smoke_contract.inputs", return_value={"verified": True}):
                self.assertEqual(admission(root, plan), {"verified": True})
                with self.assertRaises(AssertionError):
                    admission(root, dict(plan, lr=.001))
                review["stage"] = "M2_relative_current_E0_train_engine"
                (root / "review.json").write_text(json.dumps(review))
                plan["source_review_sha256"] = sha(root / "review.json")
                with self.assertRaises(AssertionError):
                    admission(root, plan)

    def test_actual_six_prefix_joins_reuse_original_candidates_and_float64_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "current").mkdir()
            manifest = {"train": []}
            smoke = dict(manifest="manifest.json", sequence_names=["a", "b", "c"],
                         run="current", cache_sha256={name + ".npz": "bound elsewhere"
                                                     for name in ("a", "b", "c")})
            for name in smoke["sequence_names"]:
                frames = np.array([1, 26, 51], np.int64)
                predictions = np.tile(np.array([1., 2., 4., 5.], np.float32), (60, 1))
                gt = predictions.astype(np.float64)
                boxes = np.repeat(predictions[frames, None], 24, axis=1)
                labels = np.ones((3, 25), np.int8)
                labels[:, 2] = -1
                np.savez(root / (name + ".npz"), frame=frames, predictions=predictions,
                         anchor=np.ones((2, 768), np.float32),
                         features=np.ones((3, 24, 2, 768), np.float32),
                         coverage=np.ones((3, 24, 2), np.float32),
                         labels=labels, box=boxes, iou=collected_iou(boxes, gt[frames]))
                np.savez(root / "current" / (name + ".npz"), frame=frames[:2],
                         predictions=predictions[:31], e0_features=np.ones((2, 2, 768), np.float32),
                         e0_coverage=np.ones((2, 2), np.float32))
                manifest["train"].append(dict(name=name, boxes=gt.tolist()))
            (root / "manifest.json").write_text(json.dumps(manifest))
            data = load_prefix_data(root, smoke)
            self.assertEqual(data["candidates"].shape, (6, 24, 2, 768))
            self.assertEqual(data["sequence_index"].tolist(), [0, 0, 1, 1, 2, 2])
            self.assertTrue((data["current_iou"].numpy() == 1).all())
            self.assertEqual(data["candidate_iou"].numpy().dtype, np.float64)
            self.assertEqual(data["labels"][:, 2].tolist(), [-1] * 6)
            with np.load(root / "current/a.npz") as arrays:
                changed = {key: value.copy() for key, value in arrays.items()}
            changed["predictions"][2, 0] += 1
            np.savez(root / "current/a.npz", **changed)
            with self.assertRaises(AssertionError):
                load_prefix_data(root, smoke)


if __name__ == "__main__":
    unittest.main()
