"""Reject misaligned or malformed evidence rather than joining different states."""
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1] / "tools")]
from current_e0_supplement_contract import roi_coverage, verify_arrays, verify_native, verify_run


class SupplementTest(unittest.TestCase):
    def setUp(self):
        self.sequence = dict(name="train_sequence", rgb=["image"] * 100,
                             valid_frames=list(range(100)))
        predictions = np.tile(np.asarray([20., 30., 16., 12.], np.float32), (100, 1))
        self.original = dict(frame=np.arange(1, 100, 25, dtype=np.int64),
                             predictions=predictions)
        centre = predictions[0, :2] + predictions[0, 2:] / 2
        side = math.sqrt(float(predictions[0, 2] * predictions[0, 3])) * 4
        geometry = np.asarray([*(centre - side / 2), side], np.float32)
        box = predictions[0]
        left, top, side = geometry
        normalized = np.concatenate([
            (box[:2] + box[2:] / 2 - [left, top]) / side,
            box[2:] / side]).astype(np.float32)
        self.values = dict(frame=np.asarray([1, 26], np.int64),
                           e0_features=np.ones((2, 2, 768), np.float32),
                           e0_coverage=np.ones((2, 2), np.float32),
                           normalized_box=np.tile(normalized, (2, 1)),
                           geometry=np.tile(geometry, (2, 1)),
                           current_box=predictions[[1, 26]].copy(),
                           previous_box=predictions[[0, 25]].copy(),
                           observation=np.ones((2, 914), np.float32),
                           route=np.asarray([[0., 1., .8], [1., 0., .6]], np.float64),
                           predictions=predictions[:31].copy(),
                           frame_seconds=np.ones(31, np.float64))

    def test_aligned_smoke_and_explicit_missing_coverage(self):
        outside = np.asarray([200., 200., 16., 12.], np.float32)
        self.original["predictions"][26] = outside
        self.values["predictions"][26] = outside
        self.values["current_box"][1] = outside
        left, top, side = self.values["geometry"][1]
        self.values["normalized_box"][1] = np.concatenate([
            (outside[:2] + outside[2:] / 2 - [left, top]) / side,
            outside[2:] / side]).astype(np.float32)
        self.values["e0_coverage"][1] = 0
        self.values["e0_features"][1] = 0
        result = verify_arrays(self.values, self.sequence, self.original)
        self.assertEqual(result, dict(sequence="train_sequence", frames=31,
                                      candidate_sets=2, no_fully_covered_modality=1))

    def test_corruption_rejected(self):
        changes = [
            ("frame", lambda a: a.__setitem__(0, 2)),
            ("predictions", lambda a: a.__setitem__((5, 0), 21)),
            ("previous_box", lambda a: a.__setitem__((1, 0), 21)),
            ("current_box", lambda a: a.__setitem__((0, 0), 21)),
            ("geometry", lambda a: a.__setitem__((1, 0), 100)),
            ("normalized_box", lambda a: a.__setitem__((0, 0), .9)),
            ("e0_coverage", lambda a: a.__setitem__((1, 0), 1.1)),
            ("e0_coverage", lambda a: a.__setitem__((1, 0), 0)),
            ("e0_coverage", lambda a: a.__setitem__((1, 0), .123)),
            ("e0_features", lambda a: a.__setitem__((0, 0, 0), np.nan)),
            ("route", lambda a: a.__setitem__((0, 1), 5)),
            ("frame_seconds", lambda a: a.__setitem__(0, 0)),
        ]
        for key, change in changes:
            with self.subTest(key=key):
                values = copy.deepcopy(self.values)
                change(values[key])
                with self.assertRaises(AssertionError):
                    verify_arrays(values, self.sequence, self.original)
        values = copy.deepcopy(self.values)
        values["observation"] = values["observation"].astype(np.float64)
        with self.assertRaises(AssertionError):
            verify_arrays(values, self.sequence, self.original)
        values = copy.deepcopy(self.values)
        values["e0_features"] = values["e0_features"][:, :, :-1]
        with self.assertRaises(AssertionError):
            verify_arrays(values, self.sequence, self.original)

    def test_irregular_valid_frames_do_not_join_by_position(self):
        sequence = copy.deepcopy(self.sequence)
        sequence["valid_frames"] = [0, 2, 4, 6, 10]
        with self.assertRaises(AssertionError):
            verify_arrays(self.values, sequence, self.original)

    def test_zero_coverage_requires_zero_descriptor(self):
        self.test_aligned_smoke_and_explicit_missing_coverage()
        self.values["e0_features"][1, 0, 0] = 1
        with self.assertRaises(AssertionError):
            verify_arrays(self.values, self.sequence, self.original)

    def test_coverage_matches_actual_roi_pooling(self):
        import torch
        from dual_rgbt.e0_observation import roi_descriptor
        torch.set_num_threads(1)
        rng = np.random.RandomState(2027)
        boxes = rng.uniform(-1, 2, (100, 4)).astype(np.float32)
        boxes[:, 2:] = np.abs(boxes[:, 2:]) + .001
        boxes = np.concatenate([boxes, np.asarray([
            [0., 0., 1., 1.], [1., 1., 1., 1.], [.5, .5, 3., 3.],
            [0., .5, 0., .2], [1., .5, 0., .2]], np.float32)])
        for box in boxes:
            _, covered = roi_descriptor(torch.ones(1, 256, 768),
                                        torch.from_numpy(box[None]))
            self.assertEqual(roi_coverage(box), covered.item())

    def test_native_task_host_command_gpu_and_state_are_bound(self):
        root = Path("/data2/gb/Dual-RGBT")
        plan = dict(task="smoke_r2", assigned_gpu="1", target_hostname="actual26")
        plan_path = "reports/smoke_r2/plan.json"
        command = [str(root / "conda-envs/dual_rgbt/bin/python"),
                   str(root / "scripts/collect_e0_current_supplement.py"),
                   "--plan", plan_path]
        native = dict(task="smoke_r2", hostname="actual26", physical_gpu="1",
                      command=command, state="RUNNING")
        with patch("current_e0_supplement_contract.socket.gethostname", return_value="actual26"):
            self.assertEqual(verify_native(root, plan, native, plan_path, "RUNNING"), command)
            for key, wrong in (("task", "other"), ("hostname", "otherhost"),
                               ("physical_gpu", "0"), ("state", "COMPLETE"),
                               ("command", command[:-1] + ["otherplan"])):
                with self.subTest(key=key), self.assertRaises(AssertionError):
                    verify_native(root, plan, dict(native, **{key: wrong}), plan_path, "RUNNING")

    def test_wrong_run_directory_rejected_before_admission(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "wrong_run"
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps(dict(plan=dict(run="expected_run"))))
            (directory / "COMPLETE.json").write_text("{}")
            with patch("current_e0_supplement_contract.admission",
                       side_effect=RuntimeError("Directory check was bypassed")):
                with self.assertRaises(AssertionError):
                    verify_run(root, directory)


if __name__ == "__main__":
    unittest.main()
