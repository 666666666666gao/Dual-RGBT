"""Reject misaligned or malformed evidence rather than joining different states."""
import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from current_e0_supplement_contract import verify_arrays


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


if __name__ == "__main__":
    unittest.main()
