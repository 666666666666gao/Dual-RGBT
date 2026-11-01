"""Verify diagnostic geometry against actual E0 crop conversion."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from dual_rgbt.data import crop_pair
from lib.utils.box_ops import clip_box


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("diagnostic",
                                             ROOT / "scripts/diagnose_e0_m1_m2_train.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GeometryTests(unittest.TestCase):
    def test_original_box_matches_tracker_conversion(self):
        image = np.zeros((240, 320, 6), np.uint8)
        for previous in ([30., 50., 40., 60.], [-8., 5., 12., 20.], [250., 190., 60., 40.]):
            previous = np.asarray(previous, np.float32)
            _, _, geometry = crop_pair(image, previous, 4., 256)
            for normalized in ([.5, .5, .2, .3], [1.2, -.3, .6, .4]):
                normalized = np.asarray(normalized, np.float32)
                left, top, side = geometry
                xywh = np.concatenate([
                    normalized[:2] * side + [left, top] - normalized[2:] * side / 2,
                    normalized[2:] * side])
                expected = np.asarray(clip_box(xywh.tolist(), 240, 320, margin=10), np.float32)
                actual = MODULE.original_e0_box(previous, normalized, 240, 320)
                np.testing.assert_array_equal(actual, expected)

    def test_iou_values(self):
        self.assertEqual(MODULE.overlap([0, 0, 10, 10], [0, 0, 10, 10]), 1.)
        self.assertEqual(MODULE.overlap([0, 0, 10, 10], [20, 20, 10, 10]), 0.)
        self.assertAlmostEqual(MODULE.overlap([0, 0, 10, 10], [5, 0, 10, 10]), 1. / 3)


if __name__ == "__main__":
    unittest.main()
