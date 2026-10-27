import importlib.util
from pathlib import Path
import unittest

source = Path(__file__).resolve().parents[1] / 'tools/calibrate_query_acceptance.py'
spec = importlib.util.spec_from_file_location('calibrate', source)
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)


class CalibrationTests(unittest.TestCase):
    def test_empty_routes_are_explicit(self):
        thresholds, reports = calibrate.fit([
            dict(route='rgb_local128', action=3, confidence=.4, primary_iou=.2, query_iou=.8),
            dict(route='rgb_local128', action=3, confidence=.9, primary_iou=.2, query_iou=.3),
        ])
        self.assertEqual(thresholds['rgb_local128'], .0)
        self.assertEqual(reports['rgb_local128']['accepted_query_frames'], 2)
        self.assertEqual(reports['tir_wide']['status'], 'NO_QUERY_ROWS')

    def test_route_fit_can_reject_harmful_query(self):
        thresholds, reports = calibrate.fit([
            dict(route='tir_wide', action=4, confidence=.2, primary_iou=.6, query_iou=.1),
            dict(route='tir_wide', action=4, confidence=.9, primary_iou=.2, query_iou=.8),
        ])
        self.assertEqual(thresholds['tir_wide'], .9)
        self.assertEqual(reports['tir_wide']['accepted_query_frames'], 1)
        self.assertEqual(reports['tir_wide']['immediate_mean_iou'], .7)


if __name__ == '__main__':
    unittest.main()
