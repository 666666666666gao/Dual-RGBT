import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
from unittest.mock import patch

source = Path(__file__).resolve().parents[1] / 'tools/calibrate_query_acceptance.py'
spec = importlib.util.spec_from_file_location('calibrate', source)
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)


class CalibrationTests(unittest.TestCase):
    def test_saved_same_frame_primary_boxes_drive_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'runs' / 'fixture'
            artifacts = {}

            def save(relative, text):
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding='utf-8')
                artifacts[path.relative_to(root).as_posix()] = calibrate.sha(path)

            names = ['sequence%d' % i for i in range(97)]
            save('config.json', json.dumps(dict(evidence_schema='primary_executed_box_and_selected_gain_v1', sequences=names)))
            for name in names:
                save('predictions/' + name + '.txt', '100,100,10,10\n0,0,10,10\n100,100,10,10\n0,0,10,10\n')
                save('groundtruth/' + name + '.txt', '0,0,10,10\n' * 4)
                save('traces/' + name + '.csv', 'frame,modality,action,confidence\n1,1,0,0.5\n2,1,4,0.2\n3,1,4,0.9\n')
                save('evidence/' + name + '.csv', 'frame,primary_x,primary_y,primary_w,primary_h,primary_confidence\n1,0,0,10,10,0.8\n2,0,0,10,10,0.8\n3,100,100,10,10,0.8\n')
            report = root / 'report.json'
            report.write_text(json.dumps(dict(status='SCORED_COMPLETE', benchmark='LasHeR', split='dev', sequences=97,
                                             run_names=['fixture'], run_artifacts_sha256=artifacts)))
            with patch.object(calibrate, 'ROOT', root):
                _, rows, _ = calibrate.rows_from_report(report)
            self.assertEqual([row['primary_iou'] for row in rows[:3]], [1.0, 1.0, 0.0])
            thresholds, reports = calibrate.fit(rows)
            self.assertEqual(thresholds['tir_wide'], .9)
            self.assertEqual(reports['tir_wide']['immediate_mean_iou'], 1.0)
            self.assertEqual(reports['tir_wide']['accepted_query_frames'], 97)

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
