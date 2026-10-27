"""Analytic offline-diagnostic fixtures, not model accuracy experiments."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

source = Path(__file__).resolve().parents[1] / 'tools/analyze_lasher_query_benefit.py'
spec = importlib.util.spec_from_file_location('benefit', source)
benefit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benefit)


class QueryBenefitTests(unittest.TestCase):
    def test_continuous_geometry(self):
        gt = np.asarray([[0, 0, 10, 10]] * 3, dtype=float)
        boxes = np.asarray([[0, 0, 10, 10], [5, 0, 10, 10], [20, 0, 10, 10]], dtype=float)
        iou, ce, nce = benefit.localization(boxes, gt)
        np.testing.assert_allclose(iou, [1, 1 / 3, 0])
        np.testing.assert_allclose(ce, [0, 5, 20])
        np.testing.assert_allclose(nce, [0, .5, 2])

    def fixture(self, root):
        run = root / 'runs/fixture'
        for folder in ('predictions', 'groundtruth', 'traces', 'evidence'):
            (run / folder).mkdir(parents=True)
        names = ['synthetic_%03d' % i for i in range(97)]
        (run / 'config.json').write_text(json.dumps(dict(sequences=names,
            evidence_schema='primary_executed_box_and_selected_gain_v1',
            checkpoint_sha256='synthetic-no-model', cost_weight=.05)))
        for name in names:
            np.savetxt(run / 'predictions' / (name + '.txt'),
                       [[0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 10, 10], [0, 0, 10, 10]], delimiter=',')
            gt = [[0, 0, 10, 10]] * 4
            if name == names[0]:
                gt[2] = [0, 0, 0, 0]
            np.savetxt(run / 'groundtruth' / (name + '.txt'), gt, delimiter=',')
            with (run / 'traces' / (name + '.csv')).open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['frame', 'action', 'modality'])
                writer.writerows([[1, 1, 0], [2, 2, 1], [3, 0, 0]])
            with (run / 'evidence' / (name + '.csv')).open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['frame', 'primary_x', 'primary_y', 'primary_w', 'primary_h',
                                 'selected_gain', 'selected_token_cost', 'selected_utility'])
                writer.writerows([[1, 20, 0, 10, 10, .5, 1, .45],
                                  [2, 0, 0, 10, 10, .5, 1, .45],
                                  [3, 0, 0, 10, 10, 0, 0, 0]])
        report = root / 'report.json'
        report.write_text(json.dumps(dict(status='SCORED_COMPLETE', benchmark='LasHeR', split='dev',
            sequences=97, frames=388, cost_weight=.05, checkpoint_sha256='synthetic-no-model',
            run_names=['fixture'], run_artifacts_sha256={p.relative_to(root).as_posix(): benefit.sha(p) for p in run.rglob('*') if p.is_file()})))
        return report

    def test_complete_fixture_and_invalid_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, rows = benefit.analyze(root, self.fixture(root))
            summary = result['all_frames']
            self.assertEqual(summary['frames'], 291)
            self.assertEqual(summary['invalid_gt_frames'], 1)
            self.assertEqual(summary['helpful_queries'], 97)
            self.assertEqual(summary['harmful_queries'], 96)
            self.assertEqual(summary['stop_frames'], 97)
            self.assertAlmostEqual(summary['mean_iou_gain'], 1 / 193)
            self.assertEqual(summary['negative_realized_utility_queries'], 96)
            self.assertEqual(result['by_action']['0']['query_frames'], 0)
            self.assertNotIn('iou_gain', rows[1])

    def test_changed_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.fixture(root)
            (root / 'runs/fixture/predictions/synthetic_000.txt').write_text('0,0,1,1\n')
            with self.assertRaises(AssertionError):
                benefit.analyze(root, report)

    def test_partial_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.fixture(root)
            data = json.loads(report.read_bytes())
            data['sequences'] = 20
            report.write_text(json.dumps(data))
            with self.assertRaises(AssertionError):
                benefit.analyze(root, report)


if __name__ == '__main__':
    unittest.main()
