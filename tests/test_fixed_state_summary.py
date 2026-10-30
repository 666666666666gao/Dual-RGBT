"""Analytic diagnostic fixtures; no tracker accuracy measurements."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_fixed_state_dev import summary


def frame(index, iou, sampled=False, write=False):
    metrics = dict(valid_gt=True, iou=iou, center_error=10, area_ratio=1)
    return dict(frame=index, action=0, sampled=sampled, primary=metrics, final=metrics,
                other_primary=dict(iou=.05), template_write_modalities=[0] if write else [])


class SummaryTests(unittest.TestCase):
    def test_low_stop_does_not_imply_recoverable(self):
        rows = [frame(1, .05), frame(2, .05, write=True), frame(3, .7)]
        result = summary(rows, [])
        self.assertEqual(result['sampled_recoverable_states'], 0)
        self.assertEqual(result['sampled_recoverable_stop_states'], 0)
        self.assertEqual(result['observed_template_writes_when_lost'], 1)
        self.assertEqual(result['longest_lost_intervals'][0], dict(start=1, end=2, frames=2))

    def test_candidate_recoveries_count_states_once(self):
        row = frame(5, .05, sampled=True)
        queries = [dict(frame=5, selected=False, metrics=dict(valid_gt=True), recovery=True,
                        harm=False, action=1, gain=.6, utility_prediction=-.01) for _ in range(2)]
        result = summary([row], queries)
        self.assertEqual(result['sampled_recoverable_states'], 1)
        self.assertEqual(result['sampled_recoverable_stop_states'], 1)
        self.assertEqual(result['actions']['1']['recovery_queries'], 2)
        self.assertEqual(result['actions']['1']['recoveries_predicted_positive'], 0)

    def test_invalid_gt_excluded_from_lost_and_size_counts(self):
        row = frame(1, None, write=True)
        row['primary'] = dict(valid_gt=False)
        row['final'] = dict(valid_gt=False)
        result = summary([row], [])
        self.assertEqual(result['valid_gt_frames'], 0)
        self.assertEqual(result['center_within20_iou_below_half'], 0)
        self.assertEqual(result['observed_template_writes_when_lost'], 0)


if __name__ == '__main__':
    unittest.main()
