import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from diagnose_tracking_failures import failure_episodes, summarize


def row(sequence, frame, primary=.05, final=.05, action=0, confidence=.8, write=0, valid=True):
    return dict(sequence=sequence, frame=frame, primary_iou=primary, final_iou=final,
        action=action, confidence=confidence, template_updated=write, valid_gt=valid)


class DiagnosticTests(unittest.TestCase):
    def test_failure_episodes_do_not_cross_sequence_invalid_gt_or_frame_gaps(self):
        rows = [row('a', 1), row('a', 2, write=1), row('a', 3, valid=False),
                row('a', 4), row('a', 6), row('b', 1), row('b', 2, final=.7)]
        episodes = failure_episodes(rows)
        self.assertEqual([e['length'] for e in episodes], [2, 1, 1, 1])
        self.assertEqual(episodes[0]['template_writes'], 1)

    def test_unqueried_failure_is_separate_from_executed_query_recovery(self):
        rows = [row('a', 1, write=1), row('a', 2, final=.7, action=3),
                row('a', 3, primary=.8, action=4, write=1), row('a', 4, valid=False)]
        result = summarize(rows)
        self.assertEqual(result['valid_frames'], 3)
        self.assertEqual(result['primary_lost_stop_frames'], 1)
        self.assertEqual(result['immediate_query_recoveries'], 1)
        self.assertEqual(result['immediate_query_failures'], 1)
        self.assertEqual(result['lost_template_writes'], 2)
        self.assertEqual(result['bad_template_writes_after_query'], 1)


if __name__ == '__main__':
    unittest.main()
