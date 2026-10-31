"""Check diagnostic category boundaries and run-relative artifact binding."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_lasher_test_trajectories import geometry_summary, read_checked
from diagnose_tracking_failures import failure_episodes


class FullTestTrajectoryAnalysisTest(unittest.TestCase):
    def test_geometry_partitions_low_overlap(self):
        result = geometry_summary(
            np.asarray([.49, .49, .5, .1]), np.asarray([20., 20.1, 2., 0.]),
            np.asarray([.49, 2.1, 3., 2.]))
        self.assertEqual(result['low_overlap_frames'], 3)
        self.assertEqual(result['low_overlap_center_within20_frames'], 2)
        self.assertEqual(result['low_overlap_center_beyond20_frames'], 1)
        self.assertEqual(result['low_overlap_center_within20_area_mismatch_frames'], 1)

    def test_run_relative_hash_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / 'runs' / 'candidate'
            run.mkdir(parents=True)
            path = run / 'config.json'
            path.write_bytes(b'{}')
            artifacts = {'config.json': hashlib.sha256(b'{}').hexdigest()}
            used = {}
            self.assertEqual(read_checked(root, run, artifacts, 'config.json', used), path)
            self.assertEqual(set(used), {'runs/candidate/config.json'})
            path.write_bytes(b'{"changed":true}')
            with self.assertRaises(AssertionError):
                read_checked(root, run, artifacts, 'config.json', used)

    def test_invalid_gt_and_sequence_boundaries_break_failure_episode(self):
        rows = [dict(sequence=s, frame=f, valid_gt=v, final_iou=0.,
                     action=0, confidence=.9, template_updated=0)
                for s, f, v in [('a', 1, True), ('a', 2, False), ('a', 3, True),
                                ('b', 1, True), ('b', 2, True)]]
        self.assertEqual([e['length'] for e in failure_episodes(rows)], [1, 1, 2])


if __name__ == '__main__':
    unittest.main()
