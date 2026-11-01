"""CPU geometry, real-query accounting and supervision contract checks."""
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collect_e0_identity_train import (
    bound_smoke_plan_path, check_admission, execute_candidates, global_windows, normalized_cxcywh,
    overlap_labels, reviewed_draft_path, returned_box, sampled_frames, smoke_artifact_names)


class CandidateEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def primary(self, templates, search, modality):
        self.calls += 1
        return dict(box=torch.tensor([[.5, .5, .2, .2]]),
            features=torch.full((1, 256, 768), float(modality + 1)),
            score=torch.full((1, 1, 16, 16), .6 + .1 * modality))


class CandidateCollectionTest(unittest.TestCase):
    def test_smoke_plan_alias_must_match_bound_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'smoke-plan-r2.json').write_text('{"phase":"smoke"}')
            digest = hashlib.sha256((root / 'smoke-plan-r2.json').read_bytes()).hexdigest()
            native = {'command': ['python', 'collector.py', '--plan',
                'smoke-plan-r2.json', '--run', 'smoke']}
            self.assertEqual(bound_smoke_plan_path(root,
                {'smoke_plan_sha256': digest}, native), 'smoke-plan-r2.json')
            with self.assertRaises(AssertionError):
                bound_smoke_plan_path(root, {'smoke_plan_sha256': '0' * 64}, native)

    def test_phase_cannot_be_bound_to_the_other_reviewed_draft(self):
        plan = dict(mode='protected_e0_identity_train_collection', launch_allowed=True,
            training_allowed=False, split='train', test_eligible=False, phase='smoke',
            reviewed_draft='reports/E0_M2_IDENTITY_FULL_DRAFT_20261031.json')
        with self.assertRaises(AssertionError):
            check_admission(Path('.'), plan)

    def test_versioned_draft_path_matches_only_its_phase(self):
        self.assertEqual(reviewed_draft_path('smoke'),
            'reports/E0_M2_IDENTITY_SMOKE_DRAFT_R7_20261031.json')
        self.assertEqual(reviewed_draft_path('full'),
            'reports/E0_M2_IDENTITY_FULL_DRAFT_R7_20261031.json')
        with self.assertRaises(AssertionError):
            reviewed_draft_path('smoke ' + 'full')

    def test_smoke_receipt_artifact_set_excludes_self_receipt(self):
        expected = smoke_artifact_names(['2up', 'ab_pingpongball3', '1strow3rdboymid'])
        self.assertEqual(expected, {'config.json', 'progress.json', '2up.npz',
            'ab_pingpongball3.npz', '1strow3rdboymid.npz'})

    def test_valid_frame_stride_keeps_sequence_without_valid_multiple_of25(self):
        valid = list(range(25)) + list(range(28, 50))
        self.assertEqual(sampled_frames(dict(valid_frames=valid), 50), [1, 29])
        self.assertEqual(sampled_frames(dict(valid_frames=valid), 31), [1, 29])

    def test_coordinates_and_clipped_boxes(self):
        geometry = np.asarray([-20., 40., 160.], np.float32)
        boxes = np.asarray([[4., 60., 24., 32.], [70., 100., 20., 10.]], np.float32)
        for box, normalized in zip(boxes, normalized_cxcywh(boxes, geometry)):
            np.testing.assert_allclose(returned_box(normalized, geometry, 400, 600), box, atol=2e-5)
        np.testing.assert_array_equal(returned_box(
            np.asarray([0., 0., .25, .25]), [0., 0., 100.], 100, 100), [0., 0., 12.5, 12.5])

    def test_global_windows_cover_every_image_pixel(self):
        height, width = 120, 200
        windows = global_windows(height, width)
        self.assertEqual(len(windows), 9)
        yy, xx = np.mgrid[:height, :width]
        covered = np.zeros((height, width), bool)
        for box in windows:
            centre, side = box[:2] + box[2:] / 2, np.sqrt(box[2] * box[3]) * 4
            covered |= (xx >= centre[0] - side / 2) & (xx < centre[0] + side / 2) & (
                yy >= centre[1] - side / 2) & (yy < centre[1] + side / 2)
        self.assertTrue(covered.all())

    def test_ambiguous_unobserved_and_no_match_semantics(self):
        boxes = np.tile(np.asarray([100., 100., 10., 10.], np.float32), (24, 1))
        boxes[0], boxes[1], boxes[2] = [0, 0, 10, 10], [5, 0, 10, 10], [0, 0, 10, 10]
        coverage = np.ones((24, 2), np.float32)
        coverage[2] = .5
        iou, labels = overlap_labels(boxes, coverage, [0, 0, 10, 10])
        self.assertEqual(labels[:4].tolist(), [0, 1, -1, -1])
        self.assertEqual(labels[4:].tolist(), [0] * 21)
        self.assertEqual(iou[0], 1.)
        boxes[0] = [100, 100, 10, 10]
        _, labels = overlap_labels(boxes, coverage, [0, 0, 10, 10])
        self.assertEqual(labels[0], -1)
        boxes[1:3] = [100, 100, 10, 10]
        _, labels = overlap_labels(boxes, coverage, [0, 0, 10, 10])
        self.assertEqual(labels[0], 1)
        coverage[:] = 0
        _, labels = overlap_labels(boxes, coverage, [0, 0, 10, 10])
        self.assertEqual(labels.tolist(), [-1] * 25)

    def test_queries_execute_both_modalities_and_preserve_templates(self):
        torch.set_num_threads(1)
        model = CandidateEncoder().eval()
        templates = torch.zeros(1, 2, 6, 128, 128)
        before = templates.clone()
        image = np.zeros((120, 200, 6), np.uint8)
        result = execute_candidates(model, templates, image, np.asarray([40., 30., 20., 20.], np.float32))
        self.assertEqual(model.calls, 24)
        self.assertEqual(result['features'].shape, (24, 2, 768))
        self.assertEqual(result['coverage'].shape, (24, 2))
        self.assertEqual(result['geometry'].shape, (12, 3))
        self.assertEqual(result['refinement_parents'].tolist(), [3, 5])
        self.assertEqual(result['query'].tolist(), list(np.repeat(np.arange(12), 2)))
        self.assertTrue(torch.equal(templates, before))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
