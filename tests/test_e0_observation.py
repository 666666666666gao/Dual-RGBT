"""Test ROI locality and unchanged predictor arguments/results on CPU."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_rgbt.e0_observation import E0Observation, roi_descriptor


class FakeE0:
    def __init__(self):
        self.calls = 0
        self.output = dict(action=0, modality=1,
            box=torch.tensor([[.5, .5, .25, .25]]),
            primary=dict(features=torch.ones(1, 256, 768),
                box=torch.tensor([[.4, .4, .2, .2]]), score=torch.full((1, 1, 16, 16), .5)),
            preview=dict(pools=[torch.ones(1, 32)] * 2, descriptors=[torch.ones(1, 32)] * 2))

    def predict(self, *args):
        self.calls += 1
        self.arguments = args
        return self.output


class E0ObservationTest(unittest.TestCase):
    def test_disabled_has_one_original_call_and_identical_return(self):
        model = FakeE0()
        observer = E0Observation(model)
        values = [object(), object(), torch.zeros(1, 8), .0, object()]
        observer.begin(False)
        self.assertIs(model.predict(*values), model.output)
        self.assertEqual(model.calls, 1)
        self.assertEqual(observer.calls, 1)
        self.assertIsNone(observer.record)
        for actual, expected in zip(model.arguments, values):
            self.assertIs(actual, expected)
        observer.close()
        self.assertEqual(model.predict.__func__, FakeE0.predict)

    def test_enabled_is_read_only_and_preserves_feature_snapshot(self):
        model = FakeE0()
        before = model.output['primary']['features'].clone()
        observer = E0Observation(model)
        observer.begin(True)
        self.assertIs(model.predict(None, None, torch.zeros(1, 8), 0.), model.output)
        self.assertEqual(model.calls, 1)
        self.assertTrue(torch.equal(before, model.output['primary']['features']))
        self.assertEqual(observer.record['features'].shape, (914,))
        self.assertEqual(observer.record['primary_roi_coverage'], 1.)
        frozen = observer.record['features'].copy()
        model.output['primary']['features'].zero_()
        np.testing.assert_array_equal(observer.record['features'], frozen)

    def test_roi_is_spatially_local_and_marks_outside_crop(self):
        grid = torch.arange(16).float()[None, :].expand(16, -1)
        features = grid.reshape(1, 256, 1).expand(-1, -1, 768)
        left, _ = roi_descriptor(features, torch.tensor([[.25, .5, .1, .1]]))
        right, _ = roi_descriptor(features, torch.tensor([[.75, .5, .1, .1]]))
        self.assertTrue((right > left).all())
        outside, coverage = roi_descriptor(features, torch.tensor([[2., 2., .1, .1]]))
        self.assertEqual(coverage.item(), 0.)
        self.assertEqual(outside.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
