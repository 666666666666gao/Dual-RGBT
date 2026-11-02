"""Check current-proposal geometry and read-only dual-modal observation."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_rgbt.data import crop_pair
from dual_rgbt.e0_current_evidence import current_proposal_evidence


class EncoderOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def encode(self, templates, search):
        self.calls.append((templates.clone(), search.clone()))
        return search.mean().expand(1, 256, 768), 385


class CurrentEvidenceTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = EncoderOnly().eval()
        self.templates = torch.arange(6.).view(1, 1, 6, 1, 1).expand(
            1, 2, 6, 128, 128).clone()

    def test_previous_crop_geometry_and_modality_order_preserve_inputs(self):
        image = np.full((96, 128, 6), 127, np.uint8)
        previous = np.asarray([40., 30., 20., 10.], np.float32)
        current = np.asarray([47., 26., 16., 12.], np.float32)
        _, _, geometry = crop_pair(image, previous, 4., 256)
        search = torch.cat([torch.ones(1, 3, 256, 256),
                            torch.full((1, 3, 256, 256), 2.)], 1)
        templates_before, search_before = self.templates.clone(), search.clone()
        current_before, geometry_before = current.copy(), geometry.copy()
        output = current_proposal_evidence(
            self.model, self.templates, search, geometry, current)
        expected = np.concatenate([
            (current[:2] + current[2:] / 2 - geometry[:2].tolist()) / geometry[2],
            current[2:] / geometry[2]]).astype(np.float32)
        np.testing.assert_array_equal(output["normalized_box"], expected)
        self.assertEqual(len(self.model.calls), 2)
        for modality, (templates, channels) in enumerate(self.model.calls):
            self.assertTrue(torch.equal(templates, self.templates[:, :, modality * 3:modality * 3 + 3]))
            self.assertEqual(channels.mean().item(), modality + 1)
        self.assertEqual(output["features"].shape, (2, 768))
        self.assertTrue(torch.allclose(output["features"][0], torch.ones(768)))
        self.assertTrue(torch.allclose(output["features"][1], torch.full((768,), 2.)))
        self.assertEqual(output["coverage"].tolist(), [1., 1.])
        self.assertFalse(output["features"].requires_grad)
        self.assertTrue(torch.equal(self.templates, templates_before))
        self.assertTrue(torch.equal(search, search_before))
        np.testing.assert_array_equal(current, current_before)
        np.testing.assert_array_equal(geometry, geometry_before)

    def test_uncovered_current_box_remains_an_observation(self):
        output = current_proposal_evidence(
            self.model, self.templates, torch.ones(1, 6, 256, 256),
            np.asarray([0., 0., 100.], np.float32),
            np.asarray([200., 200., 10., 10.], np.float32))
        self.assertEqual(output["coverage"].tolist(), [0., 0.])
        self.assertEqual(output["features"].abs().sum().item(), 0.)
        self.assertEqual(len(self.model.calls), 2)
        self.assertEqual(set(output), {"features", "coverage", "normalized_box"})


if __name__ == "__main__":
    unittest.main()
