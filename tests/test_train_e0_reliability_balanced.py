"""Check fit-only weighting and exact historical-source provenance."""
import importlib.util
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "balanced", ROOT / "scripts/train_e0_reliability_balanced.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BalancedTests(unittest.TestCase):
    def test_inverse_weights_equalize_class_mass(self):
        iou = torch.tensor([.9] * 6 + [.3] * 3 + [.01])
        weights = MODULE.class_weights(iou)
        mass = weights * torch.tensor([6, 3, 1])
        torch.testing.assert_close(mass, torch.full((3,), 10 / 3))
        logits = torch.zeros(10, 3, requires_grad=True)
        loss = F.cross_entropy(logits, MODULE.localization_targets(iou), weight=weights)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[-1].abs().sum().item(), logits.grad[0].abs().sum().item())

    def test_missing_fit_class_is_rejected(self):
        with self.assertRaises(AssertionError):
            MODULE.class_weights(torch.tensor([.9, .01]))

    def test_historical_wrapper_is_exact_archived_git_bytes(self):
        self.assertEqual(MODULE.sha(ROOT / MODULE.SNAPSHOT), MODULE.OLD_WRAPPER)
        current = (ROOT / MODULE.HISTORICAL).read_text()
        historical = (ROOT / MODULE.SNAPSHOT).read_text()
        self.assertIn("OPENCV_FOR_THREADS_NUM", current)
        self.assertNotIn("OPENCV_FOR_THREADS_NUM", historical)
        self.assertIn("hostname", current)
        self.assertNotIn("hostname", historical)


if __name__ == "__main__":
    unittest.main()
