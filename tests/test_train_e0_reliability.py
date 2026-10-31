"""CPU checks for M1 offline preprocessing and failure reporting."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_e0_reliability import fit_normalization, load_collection, metrics


class TrainingTest(unittest.TestCase):
    def test_fit_normalization_constant_channel(self):
        fit = torch.tensor([[1., 2.], [3., 2.]])
        mean, scale = fit_normalization(fit)
        self.assertEqual(mean.tolist(), [2., 2.])
        self.assertEqual(scale.tolist(), [1., 1.])
        self.assertTrue(torch.equal((fit - mean) / scale,
                                    torch.tensor([[-1., 0.], [1., 0.]])))
        held = torch.tensor([[101., 2.]])
        self.assertEqual(((held - mean) / scale).tolist(), [[99., 0.]])

    def test_metrics_reports_failure_and_false_interruption(self):
        logits = torch.tensor([[8., 0., 0.], [0., 8., 0.], [8., 0., 0.], [0., 0., 8.]])
        with self.assertRaises(AssertionError):
            metrics(torch.nn.Identity(), logits, torch.tensor([.9, .7, .01, .02]), 2)

    def test_complete_three_class_report(self):
        logits = torch.tensor([[8., 0., 0.], [0., 8., 0.], [8., 0., 0.],
                               [0., 0., 8.], [0., 8., 0.]])
        result = metrics(torch.nn.Identity(), logits, torch.tensor([.9, .7, .01, .02, .3]), 2)
        self.assertEqual(result['confusion_true_rows_predicted_columns'], [[1, 1, 0], [0, 1, 0], [1, 0, 1]])
        self.assertEqual(result['lost_recall'], .5)
        self.assertEqual(result['reliable_interrupted_fraction'], .5)

    def test_prefix_collection_rejected(self):
        with self.assertRaises(AssertionError):
            load_collection(Path('.'), dict(caches=[dict(run='prefix')]))


if __name__ == '__main__':
    unittest.main()
