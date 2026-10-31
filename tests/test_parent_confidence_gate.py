"""Verify confidence-gate fitting, strict ties, ranking and partition isolation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from diagnose_parent_confidence_gate import diagnose, fit_threshold, lost_vs_good_auc, operating_point


def row(confidence, iou, partition='fit', sequence='a'):
    return dict(confidence=confidence, final_iou=iou, partition=partition, sequence=sequence)


class ParentConfidenceGateTest(unittest.TestCase):
    def test_auc_perfect_reversed_and_tied_scores(self):
        self.assertEqual(lost_vs_good_auc([row(.1, 0), row(.9, 1)])['auc'], 1)
        self.assertEqual(lost_vs_good_auc([row(.9, 0), row(.1, 1)])['auc'], 0)
        self.assertEqual(lost_vs_good_auc([row(.9, 0), row(.9, 1)])['auc'], .5)

    def test_auc_excludes_uncertain_overlap_and_averages_ties(self):
        rows = [row(.1, 0), row(.4, 0), row(.4, 1), row(.9, 1), row(0., .2)]
        result = lost_vs_good_auc(rows)
        self.assertEqual(result['auc'], .875)
        self.assertEqual(result['excluded_uncertain_states'], 1)

    def test_strict_gate_preserves_threshold_ties(self):
        rows = [row(.4, 0), row(.65, 0), row(.65, .5), row(.3, .5)]
        point = operating_point(rows, .65)
        self.assertEqual(point['lost_detected'], 1)
        self.assertEqual(point['lost_missed'], 1)
        self.assertEqual(point['good_interrupted'], 1)
        self.assertEqual(point['good_retained'], 1)

    def test_fit_threshold_preserves_at_least_95_percent_good_states(self):
        rows = [row(i / 100, 1) for i in range(100)] + [row(.9, 0)]
        threshold = fit_threshold(rows)
        self.assertEqual(threshold, .05)
        self.assertEqual(operating_point(rows, threshold)['good_retained_fraction'], .95)
        tied = [row(.7, 1) for _ in range(100)] + [row(.9, 0)]
        self.assertEqual(operating_point(tied, fit_threshold(tied))['good_retained_fraction'], 1)

    def test_holdout_changes_do_not_change_fit_threshold(self):
        fit = [row(i / 100, 1) for i in range(100)] + [row(.9, 0)]
        held = [row(.1, 0, 'holdout', 'b'), row(.8, 1, 'holdout', 'b')]
        expected = diagnose(fit + held)['threshold_selection']['threshold']
        changed = [row(.99, 0, 'holdout', 'b'), row(.01, 1, 'holdout', 'b')]
        self.assertEqual(diagnose(fit + changed)['threshold_selection']['threshold'], expected)


if __name__ == '__main__':
    unittest.main()
