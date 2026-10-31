"""Check analytic shape accounting without importing a GPU framework."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('full_test_macs', ROOT / 'tools/estimate_full_test_macs.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullTestMacsTest(unittest.TestCase):
    def setUp(self):
        self.measured = json.loads(
            (ROOT / 'reports/TRACE_WEIGHTED_MACS_RESULT_20261030.json').read_bytes())['measured_cases']

    def test_actual_head_dimensions(self):
        self.assertEqual(MODULE.query_macs(
            (ROOT / 'dual_rgbt_evidence/policy.py').read_text()), 61184)

    def test_all_measured_cases_are_exact(self):
        mixture = {(r['modality'], r['action'], r['candidate_count']): r['frames']
                   for r in self.measured}
        actual = MODULE.shape_estimates(self.measured, mixture, 61184)
        self.assertEqual(len(actual), 23)
        expected = {(r['modality'], r['action'], r['candidate_count']): r['conv_matrix_macs']
                    for r in self.measured}
        for row in actual:
            self.assertEqual(row['provenance'], 'existing_profiled_shape')
            self.assertEqual(row['conv_matrix_macs'],
                             expected[(row['modality'], row['action'], row['candidate_count'])])

    def test_unmeasured_shape_is_not_labeled_measured(self):
        rows = MODULE.shape_estimates(self.measured, {(0, 4, 3): 7, (1, 0, 8): 9}, 61184)
        self.assertEqual(rows[0]['provenance'], 'analytic_shape_estimate')
        self.assertEqual(rows[0]['frames'], 7)
        self.assertEqual(rows[0]['conv_matrix_macs'], 80145924992 - 5 * 122368)
        self.assertEqual(rows[1]['provenance'], 'existing_profiled_shape')

    def test_inconsistent_measurement_rejects_formula(self):
        measured = copy.deepcopy(self.measured)
        measured[0]['conv_matrix_macs'] += 1
        with self.assertRaises(AssertionError):
            MODULE.shape_estimates(measured, {(0, 0, 8): 1}, 61184)

    def test_changed_head_rejected(self):
        source = 'self.query_gain = nn.Sequential(nn.Linear(956, 64), nn.GELU(), nn.Linear(64, 1))'
        with self.assertRaises(AssertionError):
            MODULE.query_macs(source)


if __name__ == '__main__':
    unittest.main()
