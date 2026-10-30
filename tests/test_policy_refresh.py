"""Sampling invariants for TRAIN refresh, not a tracking accuracy test."""
import ast
from pathlib import Path
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/collect_policy_query_features.py'
TREE = ast.parse(SOURCE.read_text())
FUNCTION = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'choices')
NAMESPACE = {'np': np}
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), str(SOURCE), 'exec'), NAMESPACE)
AUDIT = Path(__file__).resolve().parents[1] / 'tools/audit_policy_refresh.py'
AUDIT_TREE = ast.parse(AUDIT.read_text())
COVERAGE = next(node for node in AUDIT_TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'verify_coverage')
exec(compile(ast.Module(body=[COVERAGE], type_ignores=[]), str(AUDIT), 'exec'), NAMESPACE)


class SamplingTests(unittest.TestCase):
    def test_one_per_family_and_deployed_extra_has_no_duplicate(self):
        kinds = np.array([0, 1] + [2, 3] * 8)
        for selected in range(-1, len(kinds)):
            rows = NAMESPACE['choices'](kinds, selected, np.random.RandomState(2027))
            self.assertEqual([kinds[index] for index, purpose, probability in rows[:4]], [0, 1, 2, 3])
            self.assertEqual(len({row[0] for row in rows}), len(rows))
            self.assertEqual([row[2] for row in rows[:4]], [1, 1, .125, .125])
            if selected >= 0:
                self.assertIn(selected, [row[0] for row in rows])

    def test_sampling_is_seeded_and_does_not_depend_on_annotations(self):
        kinds = np.array([0, 1, 2, 3, 2, 3])
        sample = NAMESPACE['choices']
        self.assertEqual(sample(kinds, -1, np.random.RandomState(2027)),
                         sample(kinds, -1, np.random.RandomState(2027)))
        self.assertEqual([arg.arg for arg in FUNCTION.args.args], ['kinds', 'selected', 'rng'])


class CoverageTests(unittest.TestCase):
    def test_full_cache_cannot_omit_planned_sequences_or_frames(self):
        config = dict(smoke=False, selected_sequences=['a', 'b'],
                      plan=dict(fit_sequences=['a'], holdout_sequences=['b'],
                                expected_frames=200, smoke_frames=31))
        receipt = dict(sequences=2, frames=200, frozen_state_tensors=265)
        verify = NAMESPACE['verify_coverage']
        verify(config, receipt)
        with self.assertRaises(AssertionError):
            verify(dict(config, selected_sequences=['a']), dict(receipt, sequences=1))
        with self.assertRaises(AssertionError):
            verify(config, dict(receipt, frames=1))

    def test_holdout_only_shard_can_supply_a_real_smoke(self):
        config = dict(smoke=True, selected_sequences=['held'],
                      plan=dict(fit_sequences=[], holdout_sequences=['held'],
                                expected_frames=8574, smoke_frames=31))
        receipt = dict(sequences=1, frames=31, frozen_state_tensors=265,
                       observer_policy_trajectory_equal=True)
        NAMESPACE['verify_coverage'](config, receipt)


if __name__ == '__main__':
    unittest.main()
