"""Keep dev behavior fixed and require an explicit exploratory-test plan."""
import unittest

from cached_gain_evaluation_contract import scope


class EvaluationSplitTests(unittest.TestCase):
    def test_dev_smoke_and_full_scope_remain_unchanged(self):
        self.assertEqual(scope('dev', 11, False, False), (97, 54646, False))
        self.assertEqual(scope('dev', 0, False, True), (97, 54646, False))

    def test_test_requires_explicit_exploratory_plan_and_full_run(self):
        self.assertEqual(scope('test', 0, True, True), (245, 220703, True))
        for smoke_frames, exploratory_test, full_allowed in (
                (11, True, True), (0, False, True), (0, True, False)):
            with self.subTest(smoke_frames=smoke_frames, exploratory_test=exploratory_test,
                              full_allowed=full_allowed):
                with self.assertRaises(AssertionError):
                    scope('test', smoke_frames, exploratory_test, full_allowed)

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(AssertionError):
            scope('train', 0, False, False)


if __name__ == '__main__':
    unittest.main()
