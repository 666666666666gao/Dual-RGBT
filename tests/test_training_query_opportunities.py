import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_training_query_opportunities import count_label


class QueryOpportunityTests(unittest.TestCase):
    def test_recovery_is_not_confused_with_small_improvement_or_harm(self):
        group = Counter()
        count_label(group, .05, .75, 3, True)
        count_label(group, .8, .05, 0, False)
        count_label(group, .05, .2, 2, True)
        self.assertEqual(group['labels'], 3)
        self.assertEqual(group['main_lost'], 2)
        self.assertEqual(group['recovers_lost_to_0p5'], 1)
        self.assertEqual(group['stopped_recovery'], 1)
        self.assertEqual(group['destroys_good_to_0p1'], 1)

    def test_utility_accounts_for_local_and_wide_token_costs(self):
        local, wide = Counter(), Counter()
        count_label(local, .2, .28, 2, False)
        count_label(wide, .2, .28, 3, False)
        self.assertEqual(local['positive_utility_at_0.1'], 1)
        self.assertEqual(wide['positive_utility_at_0.1'], 0)
        self.assertEqual(local['positive_utility_at_0'], wide['positive_utility_at_0'])
        self.assertEqual(local['stopped_recovery'], 0)


if __name__ == '__main__':
    unittest.main()
