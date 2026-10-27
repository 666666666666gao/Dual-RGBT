import unittest

from dual_rgbt_global.query_acceptance import QueryAcceptance, ROUTES, route_name


class QueryAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.policy = QueryAcceptance({name: .65 for name in ROUTES})

    def test_routes_are_modality_specific(self):
        self.assertEqual(route_name(0, 1), 'rgb_local128')
        self.assertEqual(route_name(1, 1), 'tir_local128')
        self.assertEqual(route_name(0, 4), 'rgb_wide')
        self.assertEqual(route_name(1, 4), 'tir_wide')
        self.assertEqual(route_name(0, 0), 'rgb_primary')

    def test_stop_never_accepts(self):
        self.assertFalse(self.policy.accepts(0, 0, 1.0))
        self.assertFalse(self.policy.accepts_template(0, 0, 1.0, True))

    def test_route_threshold_and_template_separation(self):
        self.assertFalse(self.policy.accepts(0, 4, .649999))
        self.assertTrue(self.policy.accepts(0, 4, .65))
        self.assertTrue(self.policy.accepts_template(0, 4, .65, True))
        self.assertFalse(self.policy.accepts_template(0, 4, .65, False))

    def test_missing_or_invalid_thresholds_are_rejected(self):
        with self.assertRaises(AssertionError):
            QueryAcceptance({'rgb_primary': .65})
        with self.assertRaises(AssertionError):
            QueryAcceptance({name: 1.1 for name in ROUTES})
        with self.assertRaises(AssertionError):
            self.policy.accepts(0, 4, 1.1)


if __name__ == '__main__':
    unittest.main()
