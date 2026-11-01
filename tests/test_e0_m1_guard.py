"""Check exact commit-only effects, including a forced rejected write, on CPU."""
import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_rgbt.e0_m1_guard import M1TemplateGuard
from test_e0_observation import FakeE0


class FixedHead(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def forward(self, values):
        logits = torch.zeros(1, 3)
        logits[:, self.state] = 10
        return logits


class FakeTracker:
    def __init__(self):
        self.model = FakeE0()
        self.frame = 4
        self.templates = torch.zeros(1, 2, 6, 2, 2)
        self.history = torch.zeros(1, 8)
        self.since_query = 3
        self.state = np.asarray([1., 2., 3., 4.])

    def track(self, image):
        self.model.predict(None, None, self.history, 0.)
        self.frame += 1
        self.templates[:, 1, :3] = 7
        self.history += 1
        self.since_query += 1
        self.state += 1
        return self.state.copy(), dict(modality=0, action=0, confidence=0.9)


class M1GuardTest(unittest.TestCase):
    def make(self, state, enabled):
        tracker = FakeTracker()
        guard = M1TemplateGuard(tracker, FixedHead(state),
                                torch.zeros(914), torch.ones(914), enabled)
        return tracker, guard

    def test_rejected_write_restores_only_dynamic_template(self):
        direct = FakeTracker()
        expected_box, expected_route = direct.track(None)
        tracker, guard = self.make(2, True)
        box, route = guard.track(None)
        np.testing.assert_array_equal(box, expected_box)
        self.assertEqual(route, expected_route)
        self.assertTrue(torch.equal(tracker.history, direct.history))
        self.assertEqual((tracker.frame, tracker.since_query),
                         (direct.frame, direct.since_query))
        self.assertEqual(tracker.templates.abs().sum().item(), 0.)
        self.assertGreater(direct.templates.abs().sum().item(), 0.)
        self.assertEqual(guard.record["template_write_blocked"], 1)
        guard.close()
        self.assertEqual(tracker.model.predict.__func__, FakeE0.predict)

    def test_reliable_write_is_identical_to_original(self):
        tracker, guard = self.make(0, True)
        direct = FakeTracker()
        expected_box, expected_route = direct.track(None)
        box, route = guard.track(None)
        np.testing.assert_array_equal(box, expected_box)
        self.assertEqual(route, expected_route)
        self.assertTrue(torch.equal(tracker.templates, direct.templates))
        self.assertEqual(guard.record["template_write_blocked"], 0)

    def test_observe_mode_never_changes_a_write_even_when_lost(self):
        tracker, guard = self.make(2, False)
        direct = FakeTracker()
        direct.track(None)
        guard.track(None)
        self.assertTrue(torch.equal(tracker.templates, direct.templates))
        self.assertEqual(guard.record["m1_state"], 2)
        self.assertEqual(guard.record["template_write_blocked"], 0)


if __name__ == "__main__":
    unittest.main()
