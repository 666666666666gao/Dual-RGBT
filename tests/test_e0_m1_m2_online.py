"""M1 gate must preserve E0 without executing candidate generation."""
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
import torch

from dual_rgbt.e0_m1_m2_online import E0M1M2Tracker, recovery_due
from dual_rgbt.e0_m2_online import E0M2Tracker
from dual_rgbt.e0_observation import E0Observation


class M1M2GateTests(unittest.TestCase):
    def test_initialize_clears_previous_sequence_diagnostics(self):
        tracker = object.__new__(E0M1M2Tracker)
        observer = object.__new__(E0Observation)
        observer.enabled, observer.calls, observer.record = True, 1, {"old": True}
        tracker.guard = SimpleNamespace(record={"m1_state": 2}, observer=observer)
        with patch.object(E0M2Tracker, "initialize", return_value=.5) as initialize:
            self.assertEqual(tracker.initialize("image", "box"), .5)
        initialize.assert_called_once_with("image", "box")
        self.assertIsNone(tracker.guard.record)
        self.assertIsNone(observer.record)
        self.assertEqual(observer.calls, 0)
        self.assertFalse(observer.enabled)

    def test_only_lost_at_existing_cadence_can_recover(self):
        for state in (0, 1, 2):
            for frame in (1, 2, 25, 26):
                self.assertEqual(recovery_due(frame, (1, 26), state),
                                 state == 2 and frame in (1, 26))

    def test_non_recovery_returns_exact_guard_output(self):
        for state, frame in ((0, 1), (1, 1), (2, 2)):
            tracker = object.__new__(E0M1M2Tracker)
            tracker.base = Mock()
            tracker.base.state = np.asarray([10., 20., 30., 40.], np.float32)
            tracker.base.frame = frame
            tracker.valid_frames = (1, 26)
            tracker.guard = Mock()
            tracker.guard.record = dict(m1_state=state)
            box = tracker.base.state.copy()
            route = dict(confidence=.9)
            tracker.guard.track.return_value = (box, route)
            actual_box, actual_route = tracker.track("image")
            self.assertIs(actual_box, box)
            self.assertIs(actual_route, route)
            tracker.guard.track.assert_called_once_with("image")
            self.assertFalse(tracker.last_m2["invoked"])
            np.testing.assert_array_equal(tracker.base.state, box)

    def test_sampled_lost_no_match_or_replacement_preserves_templates(self):
        for selected in (0, 1):
            tracker = object.__new__(E0M1M2Tracker)
            tracker.base = SimpleNamespace(
                state=np.asarray([10., 20., 30., 40.], np.float32),
                frame=1, since_query=7, history=torch.zeros(1, 8),
                templates=torch.arange(12).reshape(1, 2, 6).clone())
            original = tracker.base.state.copy()
            templates = tracker.base.templates.clone()
            history = tracker.base.history
            tracker.valid_frames = (1, 26)
            tracker.guard = Mock()
            tracker.guard.record = dict(m1_state=2)
            route = dict(confidence=.9, modality=0, action=0)
            tracker.guard.track.return_value = (original, route)
            tracker.model = object()
            tracker.initial_templates = object()
            tracker.anchor = object()
            tracker.head = Mock(return_value=torch.tensor(
                [[2., 1.]] if selected == 0 else [[1., 2.]]))
            candidate_box = np.asarray([[12., 23., 32., 42.]], np.float32)
            execute = Mock(return_value=dict(
                box=candidate_box, features=np.zeros((1, 2, 768), np.float32),
                coverage=np.ones((1, 2), np.float32)))
            collector = SimpleNamespace(execute_candidates=execute)
            with patch.dict("sys.modules", {"scripts.collect_e0_identity_train": collector}):
                with patch.object(torch.Tensor, "cuda", lambda value: value):
                    box, returned_route = tracker.track("image")
            self.assertIs(returned_route, route)
            self.assertTrue(tracker.last_m2["invoked"])
            self.assertEqual(tracker.last_m2["class_index"], selected)
            execute.assert_called_once()
            np.testing.assert_array_equal(execute.call_args[0][3], original)
            self.assertTrue(torch.equal(tracker.base.templates, templates))
            self.assertEqual(tracker.base.since_query, 7)
            self.assertEqual(tracker.base.frame, 1)
            if selected == 0:
                self.assertIs(box, original)
                self.assertIs(tracker.base.history, history)
                np.testing.assert_array_equal(tracker.base.state, original)
            else:
                np.testing.assert_array_equal(box, candidate_box[0])
                expected = torch.tensor([[1., 0., .9, 3. / 30., 4. / 40.,
                                          np.log(32. / 30.), np.log(42. / 40.), .07]])
                torch.testing.assert_close(tracker.base.history, expected.float())

    def test_close_restores_observer(self):
        tracker = object.__new__(E0M1M2Tracker)
        tracker.guard = Mock()
        tracker.close()
        tracker.guard.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
