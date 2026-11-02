"""Check M1 gating, explicit current evidence, KEEP and replacement commits."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from dual_rgbt.e0_m1_relative_m2_online import E0M1RelativeM2Tracker
from dual_rgbt.e0_m1_m2_online import E0M1M2Tracker
from dual_rgbt.e0_relative_identity import E0RelativeIdentity


class RelativeRuntimeTests(unittest.TestCase):
    def test_initialize_resets_relative_diagnostics(self):
        tracker = object.__new__(E0M1RelativeM2Tracker)
        tracker.last_m2 = dict(invoked=True, class_index=3)
        with patch.object(E0M1M2Tracker, "initialize", return_value=.5) as initialize:
            self.assertEqual(tracker.initialize("image", "box"), .5)
        initialize.assert_called_once_with("image", "box")
        self.assertFalse(tracker.last_m2["invoked"])
        self.assertEqual(tracker.last_m2["class_index"], -1)
        self.assertIsNone(tracker.last_m2["keep_score"])

    def fixture(self, state=2, frame=1):
        tracker = object.__new__(E0M1RelativeM2Tracker)
        previous = np.asarray([10., 20., 30., 40.], np.float32)
        proposal = np.asarray([11., 22., 30., 40.], np.float32)
        tracker.base = SimpleNamespace(
            state=previous.copy(), frame=frame - 1, since_query=7,
            history=torch.arange(8).float()[None],
            templates=torch.arange(12).reshape(1, 2, 6).clone())
        route = dict(confidence=.9, modality=0, action=0)
        tracker.guard = Mock()
        tracker.guard.record = dict(m1_state=state)

        def guarded(_):
            tracker.base.state = proposal.copy()
            tracker.base.frame = frame
            return proposal, route

        tracker.guard.track.side_effect = guarded
        tracker.model = object()
        tracker.initial_templates = torch.zeros(1, 2, 6, 128, 128)
        tracker.anchor = torch.ones(1, 2, 768)
        tracker.valid_frames = (1, 26)
        tracker.head = Mock()
        return tracker, previous, proposal, route

    def test_non_lost_or_unsampled_never_queries_recovery(self):
        for state, frame in ((0, 1), (1, 1), (2, 2)):
            tracker, _, proposal, route = self.fixture(state, frame)
            with patch(
                    "dual_rgbt.e0_m1_relative_m2_online.current_proposal_evidence"
            ) as evidence:
                box, actual_route = tracker.track("image")
            self.assertIs(box, proposal)
            self.assertIs(actual_route, route)
            evidence.assert_not_called()
            tracker.head.assert_not_called()
            self.assertFalse(tracker.last_m2["invoked"])
            self.assertEqual(tracker.last_m2["current_encoder_calls"], 0)

    def run_recovery(self, selected, current_coverage=None):
        tracker, previous, proposal, route = self.fixture()
        templates = tracker.base.templates.clone()
        history = tracker.base.history
        current = dict(features=torch.full((2, 768), 2.),
                       coverage=torch.ones(2) if current_coverage is None else current_coverage)
        candidate = np.asarray([[12., 23., 32., 42.]], np.float32)
        execute = Mock(return_value=dict(
            box=candidate, features=np.full((1, 2, 768), 3., np.float32),
            coverage=np.ones((1, 2), np.float32)))
        crop = Mock(return_value=(torch.zeros(6, 256, 256), None, (4, 5, 6)))
        if current_coverage is None:
            tracker.head.return_value = torch.tensor(
                [[2., 1.]] if selected == 0 else [[1., 2.]])
        else:
            tracker.head = E0RelativeIdentity().eval()
        with patch.dict("sys.modules", {
                "profile_crop_api": SimpleNamespace(six_channel_crop=crop),
                "scripts.collect_e0_identity_train": SimpleNamespace(execute_candidates=execute)}):
            with patch(
                    "dual_rgbt.e0_m1_relative_m2_online.current_proposal_evidence",
                    return_value=current) as evidence:
                box, actual_route = tracker.track("image")
        np.testing.assert_array_equal(crop.call_args[0][1], previous)
        np.testing.assert_array_equal(evidence.call_args[0][4], proposal)
        np.testing.assert_array_equal(execute.call_args[0][3], previous)
        self.assertIs(evidence.call_args[0][1], tracker.initial_templates)
        self.assertIs(execute.call_args[0][1], tracker.initial_templates)
        self.assertIs(actual_route, route)
        self.assertTrue(torch.equal(tracker.base.templates, templates))
        self.assertEqual(tracker.base.frame, 1)
        self.assertEqual(tracker.base.since_query, 7)
        self.assertEqual(tracker.last_m2["class_index"], selected)
        self.assertEqual(tracker.last_m2["current_encoder_calls"], 2)
        self.assertEqual(tracker.last_m2["candidate_encoder_calls"], 24)
        if current_coverage is None:
            args = tracker.head.call_args[0]
            self.assertTrue(torch.equal(args[1], current["features"][None]))
            self.assertTrue(torch.equal(args[2], current["coverage"][None]))
        if selected == 0:
            self.assertIs(box, proposal)
            self.assertIs(tracker.base.history, history)
            np.testing.assert_array_equal(tracker.base.state, proposal)
        else:
            np.testing.assert_array_equal(box, candidate[0])
            expected = torch.tensor([[1., 0., .9, 3. / 30., 4. / 40.,
                                      np.log(32. / 30.), np.log(42. / 40.), .07]])
            torch.testing.assert_close(tracker.base.history, expected.float())

    def test_keep_preserves_guarded_e0_state(self):
        self.run_recovery(0)

    def test_replacement_uses_selected_candidate_without_template_write(self):
        self.run_recovery(1)

    def test_unobserved_current_forces_keep_with_actual_relative_head(self):
        self.run_recovery(0, current_coverage=torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
