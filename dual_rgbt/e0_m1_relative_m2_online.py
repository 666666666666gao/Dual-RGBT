"""M1-gated relative M2 runtime; caller must admit the trained head and run."""
import numpy as np
import torch

from dual_rgbt.e0_current_evidence import current_proposal_evidence
from dual_rgbt.e0_m1_guard import M1TemplateGuard
from dual_rgbt.e0_m1_m2_online import E0M1M2Tracker, recovery_due
from dual_rgbt.e0_relative_identity import E0RelativeIdentity
from dual_rgbt.tracker import Tracker


class E0M1RelativeM2Tracker(E0M1M2Tracker):
    """Use current E0 as action zero; keep existing M1 gate and candidate order.

    Construction does not grant launch permission. The caller must verify the
    full TRAIN acceptance and qualifying checkpoint, plus separate runtime
    source/run admission. Frame timing must include this entire track call.
    """

    def __init__(self, model, cost_weight, relative_head, m1_head, mean, scale):
        assert isinstance(relative_head, E0RelativeIdentity)
        self.model = model
        self.base = Tracker(model, cost_weight)
        self.head = relative_head.to(next(model.parameters()).device).eval().requires_grad_(False)
        self.valid_frames = None
        self.guard = M1TemplateGuard(
            self.base, m1_head, mean, scale, commit_guard=True)

    def reset_relative_record(self):
        self.last_m2 = dict(
            invoked=False, class_index=-1, keep_score=None, selected_score=None,
            current_encoder_calls=0, candidate_encoder_calls=0)

    @torch.no_grad()
    def initialize(self, image, initial_box):
        elapsed = super().initialize(image, initial_box)
        self.reset_relative_record()
        return elapsed

    @torch.no_grad()
    def track(self, image):
        previous = self.base.state.copy()
        box, route = self.guard.track(image)
        self.reset_relative_record()
        if not recovery_due(self.base.frame, self.valid_frames,
                            self.guard.record["m1_state"]):
            return box, route

        from profile_crop_api import six_channel_crop
        from scripts.collect_e0_identity_train import execute_candidates

        search, _, geometry = six_channel_crop(image, previous, 4., 256)
        current = current_proposal_evidence(
            self.model, self.initial_templates, search[None].to(self.anchor),
            geometry, box)
        candidates = execute_candidates(
            self.model, self.initial_templates, image, previous)
        features = torch.from_numpy(candidates["features"][None]).to(self.anchor)
        coverage = torch.from_numpy(candidates["coverage"][None]).to(self.anchor)
        logits = self.head(
            self.anchor, current["features"][None], current["coverage"][None],
            features, coverage)[0]
        selected = int(logits.argmax().item())
        self.last_m2 = dict(
            invoked=True, class_index=selected,
            keep_score=float(logits[0].item()),
            selected_score=float(logits[selected].item()),
            current_encoder_calls=2, candidate_encoder_calls=24)
        if selected == 0:
            return box, route

        # M1 has handled the E0 write; recovery changes only position/history.
        self.base.state = np.asarray(candidates["box"][selected - 1], dtype=np.float32)
        movement = ((self.base.state[:2] + self.base.state[2:] / 2) -
                    (previous[:2] + previous[2:] / 2)) / previous[2:]
        size_change = np.log(self.base.state[2:] / previous[2:])
        identity = [float(route["modality"] == 0), float(route["modality"] == 1)]
        values = identity + [route["confidence"]] + movement.tolist() + size_change.tolist() + [
            self.base.since_query / 100.]
        self.base.history = self.base.history.new_tensor(values)[None]
        return self.base.state.copy(), route
