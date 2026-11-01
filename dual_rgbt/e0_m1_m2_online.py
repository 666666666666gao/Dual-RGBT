"""TRAIN candidate for M1-gated M2 recovery; separate from frozen R15."""
import numpy as np
import torch

from dual_rgbt.e0_m1_guard import M1TemplateGuard
from dual_rgbt.e0_m2_online import E0M2Tracker


def recovery_due(frame, sampled_frames, m1_state):
    """Use the existing lost class and cadence, without another threshold."""
    return m1_state == 2 and frame in sampled_frames


class E0M1M2Tracker(E0M2Tracker):
    def __init__(self, model, cost_weight, identity_checkpoint, head, mean, scale):
        super().__init__(model, cost_weight, identity_checkpoint)
        self.guard = M1TemplateGuard(self.base, head, mean, scale, commit_guard=True)

    @torch.no_grad()
    def initialize(self, image, initial_box):
        self.guard.record = None
        self.guard.observer.begin(False)
        return super().initialize(image, initial_box)

    @torch.no_grad()
    def track(self, image):
        previous = self.base.state.copy()
        box, route = self.guard.track(image)
        self.last_m2 = dict(invoked=False, class_index=-1, no_match_score=None,
                            selected_score=None)
        if not recovery_due(self.base.frame, self.valid_frames,
                            self.guard.record["m1_state"]):
            return box, route

        from scripts.collect_e0_identity_train import execute_candidates

        candidates = execute_candidates(self.model, self.initial_templates, image, previous)
        features = torch.from_numpy(candidates["features"][None]).cuda()
        coverage = torch.from_numpy(candidates["coverage"][None]).cuda()
        logits = self.head(self.anchor, features, coverage)[0]
        selected = int(logits.argmax().item())
        self.last_m2 = dict(invoked=True, class_index=selected,
                            no_match_score=float(logits[0].item()),
                            selected_score=float(logits[selected].item()))
        if selected == 0:
            return box, route

        # M1 already guarded this frame's E0 write; recovery does not write templates.
        self.base.state = np.asarray(candidates["box"][selected - 1], dtype=np.float32)
        movement = ((self.base.state[:2] + self.base.state[2:] / 2) -
                    (previous[:2] + previous[2:] / 2)) / previous[2:]
        size_change = np.log(self.base.state[2:] / previous[2:])
        identity = [float(route["modality"] == 0), float(route["modality"] == 1)]
        values = identity + [route["confidence"]] + movement.tolist() + size_change.tolist() + [
            self.base.since_query / 100.]
        self.base.history = self.base.history.new_tensor(values)[None]
        return self.base.state.copy(), route

    def close(self):
        self.guard.close()
