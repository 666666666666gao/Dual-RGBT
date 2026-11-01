"""Experimental M1 gate on dynamic-template commits of the protected E0."""
import numpy as np
import torch

from dual_rgbt.e0_observation import E0Observation
from dual_rgbt.e0_reliability import E0Reliability


def load_reliability(path, parent_sha256, device):
    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["protected_e0_checkpoint_sha256"] == parent_sha256
    assert checkpoint["selection_partition"] == "train_component_holdout"
    head = E0Reliability().to(device).eval().requires_grad_(False)
    head.load_state_dict(checkpoint["net"], strict=True)
    mean = checkpoint["feature_mean"].to(device)
    scale = checkpoint["feature_scale"].to(device)
    assert mean.shape == scale.shape == (914,)
    assert torch.isfinite(mean).all() and torch.isfinite(scale).all()
    assert (scale > 0).all()
    return head, mean, scale


class M1TemplateGuard:
    """Keep the E0 output/history; optionally undo only this frame's template write."""

    def __init__(self, tracker, head, mean, scale, commit_guard=True):
        self.tracker = tracker
        self.head, self.mean, self.scale = head, mean, scale
        self.commit_guard = commit_guard
        self.observer = E0Observation(tracker.model)
        self.record = None

    def initialize(self, image, initial_box):
        self.record = None
        return self.tracker.initialize(image, initial_box)

    @torch.no_grad()
    def track(self, image):
        previous = self.tracker.templates[:, 1].clone() if self.commit_guard else None
        self.observer.begin(True)
        box, route = self.tracker.track(image)
        assert self.observer.calls == 1 and self.observer.record is not None
        values = torch.from_numpy(self.observer.record["features"]).to(self.mean)[None]
        probabilities = self.head((values - self.mean) / self.scale).softmax(1)
        assert probabilities.shape == (1, 3) and torch.isfinite(probabilities).all()
        state = int(probabilities.argmax(1).item())
        proposed = self.tracker.frame % 5 == 0 and route["confidence"] >= 0.65
        blocked = bool(self.commit_guard and proposed and state != 0)
        if blocked:
            self.tracker.templates[:, 1].copy_(previous)
        self.record = dict(m1_state=state,
                           m1_reliable=float(probabilities[0, 0].item()),
                           m1_uncertain=float(probabilities[0, 1].item()),
                           m1_lost=float(probabilities[0, 2].item()),
                           template_write_proposed=int(proposed),
                           template_write_blocked=int(blocked))
        assert np.isfinite(box).all()
        return box, route

    def close(self):
        self.observer.close()
