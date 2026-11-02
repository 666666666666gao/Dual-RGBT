"""Candidate advantage over KEEP, using causal cached evidence only."""
import torch
from torch import nn
from torch.nn import functional as F


CONTINUOUS = (0, 1, 2, 3, *range(8, 16), *range(18, 24), 26, 27)


def relative_boxes(boxes, previous):
    assert (boxes[..., 2:] > 0).all() and (previous[..., 2:] > 0).all()
    centre = boxes[..., :2] + boxes[..., 2:] / 2
    old_centre = previous[..., :2] + previous[..., 2:] / 2
    return torch.cat([(centre - old_centre) / previous[..., 2:],
                      (boxes[..., 2:] / previous[..., 2:]).log()], -1)


def predicted_box_iou(current, candidates):
    left = torch.maximum(current[..., :2], candidates[..., :2])
    right = torch.minimum(current[..., :2] + current[..., 2:],
                          candidates[..., :2] + candidates[..., 2:])
    overlap = (right - left).clamp_min(0).prod(-1)
    return overlap / (current[..., 2:].prod(-1) +
                      candidates[..., 2:].prod(-1) - overlap)


def context_features(scores, detector, current_coverage, coverage, current_box,
                     candidate_boxes, previous_box, history, route):
    """No GT, identity labels, sequence IDs, or future states are accepted."""
    count, candidates = detector.shape
    assert scores.shape == (count, candidates + 1)
    assert coverage.shape == (count, candidates, 2)
    assert current_coverage.shape == (count, 2)
    assert current_box.shape == previous_box.shape == (count, 4)
    assert candidate_boxes.shape == (count, candidates, 4)
    assert history.shape == (count, 8) and route.shape == (count, 3)
    observed_current = current_coverage.eq(1).any(-1)
    observed = coverage.eq(1).any(-1) & observed_current[:, None]
    current_score = scores[:, :1].expand(-1, candidates)
    candidate_score = scores[:, 1:]
    assert torch.isfinite(current_score[observed]).all()
    assert torch.isfinite(candidate_score[observed]).all()
    # Missing observations have masked -inf scores and are never selectable.
    current_score = current_score.masked_fill(~observed, 0)
    candidate_score = candidate_score.masked_fill(~observed, 0)
    current_motion = relative_boxes(current_box, previous_box)
    candidate_motion = relative_boxes(candidate_boxes, previous_box[:, None])
    features = torch.cat([
        torch.stack([current_score, candidate_score,
                     candidate_score - current_score, detector], -1),
        current_coverage[:, None].expand(-1, candidates, -1), coverage,
        current_motion[:, None].expand(-1, candidates, -1), candidate_motion,
        history[:, None].expand(-1, candidates, -1),
        route[:, None].expand(-1, candidates, -1),
        predicted_box_iou(current_box[:, None], candidate_boxes)[..., None]], -1).to(scores.dtype)
    assert features.shape == (count, candidates, 28)
    assert torch.isfinite(features).all()
    return features, observed


class E0ContextAdvantage(nn.Module):
    """KEEP is exactly zero; the zero-initialized head initially chooses KEEP."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(28, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("mean", torch.zeros(28))
        self.register_buffer("scale", torch.ones(28))

    @torch.no_grad()
    def fit_normalization(self, fit_features, fit_observed):
        values = fit_features[fit_observed]
        assert len(values) > 0 and torch.isfinite(values).all()
        self.mean.zero_()
        self.scale.fill_(1)
        self.mean[list(CONTINUOUS)] = values[:, CONTINUOUS].mean(0)
        self.scale[list(CONTINUOUS)] = values[:, CONTINUOUS].std(
            0, unbiased=False).clamp_min(1e-6)

    def forward(self, features, observed):
        assert features.shape[-1] == 28 and observed.shape == features.shape[:2]
        assert observed.dtype == torch.bool and torch.isfinite(features).all()
        advantage = self.net((features - self.mean) / self.scale).squeeze(-1)
        return torch.cat([advantage.new_zeros(len(features), 1),
                          advantage.masked_fill(~observed, float("-inf"))], 1)


def advantage_loss(logits, observed, current_iou, candidate_iou,
                   advantage_weight, protection_weight):
    """GT supplies TRAIN targets only; full-fit sequence weights come from caller."""
    assert logits.shape == (len(current_iou), candidate_iou.shape[1] + 1)
    assert observed.shape == candidate_iou.shape
    assert torch.isfinite(current_iou).all() and torch.isfinite(candidate_iou).all()
    assert ((current_iou >= 0) & (current_iou <= 1)).all()
    assert ((candidate_iou >= 0) & (candidate_iou <= 1)).all()
    assert advantage_weight.shape == protection_weight.shape == current_iou.shape
    assert (advantage_weight >= 0).all() and (protection_weight >= 0).all()
    scores = logits[:, 1:].masked_fill(~observed, 0)
    target = (candidate_iou - current_iou[:, None]).to(scores.dtype)
    regression = F.smooth_l1_loss(scores, target, reduction="none", beta=.1)
    regression = (regression * observed).sum(1) / observed.sum(1).clamp_min(1)
    degrading = observed & current_iou[:, None].ge(.5) & target.lt(0)
    protection = (F.softplus(scores) * degrading).sum(1)
    protection = protection / degrading.sum(1).clamp_min(1)
    regression = (regression * advantage_weight).mean()
    protection = (protection * protection_weight).mean()
    return regression + protection, dict(advantage=regression, protection=protection)
