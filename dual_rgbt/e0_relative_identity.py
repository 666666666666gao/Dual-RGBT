"""Relative current-E0/candidate actions; no tracker or online integration."""
import torch

from dual_rgbt.e0_identity import E0CandidateIdentity


class E0RelativeIdentity(E0CandidateIdentity):
    """Reuse identity weights, replacing the scalar no-match action with E0."""

    def __init__(self):
        super().__init__()
        self.no_match.requires_grad_(False)

    def forward(self, anchor, current, current_coverage, candidates, coverage):
        assert current.shape == (len(anchor), 2, 768)
        assert current_coverage.shape == (len(anchor), 2)
        evidence = torch.cat([current[:, None], candidates], 1)
        observed = torch.cat([current_coverage[:, None], coverage], 1)
        scores = super().forward(anchor, evidence, observed)[:, 1:]
        eligible = current_coverage.eq(1).any(1)
        keep_only = torch.full_like(scores, float("-inf"))
        keep_only[:, 0] = 0.
        return torch.where(eligible[:, None], scores, keep_only)


def relative_action_targets(current_iou, candidate_iou, current_coverage, coverage):
    """TRAIN-only best improving observed candidate, else KEEP; exact ties first."""
    assert current_iou.ndim == 1 and candidate_iou.ndim == 2
    assert candidate_iou.shape[0] == len(current_iou) and candidate_iou.shape[1] > 0
    assert current_iou.dtype == candidate_iou.dtype == torch.float64
    assert current_coverage.shape == (len(current_iou), 2)
    assert coverage.shape == (*candidate_iou.shape, 2)
    for values in (current_iou, candidate_iou, current_coverage, coverage):
        assert torch.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()
    trainable = current_coverage.eq(1).any(1)
    improving = (coverage.eq(1).any(2) & (candidate_iou >= .5)
                 & (candidate_iou > current_iou[:, None]) & trainable[:, None])
    best = candidate_iou.masked_fill(~improving, float("-inf")).argmax(1) + 1
    targets = torch.where(improving.any(1), best, torch.zeros_like(best))
    return targets, trainable
