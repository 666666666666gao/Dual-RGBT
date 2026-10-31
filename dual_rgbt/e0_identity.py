"""Offline candidate identity prototype; not connected to E0 tracking."""
import torch
from torch import nn
from torch.nn import functional as F

from dual_rgbt.e0_observation import roi_descriptor


def primary_candidate_evidence(features, boxes):
    """Read candidate cxcywh boxes in this primary crop's normalized coordinates."""
    assert features.shape == (1, 256, 768)
    assert boxes.ndim == 2 and boxes.shape[1] == 4 and len(boxes) > 0
    assert torch.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
    descriptors, coverage = zip(*(roi_descriptor(features, box[None]) for box in boxes))
    return torch.cat(descriptors), torch.cat(coverage).squeeze(1)


@torch.no_grad()
def initial_identity_anchor(model, templates, search, box):
    """First-frame deep anchor using the frozen E0 encoder, not preview channels.

    Caller supplies the initial target in normalized cxcywh search coordinates.
    This adds two encoder calls and must be accounted for at initialization.
    It does not install hooks, alter templates, or run the E0 decision policy.
    """
    assert not model.training
    assert templates.shape == (1, 2, 6, 128, 128)
    assert search.shape == (1, 6, 256, 256) and box.shape == (1, 4)
    descriptors, coverage = [], []
    for modality in range(2):
        start = modality * 3
        features, _ = model.encode(templates[:, :, start:start + 3], search[:, start:start + 3])
        descriptor, covered = roi_descriptor(features, box)
        descriptors.append(descriptor)
        coverage.append(covered)
    assert torch.cat(coverage).eq(1).all()
    return torch.stack(descriptors, 1).detach().clone()


class E0CandidateIdentity(nn.Module):
    """Score a supplied candidate set against fixed RGB/TIR target anchors.

    Output index zero means no trusted identity match, not STOP or an E0 action.
    Each later index refers to the supplied candidate order. No GT is an input.
    """

    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU())
        self.match = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        self.no_match = nn.Parameter(torch.zeros(()))

    def forward(self, anchor, candidates, coverage):
        assert anchor.ndim == 3 and anchor.shape[1:] == (2, 768)
        assert candidates.ndim == 4 and candidates.shape[0] == len(anchor)
        assert candidates.shape[2:] == (2, 768) and candidates.shape[1] > 0
        assert coverage.shape == candidates.shape[:3]
        assert torch.isfinite(anchor).all() and torch.isfinite(candidates).all()
        assert torch.isfinite(coverage).all() and ((coverage >= 0) & (coverage <= 1)).all()
        reference = F.normalize(self.project(anchor), dim=-1)[:, None]
        evidence = F.normalize(self.project(candidates), dim=-1)
        pair = torch.cat([(evidence - reference).abs(), evidence * reference], -1)
        per_modality = self.match(pair).squeeze(-1)
        observed = coverage.eq(1)
        count = observed.sum(-1)
        score = (per_modality * observed.to(per_modality.dtype)).sum(-1) / count.clamp_min(1)
        score = score.masked_fill(count == 0, float('-inf'))
        return torch.cat([self.no_match.expand(len(anchor), 1), score], 1)


def identity_set_loss(logits, target_index):
    """TRAIN-only set assignment: zero=no match, 1..N=known target candidate.

    The caller must supply actual target-candidate supervision. An unexecuted
    candidate or an unlabeled candidate must not be relabeled as a negative.
    """
    assert target_index.dtype == torch.long and target_index.shape == (len(logits),)
    assert ((target_index >= 0) & (target_index < logits.shape[1])).all()
    assert torch.isfinite(logits.gather(1, target_index[:, None])).all()
    return F.cross_entropy(logits, target_index)
