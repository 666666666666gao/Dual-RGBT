"""Offline localization reliability prototype; not connected to E0 inference."""
import torch
from torch import nn
from torch.nn import functional as F


STATE_NAMES = ('reliable', 'uncertain', 'lost')


def localization_targets(final_iou):
    """Overlap labels describe localization, not identity or recoverability."""
    assert final_iou.ndim == 1 and torch.isfinite(final_iou).all()
    assert ((final_iou >= 0) & (final_iou <= 1)).all()
    targets = torch.ones_like(final_iou, dtype=torch.long)
    targets[final_iou >= .5] = 0
    targets[final_iou < .1] = 2
    return targets


class E0Reliability(nn.Module):
    """Read only observation features; never take GT, actions or tracker state."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(914, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, features):
        assert features.ndim == 2 and features.shape[1] == 914
        return self.classifier(features)


def localization_loss(logits, final_iou):
    assert logits.shape == (len(final_iou), len(STATE_NAMES))
    return F.cross_entropy(logits, localization_targets(final_iou))
