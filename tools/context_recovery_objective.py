"""One added fit-only recovery regression term; original loss stays unchanged."""
import torch
from torch.nn import functional as F

from context_e0_train_data import sequence_weights
from dual_rgbt.e0_context_advantage import advantage_loss


def recovery_weights(values):
    recovering = (values["observed"] & values["current_iou"][:, None].lt(.1)
                  & values["candidate_iou"].ge(.5))
    return sequence_weights(values["sequence_index"], recovering.any(1))


def recovery_loss(logits, observed, current_iou, candidate_iou,
                  advantage_weight, protection_weight, recovery_weight):
    original, parts = advantage_loss(
        logits, observed, current_iou, candidate_iou,
        advantage_weight, protection_weight)
    scores = logits[:, 1:].masked_fill(~observed, 0)
    target = (candidate_iou - current_iou[:, None]).to(scores.dtype)
    recovering = observed & current_iou[:, None].lt(.1) & candidate_iou.ge(.5)
    regression = F.smooth_l1_loss(scores, target, reduction="none", beta=.1)
    per_set = (regression * recovering).sum(1) / recovering.sum(1).clamp_min(1)
    recovery = (per_set * recovery_weight).mean()
    return original + recovery, dict(parts, recovery=recovery)
