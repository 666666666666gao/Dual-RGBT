"""C2-aware policy refresh loss; the executor remains frozen in eval mode."""
import torch
from torch.nn import functional as F

from dual_rgbt.losses import localization_loss


def configure_policy_refresh(model):
    model.requires_grad_(False)
    for module in (model.preview, model.quality, model.gain):
        module.requires_grad_(True)
    model.eval()
    return [
        dict(params=list(model.preview.parameters()), lr=1e-5),
        dict(params=list(model.quality.parameters()) + list(model.gain.parameters()),
             lr=1e-4),
    ]


def policy_refresh_loss(model, templates, search, history, target, window):
    """Relabel both primaries and eight queries using this update's preview.

    C2 missing-stream conditioning and query candidates depend on preview.
    Recompute executor outcomes without gradient, while retaining gradients
    through preview/quality/gain predictions. No tracker state is updated.
    """
    assert window is not None
    preview = model.preview_state(templates, search, history)
    primaries, candidates, qualities, gains, outcomes = [], [], [], [], []
    with torch.no_grad():
        for modality in range(2):
            primary = model.primary(
                templates, search, modality, window, preview=preview)
            indices, boxes = model.candidates(primary, preview, modality)
            main_loss, _ = localization_loss(primary["box"], target)
            action_gains, action_boxes = [], []
            for action in range(1, 5):
                refined = model.correct(
                    templates, search, primary, modality, action,
                    indices[:, action - 1], boxes[:, action - 1], window,
                    preview=preview)
                refined_loss, _ = localization_loss(refined["box"], target)
                action_gains.append(main_loss - refined_loss)
                action_boxes.append(refined["box"])
            primaries.append(primary)
            candidates.append((indices, boxes))
            qualities.append(-main_loss)
            gains.append(torch.stack(action_gains, 1))
            outcomes.append(torch.stack(action_boxes, 1))
    quality_targets = torch.stack(qualities, 1)
    quality_loss = F.smooth_l1_loss(preview["quality"], quality_targets)
    gain_loss = search.new_zeros(())
    for modality in range(2):
        indices, boxes = candidates[modality]
        predicted = model.gain_predictions(
            primaries[modality], preview, modality, indices, boxes)[:, 1:]
        gain_loss = gain_loss + F.smooth_l1_loss(predicted, gains[modality]) / 2
    centre = target[:, :2] + target[:, 2:] / 2
    inside = torch.isfinite(centre).all(1) & ((centre >= 0) & (centre < 1)).all(1)
    spatial_loss = search.new_zeros(())
    if inside.any():
        cell = (centre[inside] * 8).long()
        target_cell = cell[:, 1] * 8 + cell[:, 0]
        spatial_loss = sum(F.cross_entropy(scores[inside].flatten(1), target_cell)
                           for scores in preview["maps"]) / 2
    loss = quality_loss + gain_loss + .1 * spatial_loss
    assert torch.isfinite(loss)
    return dict(
        loss=loss, quality_loss=quality_loss, gain_loss=gain_loss,
        spatial_loss=spatial_loss, quality_targets=quality_targets,
        gain_targets=torch.stack(gains, 1),
        primary_boxes=torch.stack([item["box"] for item in primaries], 1),
        candidate_indices=torch.stack([item[0] for item in candidates], 1),
        candidate_boxes=torch.stack([item[1] for item in candidates], 1),
        query_boxes=torch.stack(outcomes, 1))
