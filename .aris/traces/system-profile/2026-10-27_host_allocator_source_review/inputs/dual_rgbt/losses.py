import torch
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import generate_heatmap


def localization_loss(prediction, target):
    predicted = box_cxcywh_to_xyxy(prediction)
    truth = box_xywh_to_xyxy(target).clamp(0, 1)
    intersection_lt = torch.maximum(predicted[:, :2], truth[:, :2])
    intersection_rb = torch.minimum(predicted[:, 2:], truth[:, 2:])
    intersection = (intersection_rb - intersection_lt).clamp(min=0).prod(1)
    pred_area = (predicted[:, 2:] - predicted[:, :2]).clamp(min=0).prod(1)
    true_area = (truth[:, 2:] - truth[:, :2]).prod(1)
    union = pred_area + true_area - intersection
    iou = intersection / union.clamp(min=1e-7)
    enclosing = (torch.maximum(predicted[:, 2:], truth[:, 2:]) -
                 torch.minimum(predicted[:, :2], truth[:, :2])).prod(1)
    giou = iou - (enclosing - union) / enclosing.clamp(min=1e-7)
    loss = 2 * (1 - giou) + 5 * (predicted - truth).abs().mean(1)
    return loss, iou


def executor_loss(output, target):
    primary_loss, _ = localization_loss(output['primary']['box'], target)
    heatmap = generate_heatmap(target.unsqueeze(0), 256, 16)[0].unsqueeze(1)
    focal = FocalLoss()(output['primary']['score'], heatmap)
    if output['action'] == 0:
        return primary_loss.mean() + focal
    refined_loss, _ = localization_loss(output['box'], target)
    refined_focal = FocalLoss()(output['score'], heatmap)
    return primary_loss.mean() + focal + refined_loss.mean() + refined_focal
