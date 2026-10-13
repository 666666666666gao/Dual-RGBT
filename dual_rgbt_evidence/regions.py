"""Raw-image regions with explicitly independent original-image coordinates."""
import math

import cv2
import numpy as np
import torch

from dual_rgbt.data import MEAN, STD
from lib.utils.box_ops import clip_box


def search_geometry(box, factor=4.):
    centre = box[:2] + box[2:] / 2
    side = math.sqrt(float(box[2] * box[3])) * factor
    return np.asarray([centre[0] - side / 2, centre[1] - side / 2, side], dtype=np.float32)


def broad_geometry(main):
    return np.asarray([main[0] - main[2] / 2, main[1] - main[2] / 2, main[2] * 2], dtype=np.float32)


def crop_region(image, geometry, size, means):
    """Warp only the requested pixels; means are cached once per current frame."""
    left, top, side = geometry
    affine = np.asarray([[size / side, 0, -left * size / side],
                         [0, size / side, -top * size / side]], dtype=np.float32)
    crops = [cv2.warpAffine(image[:, :, start:start + 3], affine, (size, size),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=tuple(means[start:start + 3].tolist()))
             for start in (0, 3)]
    pixels = np.concatenate(crops, axis=2).astype(np.float32) / 255
    return torch.from_numpy(((pixels - MEAN) / STD).transpose(2, 0, 1).copy())


def execute_boxes(pixel_boxes, image_sizes):
    """Use exactly the same margin10, float32 executed-box conversion online and in labels."""
    raw = pixel_boxes.detach().cpu().tolist()
    sizes = image_sizes.detach().cpu().tolist()
    boxes = [clip_box(box, int(size[0]), int(size[1]), margin=10) for box, size in zip(raw, sizes)]
    return pixel_boxes.new_tensor(np.asarray(boxes, dtype=np.float32))


def pixel_iou(prediction, target):
    """Uncensored GT in original-image xywh; no crop-relative GT clipping."""
    left = torch.maximum(prediction[:, :2], target[:, :2])
    right = torch.minimum(prediction[:, :2] + prediction[:, 2:], target[:, :2] + target[:, 2:])
    intersection = (right - left).clamp(min=0).prod(1)
    union = prediction[:, 2:].prod(1) + target[:, 2:].prod(1) - intersection
    return intersection / union
