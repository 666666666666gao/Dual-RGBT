"""Training-only synthetic states and actual executed-box counterfactual labels."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from dual_rgbt.data import crop_pair, read_pair
from dual_rgbt_evidence.readout import pixel_xywh
from dual_rgbt_evidence.regions import broad_geometry, crop_region, execute_boxes, pixel_iou, search_geometry


class EvidencePairs(Dataset):
    def __init__(self, manifest, samples, seed):
        self.sequences = json.loads(Path(manifest).read_text())['train']
        self.samples, self.seed = samples, seed

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed + index)
        sequence = self.sequences[rng.randint(len(self.sequences))]
        valid = np.asarray(sequence['valid_frames'])
        frame = int(valid[rng.randint(2, len(valid))])
        earlier = valid[valid < frame]
        template_ids = rng.choice(earlier, 2, replace=True)
        boxes = np.asarray(sequence['boxes'], dtype=np.float32)
        templates = torch.stack([crop_pair(read_pair(sequence, int(t)), boxes[t], 2., 128)[0]
                                 for t in template_ids])
        image = read_pair(sequence, frame)
        target = boxes[frame]
        anchor = target.copy()
        anchor[2:] *= np.exp(float(rng.normal(0, .25)))
        geometry = search_geometry(anchor)
        geometry[:2] += rng.uniform(-.75, .75, 2) * geometry[2]
        broad = broad_geometry(geometry)
        means = image.mean((0, 1))
        return dict(templates=templates, image=image, means=means,
                    main=crop_region(image, geometry, 256, means),
                    main_preview=crop_region(image, geometry, 64, means),
                    broad_preview=crop_region(image, broad, 64, means),
                    main_geometry=torch.from_numpy(geometry), broad_geometry=torch.from_numpy(broad),
                    target=torch.from_numpy(target), image_size=torch.tensor(image.shape[:2]),
                    sequence=sequence['name'], frame=frame)


def collate_pairs(rows):
    stack = ('templates', 'main', 'main_preview', 'broad_preview', 'main_geometry',
             'broad_geometry', 'target', 'image_size')
    result = {name: torch.stack([row[name] for row in rows]) for name in stack}
    result.update(images=[row['image'] for row in rows], means=[row['means'] for row in rows],
                  sequences=[row['sequence'] for row in rows], frames=[row['frame'] for row in rows])
    return result


def query_pixels(images, means, geometries, action):
    side = 256 if action == 4 else 128
    crops = [crop_region(image, geometry, side, mean)
             for image, mean, geometry in zip(images, means, geometries.detach().cpu().numpy())]
    return torch.stack(crops).cuda()


def relative_target(target, geometry):
    centre = (target[:, :2] + target[:, 2:] / 2 - geometry[:, :2]) / geometry[:, 2:3]
    return torch.cat([centre, target[:, 2:] / geometry[:, 2:3]], dim=1)


def heatmap_target(target, geometry, side):
    centre = relative_target(target, geometry)[:, :2]
    visible = ((centre >= 0) & (centre < 1)).all(dim=1)
    cells = torch.arange(side, device=target.device, dtype=target.dtype)
    x = centre[:, :1, None] * side - .5
    y = centre[:, 1:, None] * side - .5
    distance = (cells[None, None, :] - x).square() + (cells[None, :, None] - y).square()
    return torch.exp(-distance / 2)[:, None] * visible[:, None, None, None], visible


def path_loss(output, target, geometry):
    heatmap, visible = heatmap_target(target, geometry, output['score'].shape[-1])
    truth = relative_target(target, geometry)
    # Missing target centres supervise background, not an extrapolated box.
    regression = F.smooth_l1_loss(output['box'], truth, reduction='none').mean(1)
    regression = (regression * visible).sum() / visible.sum().clamp(min=1)
    return 5 * regression + F.binary_cross_entropy(output['score'], heatmap), visible


def preview_loss(preview, target, main_geometry, broad_geometry):
    loss = target.new_zeros(())
    for maps, geometry in ((preview['maps'], main_geometry), (preview['broad_maps'], broad_geometry)):
        truth, _ = heatmap_target(target, geometry, 8)
        for scores in maps:
            loss = loss + F.binary_cross_entropy_with_logits(10 * scores[:, None], truth) / 4
    return loss


@torch.no_grad()
def counterfactuals(model, templates, main, main_preview, broad_preview, history,
                    main_geometry, broad_geometry, target, image_sizes, images, means, window):
    preview = model.preview_state(templates, main_preview, broad_preview, history)
    primaries, queries, quality, gains, executed = [], [], [], [], []
    for modality in range(2):
        primary = model.executor.primary(templates, main, modality, window)
        regions = model.query_geometries(primary, preview, modality, main_geometry, broad_geometry)
        main_box = execute_boxes(pixel_xywh(primary['box'], main_geometry), image_sizes)
        main_iou = pixel_iou(main_box, target)
        action_boxes, deltas = [main_box], []
        for action in range(1, 5):
            pixels = query_pixels(images, means, regions[:, action - 1], action)
            output = model.query(templates, pixels, primary, main_geometry, regions[:, action - 1], modality)
            box = execute_boxes(output['pixel_box'], image_sizes)
            action_boxes.append(box)
            deltas.append(pixel_iou(box, target) - main_iou)
        primaries.append(primary)
        queries.append(regions)
        quality.append(main_iou)
        gains.append(torch.stack(deltas, dim=1))
        executed.append(torch.stack(action_boxes, dim=1))
    return dict(primaries=primaries, queries=queries, quality=torch.stack(quality, dim=1),
                gains=gains, executed=executed)


def policy_loss(model, targets, templates, main_preview, broad_preview, history,
                main_geometry, broad_geometry, target, train_preview):
    preview = model.preview_state(templates, main_preview, broad_preview, history)
    quality_loss = F.smooth_l1_loss(preview['quality'], targets['quality'])
    gain_loss = target.new_zeros(())
    for modality in range(2):
        predictions = model.gain_predictions(targets['primaries'][modality], preview, modality,
                                             main_geometry, broad_geometry, targets['queries'][modality])[:, 1:]
        gain_loss = gain_loss + F.smooth_l1_loss(predictions, targets['gains'][modality]) / 2
    spatial = preview_loss(preview, target, main_geometry, broad_geometry) if train_preview else target.new_zeros(())
    return quality_loss + gain_loss + .1 * spatial, quality_loss, gain_loss, spatial
