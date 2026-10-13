import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


MEAN = np.array([.485, .456, .406, .449, .449, .449], dtype=np.float32)
STD = np.array([.229, .224, .225, .226, .226, .226], dtype=np.float32)


def read_pair(sequence, frame):
    rgb = cv2.cvtColor(cv2.imread(str(sequence['rgb'][frame])), cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(cv2.imread(str(sequence['tir'][frame])), cv2.COLOR_BGR2RGB)
    assert rgb.shape == tir.shape, 'Registered pair must have equal image geometry.'
    return np.concatenate([rgb, tir], 2)


def crop_pair(image, box, factor, size, rng=None):
    centre = box[:2] + box[2:] / 2
    side = math.sqrt(float(box[2] * box[3])) * factor
    if rng is not None:
        side *= math.exp(float(rng.normal(0, .25)))
        centre = centre + rng.uniform(-.375, .375, 2) * side
    left, top = centre - side / 2
    affine = np.array([[size / side, 0, -left * size / side],
                       [0, size / side, -top * size / side]], dtype=np.float32)
    crops = [cv2.warpAffine(image[:, :, start:start + 3], affine, (size, size),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=tuple(image[:, :, start:start + 3].mean((0, 1)).tolist()))
             for start in (0, 3)]
    crop = np.concatenate(crops, 2)
    normalized_box = np.concatenate([(box[:2] - [left, top]) / side, box[2:] / side]).astype(np.float32)
    tensor = torch.from_numpy(((crop.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())
    return tensor, normalized_box, np.array([left, top, side], dtype=np.float32)


class LasHeRPairs(Dataset):
    def __init__(self, manifest, split, samples, seed=2027):
        document = json.loads(Path(manifest).read_text(encoding='utf-8'))
        self.sequences = document[split]
        self.samples = samples
        self.seed = seed

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed + index)
        sequence = self.sequences[rng.randint(len(self.sequences))]
        valid = np.array(sequence['valid_frames'])
        assert len(valid) >= 3
        search_id = int(valid[rng.randint(2, len(valid))])
        earlier = valid[valid < search_id]
        template_ids = rng.choice(earlier, 2, replace=True)
        boxes = np.asarray(sequence['boxes'], dtype=np.float32)
        templates = [crop_pair(read_pair(sequence, int(t)), boxes[t], 2., 128)[0]
                     for t in template_ids]
        image = read_pair(sequence, search_id)
        search, box, _ = crop_pair(image, boxes[search_id], 4., 256, rng)
        return {'templates': torch.stack(templates), 'search': search,
                'target': torch.from_numpy(box), 'sequence': sequence['name'],
                'frame': search_id}
