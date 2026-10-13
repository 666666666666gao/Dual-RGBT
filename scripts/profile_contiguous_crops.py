"""CPU-only paired crop experiment; current training/tracker sources stay unchanged.

The candidate keeps decoded RGB/TIR arrays separate and contiguous until crops
are combined. Compare exact tensors/geometry and crop-only time on real images.
This script is not used by the running full-training pipeline.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from dual_rgbt.data import MEAN, STD, crop_pair, read_pair


def contiguous_crop(images, box, factor, size, rng=None):
    centre = box[:2] + box[2:] / 2
    side = math.sqrt(float(box[2] * box[3])) * factor
    if rng is not None:
        side *= math.exp(float(rng.normal(0, .25)))
        centre = centre + rng.uniform(-.375, .375, 2) * side
    left, top = centre - side / 2
    affine = np.array([[size / side, 0, -left * size / side],
                       [0, size / side, -top * size / side]], dtype=np.float32)
    crops = [cv2.warpAffine(image, affine, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=tuple(image.mean((0, 1)).tolist())) for image in images]
    crop = np.concatenate(crops, 2)
    normalized_box = np.concatenate([(box[:2] - [left, top]) / side, box[2:] / side]).astype(np.float32)
    tensor = torch.from_numpy(((crop.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())
    return tensor, normalized_box, np.array([left, top, side], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '', 'CPU-only experiment.'
    manifest = json.loads((root / 'data/manifest.json').read_text())
    records = []
    for sequence in manifest['dev'][:10]:
        valid = sequence['valid_frames']
        for frame in (valid[0], valid[len(valid) // 2]):
            image = read_pair(sequence, frame)
            separate = tuple(np.ascontiguousarray(image[:, :, start:start + 3]) for start in (0, 3))
            box = np.asarray(sequence['boxes'][frame], dtype=np.float32)
            for factor, size in ((2., 128), (4., 256)):
                identity = len(records)
                seed = 2027 + identity
                def old():
                    return crop_pair(image, box, factor, size,
                                     np.random.RandomState(seed) if size == 256 else None)
                def candidate():
                    return contiguous_crop(separate, box, factor, size,
                                           np.random.RandomState(seed) if size == 256 else None)
                original, changed = old(), candidate()
                assert torch.equal(original[0], changed[0])
                assert np.array_equal(original[1], changed[1])
                assert np.array_equal(original[2], changed[2])
                old_times, new_times = [], []
                for repeat in range(3):
                    calls = ((old, old_times), (candidate, new_times))
                    if (identity + repeat) % 2:
                        calls = calls[::-1]
                    for function, times in calls:
                        start = time.perf_counter()
                        function()
                        times.append(time.perf_counter() - start)
                records.append(dict(sequence=sequence['name'], frame=frame, size=size,
                                    image_shape=list(image.shape), exact_tensor_target_geometry=True,
                                    old_seconds=old_times, candidate_seconds=new_times))
    old = [t for row in records for t in row['old_seconds']]
    new = [t for row in records for t in row['candidate_seconds']]
    report = dict(cases=len(records), sequences=10, paired_timed_calls=len(old),
                  exact_tensor_target_geometry=True, opencv_threads=cv2.getNumThreads(),
                  torch_threads=torch.get_num_threads(), old_mean_ms=1000 * float(np.mean(old)),
                  candidate_mean_ms=1000 * float(np.mean(new)),
                  old_p95_ms=1000 * float(np.quantile(old, .95)),
                  candidate_p95_ms=1000 * float(np.quantile(new, .95)),
                  image_decoding_and_separate_array_setup_excluded=True,
                  neural_execution=False, records=records,
                  scope='40 actual dev crops with alternating paired CPU timings and exact outputs; '
                        'no change to the running model, no end-to-end speed or accuracy result.')
    (root / args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}, indent=2))


if __name__ == '__main__':
    main()
