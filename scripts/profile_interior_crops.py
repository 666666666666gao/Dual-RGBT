"""CPU crop diagnostic: retain original pixels, avoid unused border means.

The running training/tracker sources stay unchanged. Compare original strided,
contiguous, and conditional-border crops on the same real dev images and RNG.
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
from profile_contiguous_crops import contiguous_crop


def interior_crop(images, box, factor, size, rng=None):
    centre = box[:2] + box[2:] / 2
    side = math.sqrt(float(box[2] * box[3])) * factor
    if rng is not None:
        side *= math.exp(float(rng.normal(0, .25)))
        centre = centre + rng.uniform(-.375, .375, 2) * side
    left, top = centre - side / 2
    affine = np.array([[size / side, 0, -left * size / side],
                       [0, size / side, -top * size / side]], dtype=np.float32)
    inverse = cv2.invertAffineTransform(affine.astype(np.float64))
    corners = np.array([[0, 0, 1], [size - 1, 0, 1], [0, size - 1, 1],
                        [size - 1, size - 1, 1]], dtype=np.float64) @ inverse.T
    height, width = images[0].shape[:2]
    interior = bool((corners.min(0) >= 1).all() and
                    (corners.max(0) <= np.array([width - 2, height - 2])).all())
    borders = ([(0., 0., 0.)] * 2 if interior else
               [tuple(image.mean((0, 1)).tolist()) for image in images])
    crops = [cv2.warpAffine(image, affine, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=border)
             for image, border in zip(images, borders)]
    crop = np.concatenate(crops, 2)
    normalized_box = np.concatenate([(box[:2] - [left, top]) / side, box[2:] / side]).astype(np.float32)
    tensor = torch.from_numpy(((crop.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())
    return tensor, normalized_box, np.array([left, top, side], dtype=np.float32), interior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '', 'CPU-only experiment.'
    output = root / args.report
    assert not output.exists()
    selected = json.loads((root / 'data/manifest.json').read_text())['dev'][:10]
    records = []
    for sequence in selected:
        valid = sequence['valid_frames']
        for frame in (valid[0], valid[len(valid) // 2]):
            image = read_pair(sequence, frame)
            images = tuple(np.ascontiguousarray(image[:, :, start:start + 3]) for start in (0, 3))
            box = np.asarray(sequence['boxes'][frame], dtype=np.float32)
            for factor, size in ((2., 128), (4., 256)):
                for trial in range(1 if size == 128 else 3):
                    identity = len(records)
                    seed = 2027 + identity
                    def original():
                        return crop_pair(image, box, factor, size,
                                         np.random.RandomState(seed) if size == 256 else None)
                    def contiguous():
                        return contiguous_crop(images, box, factor, size,
                                               np.random.RandomState(seed) if size == 256 else None)
                    def conditioned():
                        return interior_crop(images, box, factor, size,
                                             np.random.RandomState(seed) if size == 256 else None)
                    baseline, second, third = original(), contiguous(), conditioned()
                    for candidate in (second, third):
                        assert torch.equal(baseline[0], candidate[0])
                        assert np.array_equal(baseline[1], candidate[1])
                        assert np.array_equal(baseline[2], candidate[2])
                    times = {'original': [], 'contiguous': [], 'interior': []}
                    calls = [('original', original), ('contiguous', contiguous), ('interior', conditioned)]
                    for repeat in range(3):
                        offset = (identity + repeat) % 3
                        for name, function in calls[offset:] + calls[:offset]:
                            start = time.perf_counter()
                            function()
                            times[name].append(time.perf_counter() - start)
                    records.append(dict(sequence=sequence['name'], frame=frame, size=size,
                                        trial=trial, seed=seed, image_shape=list(image.shape),
                                        exact_tensor_target_geometry=True,
                                        mean_skipped=third[3], seconds=times))
    summaries = {}
    for name in ('original', 'contiguous', 'interior'):
        seconds = [t for r in records for t in r['seconds'][name]]
        summaries[name] = dict(calls=len(seconds), mean_ms=1000 * float(np.mean(seconds)),
                               p95_ms=1000 * float(np.quantile(seconds, .95)))
    report = dict(cases=len(records), sequences=len(selected),
                  mean_skipped_cases=sum(r['mean_skipped'] for r in records),
                  mean_retained_cases=sum(not r['mean_skipped'] for r in records),
                  exact_tensor_target_geometry=True, summaries=summaries,
                  torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads(),
                  image_decoding_and_separate_array_setup_excluded=True,
                  neural_execution=False, records=records,
                  scope='Real dev CPU crops, one template and three search jitters per frame, '
                        'matched three-arm timing order and exact outputs. No production change, '
                        'complete-system speed, accuracy or new training result.')
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
