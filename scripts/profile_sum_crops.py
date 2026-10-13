"""CPU-only paired crop API check using OpenCV sums for border means.

Standalone candidate; the sealed training and tracking entry points are unchanged.
Each arm includes contiguous conversion, normalization and RNG construction.
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
from profile_crop_api import six_channel_crop


def sum_crop(image, box, factor, size, rng=None):
    images = tuple(np.ascontiguousarray(image[:, :, start:start + 3]) for start in (0, 3))
    centre = box[:2] + box[2:] / 2
    side = math.sqrt(float(box[2] * box[3])) * factor
    if rng is not None:
        side *= math.exp(float(rng.normal(0, .25)))
        centre = centre + rng.uniform(-.375, .375, 2) * side
    left, top = centre - side / 2
    affine = np.array([[size / side, 0, -left * size / side],
                       [0, size / side, -top * size / side]], dtype=np.float32)
    count = images[0].shape[0] * images[0].shape[1]
    borders = [tuple(value / count for value in cv2.sumElems(part)[:3]) for part in images]
    crops = [cv2.warpAffine(part, affine, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=border)
             for part, border in zip(images, borders)]
    crop = np.concatenate(crops, 2)
    normalized_box = np.concatenate([(box[:2] - [left, top]) / side, box[2:] / side]).astype(np.float32)
    tensor = torch.from_numpy(((crop.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())
    return tensor, normalized_box, np.array([left, top, side], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    output = root / args.report
    assert not output.exists()
    selected = json.loads((root / 'data/manifest.json').read_text())['dev'][:10]
    records = []
    for sequence in selected:
        valid = sequence['valid_frames']
        for frame in (valid[0], valid[len(valid) // 2]):
            image = read_pair(sequence, frame)
            assert image.dtype == np.uint8 and image.shape[2] == 6
            count = image.shape[0] * image.shape[1]
            assert count * 255 < 2 ** 53
            for start in (0, 3):
                part = np.ascontiguousarray(image[:, :, start:start + 3])
                assert tuple(value / count for value in cv2.sumElems(part)[:3]) == tuple(part.mean((0, 1)).tolist())
            box = np.asarray(sequence['boxes'][frame], dtype=np.float32)
            for factor, size in ((2., 128), (4., 256)):
                for trial in range(1 if size == 128 else 3):
                    identity = len(records)
                    seed = 2027 + identity
                    rngs = [np.random.RandomState(seed) if size == 256 else None for _ in range(3)]
                    calls = [('original', crop_pair), ('six_channel', six_channel_crop), ('sum', sum_crop)]
                    results = [function(image, box, factor, size, rng) for (_, function), rng in zip(calls, rngs)]
                    for candidate in results[1:]:
                        assert torch.equal(results[0][0], candidate[0])
                        assert np.array_equal(results[0][1], candidate[1])
                        assert np.array_equal(results[0][2], candidate[2])
                    if size == 256:
                        reference = rngs[0].get_state()
                        for rng in rngs[1:]:
                            state = rng.get_state()
                            assert reference[0] == state[0]
                            assert np.array_equal(reference[1], state[1])
                            assert reference[2:] == state[2:]
                    seconds = {name: [] for name, _ in calls}
                    for repeat in range(6):
                        offset = (identity + repeat) % 3
                        for name, function in calls[offset:] + calls[:offset]:
                            start = time.perf_counter()
                            function(image, box, factor, size,
                                     np.random.RandomState(seed) if size == 256 else None)
                            seconds[name].append(time.perf_counter() - start)
                    records.append(dict(sequence=sequence['name'], frame=frame, size=size, trial=trial,
                                        seed=seed, image_shape=list(image.shape), exact_border_means=True,
                                        exact_tensor_target_geometry=True, exact_rng_state=True, seconds=seconds))
    summaries = {}
    for name in ('original', 'six_channel', 'sum'):
        times = [value for record in records for value in record['seconds'][name]]
        summaries[name] = dict(calls=len(times), mean_ms=1000 * float(np.mean(times)),
                               p95_ms=1000 * float(np.quantile(times, .95)))
    report = dict(cases=len(records), sequences=len(selected), summaries=summaries,
                  exact_border_means=True, exact_tensor_target_geometry=True, exact_rng_state=True,
                  contiguous_copies_included=True, rng_initialization_included=True,
                  image_decoding_excluded=True, neural_execution=False,
                  torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads(), records=records,
                  scope='Real dev CPU crop calls; six balanced-order repeats in three arms. '
                        'No active entry changes, NN execution, full-trajectory equivalence, '
                        'complete-system speed or tracking accuracy claim.')
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}, indent=2))


if __name__ == '__main__':
    main()
