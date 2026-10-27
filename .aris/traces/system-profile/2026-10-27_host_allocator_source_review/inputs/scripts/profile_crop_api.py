"""CPU-only paired crop diagnostic including six-channel conversion costs.

The reviewed interior kernel is unchanged. Its wrapper accepts the existing
six-channel input and includes all contiguous copies inside the timed call.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from dual_rgbt.data import crop_pair, read_pair
from profile_interior_crops import interior_crop


def six_channel_crop(image, box, factor, size, rng=None):
    images = tuple(np.ascontiguousarray(image[:, :, start:start + 3]) for start in (0, 3))
    return interior_crop(images, box, factor, size, rng)[:3]


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
            box = np.asarray(sequence['boxes'][frame], dtype=np.float32)
            for factor, size in ((2., 128), (4., 256)):
                for trial in range(1 if size == 128 else 3):
                    identity = len(records)
                    seed = 2027 + identity
                    first_rng = np.random.RandomState(seed) if size == 256 else None
                    second_rng = np.random.RandomState(seed) if size == 256 else None
                    baseline = crop_pair(image, box, factor, size, first_rng)
                    candidate = six_channel_crop(image, box, factor, size, second_rng)
                    assert torch.equal(baseline[0], candidate[0])
                    assert np.array_equal(baseline[1], candidate[1])
                    assert np.array_equal(baseline[2], candidate[2])
                    if first_rng is not None:
                        first_state, second_state = first_rng.get_state(), second_rng.get_state()
                        assert first_state[0] == second_state[0]
                        assert np.array_equal(first_state[1], second_state[1])
                        assert first_state[2:] == second_state[2:]
                    images = tuple(np.ascontiguousarray(image[:, :, start:start + 3]) for start in (0, 3))
                    classified = interior_crop(images, box, factor, size,
                                               np.random.RandomState(seed) if size == 256 else None)
                    seconds = dict(original=[], six_channel=[])
                    calls = [('original', crop_pair), ('six_channel', six_channel_crop)]
                    for repeat in range(4):
                        offset = (identity + repeat) % 2
                        for name, function in calls[offset:] + calls[:offset]:
                            start = time.perf_counter()
                            function(image, box, factor, size,
                                     np.random.RandomState(seed) if size == 256 else None)
                            seconds[name].append(time.perf_counter() - start)
                    records.append(dict(sequence=sequence['name'], frame=frame, size=size, trial=trial,
                                        seed=seed, image_shape=list(image.shape),
                                        exact_tensor_target_geometry=True, exact_rng_state=True,
                                        mean_skipped=classified[3], seconds=seconds))
    summaries = {}
    for name in ('original', 'six_channel'):
        times = [t for record in records for t in record['seconds'][name]]
        summaries[name] = dict(calls=len(times), mean_ms=1000 * float(np.mean(times)),
                               p95_ms=1000 * float(np.quantile(times, .95)))
    report = dict(cases=len(records), sequences=len(selected), summaries=summaries,
                  mean_skipped_cases=sum(r['mean_skipped'] for r in records),
                  mean_retained_cases=sum(not r['mean_skipped'] for r in records),
                  exact_tensor_target_geometry=True, exact_rng_state=True,
                  contiguous_copy_and_guard_included=True, rng_initialization_included=True,
                  image_decoding_excluded=True, torch_threads=torch.get_num_threads(),
                  opencv_threads=cv2.getNumThreads(), neural_execution=False, records=records,
                  scope='Paired real-dev CPU crop API calls including candidate contiguous copies. '
                        'Four balanced-order repeats per case; no production change, whole-system '
                        'latency, full tracking accuracy or module-effectiveness claim.')
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
