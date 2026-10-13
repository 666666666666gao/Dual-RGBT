"""Measure the current tracking service and CPU hotspots without changing math."""
import argparse
import cProfile
import json
import os
import pstats
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from dual_rgbt.data import read_pair
from dual_rgbt.model import DualRGBT
from dual_rgbt.tracker import Tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    if args.threads is not None:
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    model = DualRGBT().cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['net'], strict=True)
    tracker = Tracker(model, .05)
    sequence = json.loads((root / 'data/manifest.json').read_text())['dev'][0]
    tracker.initialize(read_pair(sequence, 0), np.asarray(sequence['boxes'][0], dtype=np.float32))
    rows = []
    profiler = cProfile.Profile()
    profiler.enable()
    for frame in range(1, 65):
        start = time.perf_counter()
        image = read_pair(sequence, frame)
        decode_seconds = time.perf_counter() - start
        box, route = tracker.track(image)
        rows.append(dict(frame=frame, box=box.tolist(), decode_seconds=decode_seconds, **route))
    profiler.disable()
    report_path = root / args.report
    profiler.dump_stats(str(report_path.with_suffix('.prof')))
    stats = pstats.Stats(profiler)
    hotspots = sorted(stats.stats.items(), key=lambda row: row[1][3], reverse=True)[:20]
    report = dict(torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads(),
                  frames=rows, warm_start_frame=9,
                  warm_mean_decode_ms=1000 * float(np.mean([row['decode_seconds'] for row in rows[8:]])),
                  warm_mean_track_ms=1000 * float(np.mean([row['seconds'] for row in rows[8:]])),
                  hotspots=[dict(function=str(key), calls=value[1], self_seconds=value[2],
                                 cumulative_seconds=value[3]) for key, value in hotspots],
                  scope='One dev sequence execution profile; no PR/NPR/SR or matched baseline speed claim.')
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2))


if __name__ == '__main__':
    main()
