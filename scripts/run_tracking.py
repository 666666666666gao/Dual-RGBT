"""Complete causal tracking, with one tracker process per physical GPU.

Ground truth initializes frame zero only. Subsequent predictions control search
and templates. Accuracy is computed separately with the supplied RGBT toolkit.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from dual_rgbt.data import read_pair
from dual_rgbt.model import DualRGBT
from dual_rgbt.tracker import Tracker


def sequences(split):
    manifest = json.loads((root / 'data/manifest.json').read_text())
    if split == 'dev':
        return manifest['dev']
    result = []
    for name in manifest['test_sequences']:
        directory = root / 'data/LasHeR/test' / name
        rgb = sorted((directory / 'visible').glob('*.jpg'))
        tir = sorted((directory / 'infrared').glob('*.jpg'))
        boxes = np.loadtxt(str(directory / 'init.txt'), delimiter=',', ndmin=2)
        assert len(rgb) == len(tir) == len(boxes), name
        result.append(dict(name=name, rgb=list(map(str, rgb)), tir=list(map(str, tir)),
                           boxes=boxes.tolist()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--cost-weight', type=float, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--smoke-frames', type=int, default=0,
                        help='Run one sequence prefix for execution checks only.')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1')
    assert 0 <= args.shard < args.shards
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    entire_split = sequences(args.split)
    selected = entire_split[args.shard::args.shards]
    if args.smoke_frames:
        selected = selected[:1]
    run = root / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    for directory in ('predictions', 'groundtruth', 'traces'):
        (run / directory).mkdir()
    checkpoint = Path(args.checkpoint)
    config = dict(vars(args), physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                  split_sequences=len(entire_split), split_frames=sum(len(s['boxes']) for s in entire_split),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  sequences=[s['name'] for s in selected], image_decoding_included=True)
    if args.split == 'test' and not args.smoke_frames:
        frozen = json.loads((root / 'reports/frozen_test_selection.json').read_text())
        assert config['checkpoint_sha256'] == frozen['checkpoint_sha256']
        assert args.cost_weight == frozen['cost_weight']
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    model = DualRGBT().cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
    tracker = Tracker(model, args.cost_weight)
    all_seconds, receipts = [], []
    total_start = time.perf_counter()
    for sequence in selected:
        name = sequence['name']
        boxes = np.asarray(sequence['boxes'], dtype=np.float32)
        count = min(len(boxes), args.smoke_frames) if args.smoke_frames else len(boxes)
        predictions = [boxes[0].copy()]
        start = time.perf_counter()
        image = read_pair(sequence, 0)
        initialization = tracker.initialize(image, boxes[0])
        initialization_including_decode = time.perf_counter() - start
        all_seconds.append(initialization_including_decode)
        frame_seconds = []
        actions = np.zeros(5, dtype=np.int64)
        modalities = np.zeros(2, dtype=np.int64)
        token_sum = 0
        with (run / 'traces' / (name + '.csv')).open('w', newline='') as handle:
            fields = ['frame', 'modality', 'action', 'confidence', 'seconds',
                      'seconds_including_decode', 'primary_tokens', 'auxiliary_tokens',
                      'primary_confidence', 'query_confidence', 'query_selected', 'template_updated',
                      'primary_box_x', 'primary_box_y', 'primary_box_w', 'primary_box_h',
                      'query_box_x', 'query_box_y', 'query_box_w', 'query_box_h',
                      'final_box_x', 'final_box_y', 'final_box_w', 'final_box_h']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for frame in range(1, count):
                start = time.perf_counter()
                image = read_pair(sequence, frame)
                prediction, route = tracker.track(image)
                elapsed = time.perf_counter() - start
                assert np.isfinite(prediction).all() and (prediction[2:] > 0).all()
                predictions.append(prediction)
                frame_seconds.append(elapsed)
                actions[route['action']] += 1
                modalities[route['modality']] += 1
                token_sum += route['primary_tokens'] + route['auxiliary_tokens']
                writer.writerow(dict(frame=frame, seconds_including_decode=elapsed, **route))
        predictions = np.asarray(predictions)
        assert predictions.shape == (count, 4)
        np.savetxt(run / 'predictions' / (name + '.txt'), predictions, fmt='%.6f', delimiter=',')
        np.savetxt(run / 'groundtruth' / (name + '.txt'), boxes[:count], fmt='%.6f', delimiter=',')
        all_seconds.extend(frame_seconds)
        receipt = dict(sequence=name, frames=count, initialized_once=True,
                       initialization_seconds=initialization,
                       initialization_seconds_including_decode=initialization_including_decode,
                       online_seconds_including_decode=sum(frame_seconds),
                       action_counts=actions.tolist(), modality_counts=modalities.tolist(),
                       mean_heavy_tokens=token_sum / (count - 1))
        receipts.append(receipt)
        (run / 'progress.json').write_text(json.dumps(receipts, indent=2))
        print(json.dumps(receipt), flush=True)
    summary = dict(sequences=len(receipts), frames=sum(r['frames'] for r in receipts),
                   seconds=sum(all_seconds), wall_seconds=time.perf_counter() - total_start,
                   mean_ms=1000 * float(np.mean(all_seconds)),
                   p95_ms=1000 * float(np.quantile(all_seconds, .95)),
                   fps=len(all_seconds) / sum(all_seconds), image_decoding_included=True,
                   initialization_included=True, model_loading_excluded=True,
                   smoke_only=bool(args.smoke_frames), physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                   scope='Causal tracking predictions; PR/NPR/SR require separate toolkit scoring.')
    np.save(run / 'frame_seconds.npy', np.asarray(all_seconds))
    (run / ('SMOKE.json' if args.smoke_frames else 'COMPLETE.json')).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
