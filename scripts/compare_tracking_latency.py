"""Paired causal prefix latency of original stripped CADTrack and Dual-RGBT.

No accuracy scoring or training. Each tracker decodes each image separately and
maintains its own predicted search/template state. Alternate execution order.
This standalone diagnostic is not invoked by the full training/test controller.
"""
import argparse
import hashlib
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
from dual_rgbt.data import read_pair
from dual_rgbt.model import DualRGBT
from dual_rgbt.tracker import Tracker
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.test.tracker.cadtrack import CADTrack
from lib.test.utils import TrackerParams


def statistics(seconds):
    return dict(frames=len(seconds), mean_ms=1000 * float(np.mean(seconds)),
                p95_ms=1000 * float(np.quantile(seconds, .95)),
                seconds=sum(seconds), fps=len(seconds) / sum(seconds))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cost-weight', type=float, required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--sequences', type=int, default=10)
    parser.add_argument('--frames', type=int, default=64)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1')
    assert args.sequences > 0 and args.frames > 9
    output = root / args.report
    assert not output.exists()
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    baseline_checkpoint = Path('/home/gaob/CADTrack_ep0014.pth.tar')
    checkpoint = Path(args.checkpoint)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (baseline_checkpoint, checkpoint)}
    manifest = json.loads((root / 'data/manifest.json').read_text())
    assert args.sequences <= len(manifest['dev'])
    selected = manifest['dev'][:args.sequences]
    assert all(len(s['rgb']) == len(s['tir']) == len(s['boxes']) >= args.frames
               and 0 in s['valid_frames'] for s in selected)
    update_config_from_file(str(root / 'experiments/cadtrack/cadtrack.yaml'))
    params = TrackerParams()
    params.cfg = cfg
    params.checkpoint = str(baseline_checkpoint)
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.save_all_boxes = False
    params.debug = 0
    baseline = CADTrack(params)
    baseline.network.requires_grad_(False)
    model = DualRGBT().cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
    model.requires_grad_(False)
    ours = Tracker(model, args.cost_weight)
    trackers = {'baseline': baseline, 'ours': ours}
    records = []
    for sequence_index, sequence in enumerate(selected):
        initial_box = np.asarray(sequence['boxes'][0], dtype=np.float32)
        for frame in range(args.frames):
            order = ['baseline', 'ours']
            if (sequence_index + frame) % 2:
                order.reverse()
            row = dict(sequence=sequence['name'], frame=frame, order=order)
            for name in order:
                tracker = trackers[name]
                torch.cuda.synchronize()
                start = time.perf_counter()
                image = read_pair(sequence, frame)
                if frame == 0:
                    if name == 'baseline':
                        tracker.initialize(image, {'init_bbox': initial_box.tolist()})
                    else:
                        tracker.initialize(image, initial_box)
                    box = initial_box
                else:
                    result = tracker.track(image)
                    if name == 'baseline':
                        box = np.asarray(result['target_bbox'], dtype=np.float32)
                    else:
                        box, route = result
                        row['ours_action'] = route['action']
                        row['ours_modality'] = route['modality']
                        row['ours_heavy_tokens'] = route['primary_tokens'] + route['auxiliary_tokens']
                torch.cuda.synchronize()
                row[name + '_seconds'] = time.perf_counter() - start
                assert np.isfinite(box).all() and (box[2:] > 0).all()
            records.append(row)
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    warm = [r for r in records if r['frame'] >= 9]
    services = {name: statistics([r[name + '_seconds'] for r in records])
                for name in trackers}
    warmed = {name: statistics([r[name + '_seconds'] for r in warm])
              for name in trackers}
    tracked = [r for r in records if r['frame'] > 0]
    sources = ['scripts/compare_tracking_latency.py', 'experiments/cadtrack/cadtrack.yaml',
               'lib/test/tracker/cadtrack.py', 'lib/test/tracker/data_utils.py',
               'lib/train/data/processing_utils.py', 'lib/models/cadtrack/cadtrack.py',
               'lib/models/cadtrack/vit_care.py', 'dual_rgbt/model.py',
               'dual_rgbt/data.py', 'dual_rgbt/tracker.py']
    report = dict(physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                  gpu_name=torch.cuda.get_device_name(0),
                  torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads(),
                  checkpoint_sha256=hashes, config=vars(args),
                  source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                 for p in sources},
                  sequences=[s['name'] for s in selected],
                  frames_per_sequence=args.frames, full_service=services, warm_service=warmed,
                  warm_start_frame=9,
                  full_service_reduction_percent=100 * (1 - services['ours']['mean_ms'] /
                                                       services['baseline']['mean_ms']),
                  warm_service_reduction_percent=100 * (1 - warmed['ours']['mean_ms'] /
                                                       warmed['baseline']['mean_ms']),
                  parameter_counts={'baseline': sum(p.numel() for p in baseline.network.parameters()),
                                    'ours': sum(p.numel() for p in model.parameters())},
                  ours_action_counts=[sum(r['ours_action'] == a for r in tracked) for a in range(5)],
                  ours_modality_counts=[sum(r['ours_modality'] == m for r in tracked) for m in range(2)],
                  initialization_included=True, image_decoding_included=True,
                  model_loading_excluded=True, accuracy_evaluated=False,
                  optimizer_updates=0, original_checkpoints_unchanged=True, records=records,
                  scope='Matched causal dev sequence prefixes only, on one physical GPU. '
                        'Both models remain resident; alternating arms each decode images. '
                        'No full-LasHeR speed conclusion, baseline accuracy rerun, or training.')
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
