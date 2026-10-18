"""LasHeR-dev diagnostics for candidate routing; keeps weights and architecture fixed.

This runner changes only the runtime decision policy. It is not a deployable model
and its outputs are diagnostic evidence for deciding the next repair experiment.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_inputs import load_manifest, sha
from dual_rgbt.data import read_pair
from dual_rgbt_global.candidate_policy import build_candidate_policy
from dual_rgbt_global.candidate_tracker import CandidateEvidenceTracker


class DiagnosticModel:
    def __init__(self, inner, modality_mode, query_mode):
        self.inner = inner
        self.modality_mode = modality_mode
        self.query_mode = query_mode

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def eval(self):
        self.inner.eval()
        return self

    @torch.no_grad()
    def preview_state(self, *args, **kwargs):
        preview = dict(self.inner.preview_state(*args, **kwargs))
        quality = preview['quality'].clone()
        if self.modality_mode == 'rgb':
            quality[:, 0] = quality.max(1).values + 1.
            quality[:, 1] = quality[:, 0] - 2.
        elif self.modality_mode == 'tir':
            quality[:, 1] = quality.max(1).values + 1.
            quality[:, 0] = quality[:, 1] - 2.
        preview['quality'] = quality
        return preview

    @torch.no_grad()
    def candidate_gains(self, *args, **kwargs):
        gains = self.inner.candidate_gains(*args, **kwargs)
        if self.query_mode == 'auto':
            return gains
        forced = torch.zeros_like(gains)
        if self.query_mode == 'stop':
            return forced
        kinds = kwargs.get('kinds')
        if kinds is None and len(args) >= 7:
            kinds = args[6]
        if self.query_mode == 'primary':
            forced[:, 1] = 1.
        else:
            wanted = 2 if self.query_mode == 'global_local' else 3
            matches = (kinds == wanted)
            assert matches.shape[1:] == (gains.shape[1] - 1,)
            forced[:, 1:] = matches.to(forced.dtype)
            assert matches.any(1).all(), (self.query_mode, kinds)
        return forced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', default='data/benchmarks/lasher_dev.json')
    parser.add_argument('--run', required=True)
    parser.add_argument('--modality', choices=('auto', 'rgb', 'tir'), default='auto')
    parser.add_argument('--query', choices=('auto', 'stop', 'primary', 'global_local', 'global_wide'), default='auto')
    parser.add_argument('--cost-weight', type=float, default=.15)
    parser.add_argument('--max-candidates', type=int, default=8)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--max-sequences', type=int)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
    manifest_path = ROOT / args.manifest
    manifest = load_manifest(manifest_path)
    assert manifest['benchmark'] == 'LasHeR' and manifest['split'] == 'dev'
    if args.max_sequences is not None:
        assert 1 <= args.max_sequences <= len(manifest['sequences'])
        manifest = dict(manifest, sequences=manifest['sequences'][:args.max_sequences])
    checkpoint = ROOT / args.checkpoint
    checkpoint_sha = sha(checkpoint)
    run = ROOT / args.run
    run.mkdir(parents=True, exist_ok=False)
    (run / 'predictions').mkdir()
    (run / 'groundtruth').mkdir()
    (run / 'traces').mkdir()
    config = dict(vars(args), checkpoint_sha256=checkpoint_sha,
                  manifest_sha256=sha(manifest_path), scope='Diagnostic only; fixed weights and architecture; no test selection.')
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    factory = ROOT / 'runs/stage_b_refreshed_20261013/latest.pth'
    base = build_candidate_policy(factory).cuda().eval()
    state = torch.load(checkpoint, map_location='cpu')
    base.load_state_dict(state['net'], strict=True)
    del state
    base.requires_grad_(False)
    model = DiagnosticModel(base, args.modality, args.query)
    tracker = CandidateEvidenceTracker(model, args.cost_weight, args.max_candidates)
    actions = np.zeros(5, dtype=np.int64)
    modalities = np.zeros(2, dtype=np.int64)
    seconds = []
    receipts = []
    total_frames = 0
    wall_start = time.perf_counter()
    for sequence in manifest['sequences']:
        name = sequence['name']
        predictions = [np.asarray(sequence['initial_box'], dtype=np.float32)]
        image = read_pair(sequence, 0)
        tracker.initialize(image, sequence['initial_box'])
        with (run / 'traces' / (name + '.csv')).open('w', newline='') as handle:
            fields = ['frame', 'modality', 'action', 'confidence', 'seconds', 'seconds_including_decode',
                      'primary_tokens', 'auxiliary_tokens', 'request_index', 'candidate_rank', 'candidate_count']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            online = []
            for frame in range(1, len(sequence['rgb'])):
                image = read_pair(sequence, frame)
                start = time.perf_counter()
                prediction, route = tracker.track(image)
                elapsed = time.perf_counter() - start
                assert np.isfinite(prediction).all() and (prediction[2:] > 0).all()
                predictions.append(prediction)
                online.append(elapsed)
                seconds.append(float(route['seconds']))
                actions[route['action']] += 1
                modalities[route['modality']] += 1
                writer.writerow(dict(frame=frame, seconds_including_decode=elapsed, **route))
            np.savetxt(run / 'predictions' / (name + '.txt'), predictions, fmt='%.6f', delimiter=',')
            np.savetxt(run / 'groundtruth' / (name + '.txt'), np.asarray(sequence['boxes'], dtype=np.float32), fmt='%.9g', delimiter=',')
            receipts.append(dict(sequence=name, frames=len(predictions), action_counts=actions.tolist(), modality_counts=modalities.tolist(), online_seconds=sum(online)))
            total_frames += len(predictions)
            (run / 'progress.json').write_text(json.dumps(receipts, indent=2))
            print(json.dumps(receipts[-1]), flush=True)
    timing = np.asarray(seconds, dtype=np.float64)
    summary = dict(status='COMPLETE', diagnostic=True, sequences=len(receipts), frames=total_frames,
                   tracked_frames=int(timing.size), checkpoint_sha256=checkpoint_sha,
                   action_counts=actions.tolist(), modality_counts=modalities.tolist(),
                   auxiliary_rate=float(actions[1:].sum() / timing.size), tir_rate=float(modalities[1] / timing.size),
                   mean_ms=1000 * float(timing.mean()), p95_ms=1000 * float(np.quantile(timing, .95)),
                   fps=float(timing.size / timing.sum()), wall_seconds=time.perf_counter() - wall_start,
                   scope='Diagnostic only; fixed checkpoint/architecture; use dev metrics for route diagnosis, never test tuning.')
    np.save(run / 'frame_seconds.npy', timing)
    (run / 'COMPLETE.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

