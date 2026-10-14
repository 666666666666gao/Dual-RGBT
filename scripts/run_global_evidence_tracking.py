"""Causal global-policy dev/test tracking; use the unchanged toolkit scorer."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt.data import read_pair
from dual_rgbt_global.policy import build_policy
from dual_rgbt_global.tracker import EvidenceTracker
from run_tracking import sequences
from train_global_evidence_policy import cpu_state


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def reload_probe(model, checkpoint, factory, saved, expected_box, expected_route, expected_tracker, cost_weight):
    clone = build_policy(factory).cuda().eval()
    clone.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
    freeze(clone)
    tracker = EvidenceTracker(clone, cost_weight)
    tracker.state = saved['state'].copy()
    tracker.templates = saved['templates'].clone()
    tracker.history = saved['history'].clone()
    tracker.window = saved['window'].clone()
    tracker.frame = saved['frame']
    tracker.since_query = saved['since_query']
    actual_box, actual_route = tracker.track(saved['image'])
    assert np.array_equal(actual_box, expected_box)
    fields = ('modality', 'action', 'confidence', 'primary_tokens', 'auxiliary_tokens')
    assert all(actual_route[field] == expected_route[field] for field in fields)
    assert torch.equal(tracker.history, expected_tracker.history)
    assert torch.equal(tracker.templates, expected_tracker.templates)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in clone.parameters())
    return dict(status='PASS', strict_reload_box_bit_equal=True, route_fields=list(fields),
                strict_reload_history_templates_bit_equal=True, optimizer_updates=0, backward_calls=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--split', choices=('train', 'dev', 'test'), required=True)
    parser.add_argument('--cost-weight', type=float, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--smoke-frames', type=int, default=0)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1')
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert 0 <= args.shard < args.shards and args.cost_weight >= 0
    assert args.split != 'train' or args.smoke_frames
    assert args.smoke_frames == 0 or args.smoke_frames == 11
    plan = json.loads((ROOT / args.plan).read_text())
    review = json.loads((ROOT / plan['source_review']).read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan)
    assert review['source_sha256'] == plan['source_sha256']
    training = json.loads((ROOT / plan['training_plan']).read_text())
    assert sha(ROOT / plan['training_plan']) == plan['training_plan_sha256']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert sha(ROOT / 'data/manifest.json') == training['manifest_sha256']
    factory = ROOT / training['input_checkpoint']
    assert sha(factory) == training['input_checkpoint_sha256']
    protected = Path('/home/gaob/CADTrack_ep0014.pth.tar')
    assert sha(protected) == plan['protected_checkpoint_sha256']
    checkpoint = ROOT / args.checkpoint
    assert sha(checkpoint) == args.checkpoint_sha256
    state = torch.load(checkpoint, map_location='cpu')
    stage = state['stage']
    assert stage in ('a', 'b', 'c') and not state['config']['sanity']
    assert state['config']['plan_sha256'] == plan['training_plan_sha256']
    assert args.checkpoint == 'runs/' + training['runs']['formal_' + stage] + '/latest.pth'
    expected_updates = plan['completed_updates'][stage]
    assert state['step'] == expected_updates
    native = json.loads((ROOT / 'runs/pipeline_tasks' / ('train_' + training['runs']['formal_' + stage]) / 'status.json').read_text())
    complete = json.loads((checkpoint.parent / 'COMPLETE.json').read_text())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert complete['updates'] == expected_updates and complete['stage'] == stage and not complete['sanity']
    if not args.smoke_frames:
        assert stage == 'c' and args.split in ('dev', 'test')
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if args.split == 'train':
        manifest = json.loads((ROOT / 'data/manifest.json').read_text())
        entire_split = manifest['train']
        selected = [next(row for row in entire_split if row['name'] == plan['smoke_sequence'])]
    else:
        entire_split = sequences(args.split)
        selected = entire_split[args.shard::args.shards]
        if args.smoke_frames:
            selected = selected[:1]
    split_frames = sum(len(row['boxes']) for row in entire_split)
    if not args.smoke_frames:
        assert len(entire_split) == plan['split_sizes'][args.split]['sequences']
        assert split_frames == plan['split_sizes'][args.split]['frames']
    frozen_sha = None
    if args.split == 'test' and not args.smoke_frames:
        frozen_path = ROOT / plan['frozen_test_selection']
        frozen = json.loads(frozen_path.read_text())
        assert frozen['selection_split'] == 'dev' and frozen['dev_sequences'] == 97
        assert frozen['dev_frames'] == plan['split_sizes']['dev']['frames']
        assert frozen['checkpoint_sha256'] == args.checkpoint_sha256
        assert frozen['cost_weight'] == args.cost_weight
        assert frozen['evaluation_plan_sha256'] == sha(ROOT / args.plan)
        assert frozen['training_plan_sha256'] == plan['training_plan_sha256']
        frozen_sha = sha(frozen_path)
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    for name in ('predictions', 'groundtruth', 'traces'):
        (run / name).mkdir()
    config = dict(vars(args), physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                  split_sequences=len(entire_split), split_frames=split_frames,
                  sequences=[row['name'] for row in selected], checkpoint_sha256=args.checkpoint_sha256,
                  checkpoint_stage=stage, checkpoint_updates=expected_updates,
                  plan_sha256=sha(ROOT / args.plan), training_plan_sha256=plan['training_plan_sha256'],
                  source_sha256=plan['source_sha256'], factory_checkpoint_sha256=training['input_checkpoint_sha256'],
                  frozen_selection_sha256=frozen_sha, image_decoding_included=True,
                  metric_scope='Full causal tracking requires unchanged toolkit scoring; smoke is execution only.')
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    model = build_policy(factory).cuda().eval()
    model.load_state_dict(state['net'], strict=True)
    del state
    freeze(model)
    before = cpu_state(model)
    tracker = EvidenceTracker(model, args.cost_weight)
    seconds, receipts = [], []
    wall_start = time.perf_counter()
    for sequence in selected:
        name = sequence['name']
        boxes = np.asarray(sequence['boxes'], dtype=np.float32)
        count = min(len(boxes), args.smoke_frames) if args.smoke_frames else len(boxes)
        assert count > 1
        predictions = [boxes[0].copy()]
        started = time.perf_counter()
        image = read_pair(sequence, 0)
        init_seconds = tracker.initialize(image, boxes[0])
        init_with_decode = time.perf_counter() - started
        seconds.append(init_with_decode)
        online_seconds, tokens = [], 0
        actions, modalities = np.zeros(5, dtype=np.int64), np.zeros(2, dtype=np.int64)
        with (run / 'traces' / (name + '.csv')).open('w', newline='') as handle:
            fields = ('frame', 'modality', 'action', 'confidence', 'seconds',
                      'seconds_including_decode', 'primary_tokens', 'auxiliary_tokens')
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for frame in range(1, count):
                started = time.perf_counter()
                image = read_pair(sequence, frame)
                if args.smoke_frames and frame == count - 1:
                    saved = dict(state=tracker.state.copy(), templates=tracker.templates.clone(),
                                 history=tracker.history.clone(), window=tracker.window.clone(),
                                 frame=tracker.frame, since_query=tracker.since_query, image=image)
                prediction, route = tracker.track(image)
                elapsed = time.perf_counter() - started
                assert np.isfinite(prediction).all() and (prediction[2:] > 0).all()
                predictions.append(prediction)
                online_seconds.append(elapsed)
                actions[route['action']] += 1
                modalities[route['modality']] += 1
                tokens += route['primary_tokens'] + route['auxiliary_tokens']
                writer.writerow(dict(frame=frame, seconds_including_decode=elapsed, **route))
        np.savetxt(run / 'predictions' / (name + '.txt'), np.asarray(predictions), fmt='%.6f', delimiter=',')
        np.savetxt(run / 'groundtruth' / (name + '.txt'), boxes[:count], fmt='%.6f', delimiter=',')
        seconds.extend(online_seconds)
        receipt = dict(sequence=name, frames=count, initialized_once=True,
                       initialization_seconds=init_seconds, initialization_seconds_including_decode=init_with_decode,
                       online_seconds_including_decode=sum(online_seconds),
                       action_counts=actions.tolist(), modality_counts=modalities.tolist(),
                       mean_heavy_tokens=tokens / (count - 1))
        receipts.append(receipt)
        (run / 'progress.json').write_text(json.dumps(receipts, indent=2))
        print(json.dumps(receipt), flush=True)
    wall_seconds = time.perf_counter() - wall_start
    after = cpu_state(model)
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(parameter.grad is None for parameter in model.parameters())
    if args.smoke_frames:
        sanity = reload_probe(model, checkpoint, factory, saved, prediction, route, tracker, args.cost_weight)
        sanity.update(frozen_state_tensors=len(before), tracked_frames=sum(row['frames'] - 1 for row in receipts))
        (run / 'SANITY.json').write_text(json.dumps(sanity, indent=2))
    assert sha(checkpoint) == args.checkpoint_sha256
    assert sha(factory) == training['input_checkpoint_sha256']
    assert sha(protected) == plan['protected_checkpoint_sha256']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    if frozen_sha is not None:
        assert sha(ROOT / plan['frozen_test_selection']) == frozen_sha
    timing = np.asarray(seconds)
    assert np.isfinite(timing).all() and (timing > 0).all()
    summary = dict(sequences=len(receipts), frames=sum(row['frames'] for row in receipts), seconds=sum(seconds),
                   wall_seconds=wall_seconds, mean_ms=1000 * float(timing.mean()),
                   p95_ms=1000 * float(np.quantile(timing, .95)), fps=len(seconds) / sum(seconds),
                   image_decoding_included=True, initialization_included=True, model_loading_excluded=True,
                   trace_prediction_writing_excluded_from_service_time=True,
                   checkpoint_verification_excluded_from_service_time=True,
                   smoke_only=bool(args.smoke_frames), physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                   frozen_state_tensors=len(before), optimizer_updates=0, backward_calls=0,
                   scope='Causal predictions and complete service timing, not PR/NPR/SR until toolkit scoring; smoke is not a benchmark.')
    np.save(run / 'frame_seconds.npy', timing)
    (run / ('SMOKE.json' if args.smoke_frames else 'COMPLETE.json')).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
