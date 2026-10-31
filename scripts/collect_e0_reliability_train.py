"""Collect TRAIN reliability inputs while verifying complete direct-E0 parity."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_inputs import sha
from check_e0_observation_train import equal_state, snapshot
from dual_rgbt.data import read_pair
from dual_rgbt.e0_observation import E0Observation
from joint_evaluation_contract import check_files
from run_benchmark_tracking import build_model, build_tracker, freeze
from train_global_evidence_policy import cpu_state


def shard_sequences(names, fit, holdout, forbidden, shard, shards):
    assert len(names) == len(set(names)) == 882
    assert len(fit) == len(set(fit)) == 706 and len(holdout) == len(set(holdout)) == 176
    assert set(fit).isdisjoint(holdout) and set(fit) | set(holdout) == set(names)
    assert set(names).isdisjoint(forbidden)
    assert shards == 8 and 0 <= shard < shards
    return names[shard::shards]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    assert plan['mode'] == 'protected_e0_full_train_paired_collection'
    assert plan['full_collection_allowed'] is True and plan['training_allowed'] is False
    assert plan['test_eligible'] is False and plan['split'] == 'train'
    assert plan['cost_weight'] == 0. and plan['label_stride'] == 5
    assert plan['matmul_tf32'] is False and plan['cudnn_tf32'] is False
    assert plan['checkpoint_sha256'] == 'ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c'
    check_files(ROOT, plan['source_sha256'])
    check_files(ROOT, plan['input_sha256'])
    review_path = ROOT / plan['source_review']
    assert sha(review_path) == plan['source_review_sha256']
    review = json.loads(review_path.read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['source_sha256'] == plan['source_sha256']
    smoke_dir = ROOT / 'runs' / plan['smoke_run']
    smoke_path = smoke_dir / 'COMPLETE.json'
    assert sha(smoke_path) == plan['smoke_result_sha256']
    smoke = json.loads(smoke_path.read_text())
    smoke_config = json.loads((smoke_dir / 'config.json').read_text())
    smoke_native_path = ROOT / 'runs/pipeline_tasks' / smoke['task'] / 'status.json'
    assert sha(smoke_native_path) == plan['smoke_native_sha256']
    smoke_native = json.loads(smoke_native_path.read_text())
    assert smoke_native['state'] == 'COMPLETE' and smoke_native['exit_code'] == 0
    assert smoke_native['command'] == smoke_config['native_command']
    assert smoke['status'] == 'TRAIN_OBSERVER_PARITY_PASS'
    assert smoke['sequences'] == 3 and smoke['distinct_tracked_positions'] == 90
    assert smoke['optimizer_updates'] == smoke['backward_calls'] == 0
    assert len(smoke['rows']) == 6 and all(all(row[key] for key in (
        'raw_boxes_equal', 'route_fields_equal', 'templates_equal', 'history_equal', 'counters_equal'))
        for row in smoke['rows'])
    assert smoke['checkpoint_sha256'] == plan['checkpoint_sha256']
    assert sha(ROOT / plan['smoke_plan']) == smoke_config['plan_sha256'] == smoke['plan_sha256']
    assert json.loads((ROOT / plan['smoke_plan']).read_text()) == smoke_config['plan']
    for name, digest in smoke_config['plan']['source_sha256'].items():
        assert plan['source_sha256'][name] == digest
    assert smoke_config['plan']['cost_weight'] == 0.
    assert smoke_config['plan']['matmul_tf32'] is False and smoke_config['plan']['cudnn_tf32'] is False
    for name, digest in smoke['artifacts_sha256'].items():
        assert sha(smoke_dir / name) == digest
    native = json.loads((ROOT / 'runs/pipeline_tasks' / args.task / 'status.json').read_text())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['command'] == [sys.executable] + sys.argv
    assert native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID' and os.environ['OPENCV_FOR_THREADS_NUM'] == '1'
    manifest = json.loads((ROOT / plan['manifest']).read_text())
    names = [s['name'] for s in manifest['train']]
    forbidden = {s['name'] for s in manifest['dev']} | set(manifest['test_sequences'])
    selected = shard_sequences(names, plan['fit_sequences'], plan['holdout_sequences'],
        forbidden, args.shard, plan['shards'])
    by_name = {s['name']: s for s in manifest['train']}
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = build_model('e0', None)
    model.load_state_dict(torch.load(ROOT / plan['checkpoint'], map_location='cpu')['net'], strict=True)
    freeze(model)
    before = cpu_state(model)
    assert sum(p.numel() for p in model.parameters()) == 103374215
    observer = E0Observation(model)
    directory = ROOT / 'runs' / args.run
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'predictions').mkdir()
    config = dict(plan=plan, plan_sha256=sha(plan_path), native_command=native['command'],
        task=args.task, run=args.run, shard=args.shard, selected_sequences=selected, split='train')
    (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    progress, started = [], time.time()
    fit = set(plan['fit_sequences'])
    for name in selected:
        sequence = by_name[name]
        original, observed = [build_tracker('e0', model, 0., 0) for _ in range(2)]
        image = read_pair(sequence, 0)
        original.initialize(image, sequence['boxes'][0])
        observed.initialize(image, sequence['boxes'][0])
        equal_state(snapshot(original), snapshot(observed))
        predictions = [observed.state.copy()]
        valid, features, labels = set(sequence['valid_frames']), [], []
        frame_started = time.time()
        for frame in range(1, len(sequence['rgb'])):
            image = read_pair(sequence, frame)
            # The control bypasses the observer completely, with its own causal state.
            model.predict = observer.original
            expected_box, expected_route = original.track(image)
            model.predict = observer.predict
            sample = frame % 5 == 0 and frame in valid
            observer.begin(sample)
            box, route = observed.track(image)
            assert observer.calls == 1 and np.array_equal(box, expected_box)
            assert {k: v for k, v in route.items() if k != 'seconds'} == {
                k: v for k, v in expected_route.items() if k != 'seconds'}
            equal_state(snapshot(original), snapshot(observed))
            predictions.append(box.copy())
            if sample:
                # The target is not consulted until both causal updates have finished.
                gt = np.asarray(sequence['boxes'][frame], dtype=np.float64)
                assert np.isfinite(gt).all() and (gt[2:] > 0).all()
                left, right = np.maximum(box[:2], gt[:2]), np.minimum(box[:2] + box[2:], gt[:2] + gt[2:])
                intersection = np.maximum(0, right - left).prod()
                iou = float(intersection / (box[2:].prod() + gt[2:].prod() - intersection))
                assert np.isfinite(iou) and 0 <= iou <= 1
                features.append(observer.record['features'])
                labels.append(dict(sequence=name, partition='fit' if name in fit else 'holdout',
                    frame=frame, final_iou=iou, modality=route['modality'], action=route['action'],
                    confidence=route['confidence'], primary_roi_coverage=observer.record['primary_roi_coverage']))
            if frame % 100 == 0:
                (directory / 'current.json').write_text(json.dumps(dict(
                    sequence=name, frame=frame, frames=len(sequence['rgb']), updated=time.time())))
        assert features and np.isfinite(np.asarray(features)).all()
        np.save(directory / (name + '_features.npy'), np.asarray(features, dtype=np.float32))
        np.save(directory / 'predictions' / (name + '.npy'), np.asarray(predictions))
        with (directory / (name + '_labels.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, list(labels[0]))
            writer.writeheader()
            writer.writerows(labels)
        progress.append(dict(sequence=name, frames=len(sequence['rgb']), feature_states=len(labels),
            matched_tracked_frames=len(sequence['rgb']) - 1, all_online_states_bit_equal=True,
            wall_seconds=time.time() - frame_started))
        (directory / 'progress.json').write_text(json.dumps(progress, indent=2))
        print(json.dumps(progress[-1]), flush=True)
    observer.close()
    assert all(torch.equal(before[k], v) for k, v in cpu_state(model).items())
    assert all(p.grad is None for p in model.parameters())
    check_files(ROOT, plan['source_sha256'])
    check_files(ROOT, plan['input_sha256'])
    assert sha(plan_path) == config['plan_sha256']
    receipt = dict(status='PAIRED_TRAIN_CACHE_COMPLETE', task=args.task, split='train',
        checkpoint_sha256=plan['checkpoint_sha256'], plan_sha256=config['plan_sha256'],
        sequences=len(progress), frames=sum(row['frames'] for row in progress),
        tracked_frames=sum(row['matched_tracked_frames'] for row in progress),
        feature_states=sum(row['feature_states'] for row in progress), feature_width=914,
        optimizer_updates=0, backward_calls=0, frozen_state_tensors=len(before),
        all_online_states_bit_equal=True, wall_seconds=time.time() - started,
        total_parameters=103374215, test_eligible=False,
        artifacts_sha256={p.relative_to(directory).as_posix(): sha(p)
                          for p in directory.rglob('*') if p.is_file()},
        scope='Complete paired original/observed E0 TRAIN shard; same FP32/TF32-disabled cost0 path. '
              'No recovery branch or trained module; component holdout was seen by original E0 training. '
              'Not historical TF32 reproduction, benchmark metrics or independent generalization proof.')
    (directory / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
