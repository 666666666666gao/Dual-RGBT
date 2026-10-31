"""Verify protected E0 observer parity using TRAIN only, without training."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_inputs import sha
from dual_rgbt.data import read_pair
from dual_rgbt.e0_observation import E0Observation
from joint_evaluation_contract import check_files
from run_benchmark_tracking import build_model, build_tracker, freeze
from train_global_evidence_policy import cpu_state


def snapshot(tracker):
    return dict(box=tracker.state.copy(), templates=tracker.templates.cpu().clone(),
        history=tracker.history.cpu().clone(), frame=tracker.frame, since_query=tracker.since_query)


def equal_state(actual, expected):
    assert np.array_equal(actual['box'], expected['box'])
    assert torch.equal(actual['templates'], expected['templates'])
    assert torch.equal(actual['history'], expected['history'])
    assert actual['frame'] == expected['frame'] and actual['since_query'] == expected['since_query']


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    assert plan['mode'] == 'protected_e0_train_observer_parity'
    assert plan['split'] == 'train' and plan['training_allowed'] is False
    assert plan['checkpoint_sha256'] == 'ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c'
    assert plan['frames_per_sequence'] == 31 and len(plan['sequence_names']) == 3
    assert len(set(plan['sequence_names'])) == 3 and plan['cost_weight'] == 0.
    assert plan['matmul_tf32'] is False and plan['cudnn_tf32'] is False
    check_files(ROOT, plan['source_sha256'])
    check_files(ROOT, plan['input_sha256'])
    review_path = ROOT / plan['source_review']
    assert sha(review_path) == plan['source_review_sha256']
    review = json.loads(review_path.read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['source_sha256'] == plan['source_sha256']
    native_path = ROOT / 'runs/pipeline_tasks' / args.task / 'status.json'
    native = json.loads(native_path.read_text())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['command'] == [sys.executable] + sys.argv
    assert native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID' and os.environ['OPENCV_FOR_THREADS_NUM'] == '1'
    manifest = json.loads((ROOT / plan['manifest']).read_text())
    by_name = {s['name']: s for s in manifest['train']}
    forbidden = {s['name'] for s in manifest['dev']} | set(manifest['test_sequences'])
    assert len(by_name) == 882 and not set(plan['sequence_names']) & forbidden
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = build_model('e0', None)
    model.load_state_dict(torch.load(ROOT / plan['checkpoint'], map_location='cpu')['net'], strict=True)
    freeze(model)
    before = cpu_state(model)
    assert sum(p.numel() for p in model.parameters()) == 103374215
    directory = ROOT / 'runs' / args.run
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(plan=plan, plan_sha256=sha(plan_path), task=args.task,
        native_command=native['command'], run=args.run, split='train')
    (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    rows, features, labels = [], [], []
    for name in plan['sequence_names']:
        sequence = by_name[name]
        count = plan['frames_per_sequence']
        assert len(sequence['rgb']) >= count
        baseline = build_tracker('e0', model, 0., 0)
        baseline.initialize(read_pair(sequence, 0), sequence['boxes'][0])
        control = []
        for frame in range(1, count):
            box, route = baseline.track(read_pair(sequence, frame))
            control.append((snapshot(baseline), {k: v for k, v in route.items() if k != 'seconds'}))
        observer = E0Observation(model)
        for enabled in (False, True):
            tracker = build_tracker('e0', model, 0., 0)
            tracker.initialize(read_pair(sequence, 0), sequence['boxes'][0])
            for frame in range(1, count):
                observer.begin(enabled)
                box, route = tracker.track(read_pair(sequence, frame))
                state, expected_route = control[frame - 1]
                equal_state(snapshot(tracker), state)
                assert {k: v for k, v in route.items() if k != 'seconds'} == expected_route
                assert observer.calls == 1
                if enabled and frame in set(sequence['valid_frames']):
                    # TRAIN labels are first read after the unchanged online update.
                    gt = np.asarray(sequence['boxes'][frame], dtype=np.float64)
                    left = np.maximum(box[:2], gt[:2])
                    right = np.minimum(box[:2] + box[2:], gt[:2] + gt[2:])
                    intersection = np.maximum(0, right - left).prod()
                    iou = float(intersection / (box[2:].prod() + gt[2:].prod() - intersection))
                    assert np.isfinite(iou) and 0 <= iou <= 1
                    features.append(observer.record['features'])
                    labels.append(dict(sequence=name, frame=frame, final_iou=iou,
                        modality=route['modality'], action=route['action'],
                        confidence=route['confidence'], primary_roi_coverage=observer.record['primary_roi_coverage']))
            rows.append(dict(sequence=name, observer_enabled=enabled, tracked_frames=count - 1,
                raw_boxes_equal=True, route_fields_equal=True, templates_equal=True,
                history_equal=True, counters_equal=True))
        observer.close()
        print(json.dumps(rows[-2:]), flush=True)
    assert len(labels) and np.isfinite(np.asarray(features)).all()
    assert all(torch.equal(before[k], v) for k, v in cpu_state(model).items())
    assert all(p.grad is None for p in model.parameters())
    check_files(ROOT, plan['source_sha256'])
    check_files(ROOT, plan['input_sha256'])
    assert sha(plan_path) == config['plan_sha256']
    np.save(directory / 'features.npy', np.asarray(features, dtype=np.float32))
    with (directory / 'labels.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    result = dict(status='TRAIN_OBSERVER_PARITY_PASS', task=args.task,
        plan_sha256=config['plan_sha256'], checkpoint_sha256=plan['checkpoint_sha256'],
        sequences=3, frames_per_sequence=count, distinct_tracked_positions=90,
        rows=rows, feature_states=len(labels), feature_width=914, total_parameters=103374215,
        optimizer_updates=0, backward_calls=0, frozen_state_tensors=len(before),
        full_trajectory_parity_proven=False, historical_tf32_reproduction_proven=False,
        test_eligible=False, split='train',
        artifacts_sha256={p.name: sha(p) for p in directory.iterdir() if p.is_file()},
        scope='Three TRAIN prefixes; direct unchanged E0 compared with observer disabled and enabled. '
              'FP32/TF32-disabled reference settings, not historical high-score TF32 reproduction. '
              'No recovery branch, trained reliability module, benchmark metric or full-trajectory proof.')
    (directory / 'COMPLETE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
