"""Refresh TRAIN query labels on a frozen failed-policy trajectory."""
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
from collect_frozen_query_features import GainObservation
from dual_rgbt.data import read_pair
from dual_rgbt_evidence.readout import pixel_xywh
from dual_rgbt_evidence.regions import crop_region, execute_boxes, pixel_iou
from joint_evaluation_contract import check_files, evaluation_plan
from run_benchmark_tracking import build_model, build_tracker, freeze
from train_global_evidence_policy import cpu_state


def snapshot(tracker):
    return (tracker.state.copy(), tracker.templates.clone(), tracker.history.clone(),
            tracker.frame, tracker.since_query)


def equal_state(tracker, saved):
    assert np.array_equal(tracker.state, saved[0])
    assert torch.equal(tracker.templates, saved[1]) and torch.equal(tracker.history, saved[2])
    assert (tracker.frame, tracker.since_query) == saved[3:]


def choices(kinds, selected, rng):
    result = [(int(rng.choice(np.flatnonzero(kinds == kind))), 'sampled',
               1.0 / int((kinds == kind).sum())) for kind in range(4)]
    if selected >= 0 and selected not in [item[0] for item in result]:
        result.append((selected, 'selected_extra', 1.0))
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    assert plan['mode'] == 'frozen_policy_train_refresh'
    assert plan['training_allowed'] is plan['test_eligible'] is False
    check_files(ROOT, plan['source_sha256'])
    check_files(ROOT, plan['input_sha256'])
    parent, _, _ = evaluation_plan(ROOT, plan['parent_evaluation_plan'])
    assert parent['checkpoint_sha256'] == plan['parent_checkpoint_sha256']
    overlay = torch.load(ROOT / plan['overlay'], map_location='cpu')
    assert overlay['stage'] == 'cached_query_gain'
    assert overlay['config']['parent_checkpoint_sha256'] == parent['checkpoint_sha256']
    native_training = json.loads((ROOT / plan['training_native_status']).read_text())
    trained = json.loads((ROOT / plan['training_result']).read_text())
    assert native_training['state'] == 'COMPLETE' and native_training['exit_code'] == 0
    assert native_training['command'] == overlay['config']['native_command']
    assert trained['status'] == 'TRAIN_COMPLETE_PENDING_DEV' and trained['strict_reload_bit_equal']
    assert dict(epoch=overlay['epoch'], checkpoint=plan['overlay'],
                sha256=plan['overlay_sha256']) in trained['retained']
    assert sha(ROOT / plan['overlay']) == plan['overlay_sha256']
    if not args.smoke:
        assert plan['full_collection_allowed'] is True
        smoke_path, smoke_native = [ROOT / plan[key] for key in ('smoke_result', 'smoke_native_status')]
        assert sha(smoke_path) == plan['smoke_result_sha256']
        assert sha(smoke_native) == plan['smoke_native_sha256']
        result, terminal = [json.loads(path.read_text()) for path in (smoke_path, smoke_native)]
        assert terminal['state'] == 'COMPLETE' and terminal['exit_code'] == 0
        assert result['status'] == 'SMOKE_PASS' and result['observer_policy_trajectory_equal']
        assert result['source_sha256'] == plan['source_sha256']
        assert result['checkpoint_sha256'] == plan['overlay_sha256']
        smoke_config = json.loads((smoke_path.parent / 'config.json').read_text())
        assert smoke_config['native_command'] == terminal['command']
        for key in ('source_sha256', 'input_sha256', 'manifest', 'manifest_sha256', 'overlay',
                    'fit_sequences', 'holdout_sequences', 'label_stride', 'seed', 'cost_weight'):
            assert smoke_config['plan'][key] == plan[key]
        check_files(smoke_path.parent, result['artifacts_sha256'])
    native = json.loads((ROOT / 'runs/pipeline_tasks' / args.task / 'status.json').read_text())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['command'] == [sys.executable] + sys.argv
    assert native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID' and os.environ['OPENCV_FOR_THREADS_NUM'] == '1'
    manifest = json.loads((ROOT / plan['manifest']).read_text())
    assert sha(ROOT / plan['manifest']) == plan['manifest_sha256']
    by_name = {row['name']: row for row in manifest['train']}
    names = plan['fit_sequences'] + plan['holdout_sequences']
    assert len(by_name) == 882 and len(names) == len(set(names))
    assert set(names) <= set(by_name)
    assert not set(names) & ({row['name'] for row in manifest['dev']} | set(manifest['test_sequences']))
    assert plan['cost_weight'] == .15 and parent['max_candidates'] == 8
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    state = torch.load(ROOT / parent['checkpoint'], map_location='cpu')['net']
    state.update({'query_gain.' + key: value for key, value in overlay['head'].items()})
    model = build_model('e1', ROOT / parent['factory_checkpoint'])
    model.load_state_dict(state, strict=True)
    del state, overlay
    freeze(model)
    frozen = cpu_state(model)
    selected_names = names[:1] if args.smoke else names
    control = []
    if args.smoke:
        row = by_name[selected_names[0]]
        tracker = build_tracker('e1', model, .15, 8)
        tracker.initialize(read_pair(row, 0), row['boxes'][0])
        for frame in range(1, plan['smoke_frames']):
            box, route = tracker.track(read_pair(row, frame))
            control.append((snapshot(tracker), route['action'], route['modality']))
    observer = GainObservation(model)
    tracker = build_tracker('e1', model, .15, 8)
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    config = dict(plan=plan, plan_sha256=sha(plan_path), task=args.task, native_command=native['command'],
                  checkpoint_sha256=plan['overlay_sha256'], parent_checkpoint_sha256=parent['checkpoint_sha256'],
                  smoke=args.smoke, split='train', selected_sequences=selected_names,
                  source_sha256=dict(parent['source_sha256'], **plan['source_sha256']))
    (run / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    rng = np.random.RandomState(plan['seed'])
    started, progress = time.time(), []
    torch.cuda.reset_peak_memory_stats()
    for name in selected_names:
        row = by_name[name]
        count = plan['smoke_frames'] if args.smoke else len(row['rgb'])
        valid = set(row['valid_frames'])
        tracker.initialize(read_pair(row, 0), row['boxes'][0])
        features, labels, sampled_states = [], [], 0
        for frame in range(1, count):
            label = frame % plan['label_stride'] == 0 and frame in valid
            image = read_pair(row, frame)
            templates = tracker.templates.clone() if label else None
            observer.begin(label)
            executed, route = tracker.track(image)
            if args.smoke:
                equal_state(tracker, control[frame - 1][0])
                assert (route['action'], route['modality']) == control[frame - 1][1:]
            observer.enabled = False
            if not label:
                continue
            assert observer.calls == 1 and observer.modality == route['modality']
            assert torch.equal(model.query_gain(observer.inputs).squeeze(2), observer.predictions[:, 1:])
            sampled_states += 1
            saved = snapshot(tracker)
            kinds = observer.kinds[0].cpu().numpy()
            selected = route['request_index']
            sampled = choices(kinds, selected, rng)
            size = observer.main.new_tensor([image.shape[:2]])
            # No GT is read until the deployed decision and sampling are fixed.
            target = observer.main.new_tensor(row['boxes'][frame])[None]
            main_box = execute_boxes(pixel_xywh(observer.primary['box'], observer.main), size)
            main_iou = float(pixel_iou(main_box, target).item())
            if selected < 0:
                assert np.array_equal(main_box[0].cpu().numpy(), executed)
            for index, purpose, probability in sampled:
                kind, geometry = int(kinds[index]), observer.queries[:, index]
                pixels = crop_region(image, geometry[0].cpu().numpy(),
                                     256 if kind == 3 else 128, image.mean((0, 1)))[None].cuda()
                output = model.query(templates, pixels, observer.primary, observer.main,
                                     geometry, observer.modality)
                box = execute_boxes(output['pixel_box'], size)
                if selected == index:
                    assert np.array_equal(box[0].cpu().numpy(), executed)
                query_iou = float(pixel_iou(box, target).item())
                assert np.isfinite([main_iou, query_iou]).all() and 0 <= main_iou <= 1 and 0 <= query_iou <= 1
                assert np.isfinite(box.cpu().numpy()).all() and (box[:, 2:] > 0).all()
                features.append(observer.inputs[0, index].cpu().numpy().copy())
                labels.append(dict(sequence=name, partition='fit' if name in plan['fit_sequences'] else 'holdout',
                    frame=frame, modality=observer.modality, kind=kind, request_index=index,
                    purpose=purpose, sampling_probability=probability, selected=int(index == selected),
                    deployed_action=route['action'], primary_iou=main_iou, query_iou=query_iou,
                    gain=query_iou-main_iou, predicted_gain=float(observer.predictions[0, index+1].item()),
                    primary_confidence=float(observer.primary['score'].max().item()),
                    query_confidence=float(output['score'].max().item())))
            equal_state(tracker, saved)
        assert labels and np.isfinite(np.asarray(features)).all()
        np.save(run / (name + '_features.npy'), np.asarray(features, dtype=np.float32))
        with (run / (name + '_labels.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, list(labels[0]))
            writer.writeheader()
            writer.writerows(labels)
        progress.append(dict(sequence=name, frames=count, sampled_states=sampled_states, labels=len(labels)))
        (run / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
        print(json.dumps(progress[-1]), flush=True)
    observer.handle.remove()
    assert all(torch.equal(frozen[key], value) for key, value in cpu_state(model).items())
    assert all(parameter.grad is None for parameter in model.parameters())
    check_files(ROOT, config['source_sha256'])
    check_files(ROOT, plan['input_sha256'])
    check_files(ROOT, parent['input_sha256'])
    assert sha(plan_path) == config['plan_sha256']
    result = dict(status='SMOKE_PASS' if args.smoke else 'CACHE_COMPLETE', sequences=len(progress),
                  frames=sum(row['frames'] for row in progress), labels=sum(row['labels'] for row in progress),
                  checkpoint_sha256=plan['overlay_sha256'], parent_checkpoint_sha256=parent['checkpoint_sha256'],
                  frozen_state_tensors=len(frozen), optimizer_updates=0, backward_calls=0,
                  observer_policy_trajectory_equal=True if args.smoke else None,
                  teacher_preserves_tracker_state=True, source_sha256=plan['source_sha256'],
                  plan_sha256=config['plan_sha256'], wall_seconds=time.time()-started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), test_eligible=False,
                  artifacts_sha256={path.name: sha(path) for path in run.iterdir() if path.is_file()},
                  scope='Frozen learned-policy TRAIN state refresh; uniform four-family samples, not exhaustive candidate recovery recall. No new model or benchmark claim.')
    (run / ('SMOKE.json' if args.smoke else 'COMPLETE.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
