"""Train-only causal replay: observe gain inputs, then label sampled queries."""
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
from dual_rgbt.data import read_pair
from dual_rgbt_evidence.readout import pixel_xywh
from dual_rgbt_evidence.regions import crop_region, execute_boxes, pixel_iou
from joint_evaluation_contract import check_files, evaluation_plan
from run_benchmark_tracking import build_model, build_tracker, freeze
from train_global_evidence_policy import cpu_state


class GainObservation:
    """Observe the deployed gain call without changing inputs or return values."""
    def __init__(self, model):
        self.enabled = False
        self.original = model.candidate_gains
        model.candidate_gains = self.gains
        self.handle = model.query_gain.register_forward_pre_hook(self.features)

    def features(self, module, inputs):
        if self.enabled:
            assert len(inputs) == 1
            self.inputs = inputs[0].detach().clone()

    def gains(self, primary, preview, modality, main, broad, queries, kinds):
        result = self.original(primary, preview, modality, main, broad, queries, kinds)
        if self.enabled:
            self.calls += 1
            self.primary, self.modality, self.main = primary, modality, main
            self.queries, self.kinds, self.predictions = queries, kinds, result
            assert self.inputs.shape == (1, queries.shape[1], 955)
        return result

    def begin(self, enabled):
        self.enabled, self.calls = enabled, 0


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
    assert plan['mode'] == 'frozen_train_query_cache' and plan['test_eligible'] is False
    assert plan['training_allowed'] is False
    check_files(ROOT, plan['source_sha256'])
    parent, _, _ = evaluation_plan(ROOT, plan['parent_evaluation_plan'])
    assert parent['checkpoint_sha256'] == plan['checkpoint_sha256']
    if not args.smoke:
        assert plan['full_collection_allowed'] is True
        smoke_path = ROOT / plan['smoke_result']
        assert sha(smoke_path) == plan['smoke_result_sha256']
        smoke = json.loads(smoke_path.read_text())
        assert smoke['status'] == 'SMOKE_PASS' and smoke['observer_parent_trajectory_equal']
        assert smoke['checkpoint_sha256'] == plan['checkpoint_sha256']
        assert smoke['frames'] == plan['smoke_frames'] and smoke['sequences'] == 1 and smoke['labels'] > 0
        assert smoke['optimizer_updates'] == smoke['backward_calls'] == 0 and smoke['teacher_preserves_tracker_state']
        smoke_native_path = ROOT / plan['smoke_native_status']
        assert sha(smoke_native_path) == plan['smoke_native_sha256']
        smoke_native = json.loads(smoke_native_path.read_text())
        assert smoke_native['state'] == 'COMPLETE' and smoke_native['exit_code'] == 0
        smoke_config = json.loads((smoke_path.parent / 'config.json').read_text())
        assert smoke_config['smoke'] is True and smoke_config['native_command'] == smoke_native['command']
        assert smoke_config['source_sha256'] == dict(parent['source_sha256'], **plan['source_sha256'])
        assert sha(ROOT / plan['smoke_plan']) == smoke_config['plan_sha256'] == smoke['plan_sha256']
        assert json.loads((ROOT / plan['smoke_plan']).read_text()) == smoke_config['plan']
        for key in ('mode', 'parent_evaluation_plan', 'checkpoint_sha256', 'manifest', 'manifest_sha256',
                    'source_sha256', 'seed', 'cost_weight', 'label_stride', 'smoke_frames',
                    'fit_sequences', 'holdout_sequences', 'training_allowed', 'test_eligible'):
            assert smoke_config['plan'][key] == plan[key]
        command = smoke_native['command']
        assert '--smoke' in command and command[command.index('--task') + 1] == smoke_native['task']
        assert command[command.index('--run') + 1] == smoke_path.parent.name
        assert all(sha(smoke_path.parent / name) == digest for name, digest in smoke['artifacts_sha256'].items())
    native_path = ROOT / 'runs/pipeline_tasks' / args.task / 'status.json'
    native = json.loads(native_path.read_text())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['command'] == [sys.executable] + sys.argv
    assert native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID' and os.environ['OPENCV_FOR_THREADS_NUM'] == '1'
    manifest_path = ROOT / plan['manifest']
    assert sha(manifest_path) == plan['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    names = plan['fit_sequences'] + plan['holdout_sequences']
    by_name = {s['name']: s for s in manifest['train']}
    assert len(by_name) == 882 and len(names) == len(set(names))
    assert set(names) <= set(by_name)
    assert not set(names) & ({s['name'] for s in manifest['dev']} | set(manifest['test_sequences']))
    assert plan['cost_weight'] == .15 and parent['max_candidates'] == 8
    selected_names = names[:1] if args.smoke else names
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_model('e1', ROOT / parent['factory_checkpoint'])
    model.load_state_dict(torch.load(ROOT / parent['checkpoint'], map_location='cpu')['net'], strict=True)
    freeze(model)
    before = cpu_state(model)
    control = []
    if args.smoke:
        sequence = by_name[selected_names[0]]
        baseline = build_tracker('e1', model, plan['cost_weight'], 8)
        baseline.initialize(read_pair(sequence, 0), sequence['boxes'][0])
        for frame in range(1, plan['smoke_frames']):
            box, route = baseline.track(read_pair(sequence, frame))
            control.append((box.copy(), route['action'], route['modality'],
                            baseline.templates.clone(), baseline.history.clone()))
    observation = GainObservation(model)
    tracker = build_tracker('e1', model, plan['cost_weight'], 8)
    config = dict(plan=plan, plan_sha256=sha(plan_path), native_command=native['command'],
        checkpoint_sha256=parent['checkpoint_sha256'], smoke=args.smoke,
        selected_sequences=selected_names, split='train', optimizer_updates=0,
        source_sha256=dict(parent['source_sha256'], **plan['source_sha256']))
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    rng = np.random.RandomState(plan['seed'])
    started, progress = time.time(), []
    torch.cuda.reset_peak_memory_stats()
    for name in selected_names:
        sequence = by_name[name]
        count = plan['smoke_frames'] if args.smoke else len(sequence['rgb'])
        assert count <= len(sequence['rgb'])
        valid = set(sequence['valid_frames'])
        tracker.initialize(read_pair(sequence, 0), sequence['boxes'][0])
        rows, features, sampled_states = [], [], 0
        for frame in range(1, count):
            label = frame % plan['label_stride'] == 0 and frame in valid
            image = read_pair(sequence, frame)
            old_templates = tracker.templates.clone() if label else None
            observation.begin(label)
            executed, route = tracker.track(image)
            if args.smoke:
                reference = control[frame - 1]
                assert np.array_equal(executed, reference[0])
                assert (route['action'], route['modality']) == reference[1:3]
                assert torch.equal(tracker.templates, reference[3]) and torch.equal(tracker.history, reference[4])
            observation.enabled = False
            if not label:
                continue
            sampled_states += 1
            assert observation.calls == 1 and observation.modality == route['modality']
            assert torch.equal(model.query_gain(observation.inputs).squeeze(2), observation.predictions[:, 1:])
            state_after, templates_after = tracker.state.copy(), tracker.templates.clone()
            history_after, frame_after, since_after = tracker.history.clone(), tracker.frame, tracker.since_query
            kinds = observation.kinds[0].cpu().numpy()
            choices = []
            for kind in range(4):
                eligible = np.flatnonzero(kinds == kind)
                choices.append((int(rng.choice(eligible)), 'sampled', 1.0 / len(eligible)))
            selected = route['request_index']
            if selected >= 0 and selected not in [x[0] for x in choices]:
                choices.append((selected, 'selected_extra', 1.0))
            size = observation.main.new_tensor([image.shape[:2]])
            # First annotation read after the online state and query choices are fixed.
            target = observation.main.new_tensor(sequence['boxes'][frame])[None]
            main_box = execute_boxes(pixel_xywh(observation.primary['box'], observation.main), size)
            main_iou = float(pixel_iou(main_box, target).item())
            if selected < 0:
                assert np.array_equal(main_box[0].cpu().numpy(), executed)
            for index, purpose, probability in choices:
                kind = int(kinds[index])
                geometry = observation.queries[:, index]
                pixels = crop_region(image, geometry[0].cpu().numpy(), 256 if kind == 3 else 128, image.mean((0, 1)))[None].cuda()
                query = model.query(old_templates, pixels, observation.primary, observation.main, geometry, observation.modality)
                box = execute_boxes(query['pixel_box'], size)
                if index == selected:
                    assert np.array_equal(box[0].cpu().numpy(), executed)
                query_iou = float(pixel_iou(box, target).item())
                assert np.isfinite([main_iou, query_iou]).all() and 0 <= main_iou <= 1 and 0 <= query_iou <= 1
                assert np.isfinite(box.cpu().numpy()).all() and (box[:, 2:] > 0).all()
                features.append(observation.inputs[0, index].cpu().numpy().copy())
                rows.append(dict(sequence=name, partition='fit' if name in plan['fit_sequences'] else 'holdout',
                    frame=frame, modality=observation.modality, kind=kind, request_index=index,
                    purpose=purpose, sampling_probability=probability, selected=int(index == selected),
                    deployed_action=route['action'], primary_iou=main_iou, query_iou=query_iou,
                    gain=query_iou-main_iou, predicted_gain=float(observation.predictions[0, index+1].item()),
                    primary_confidence=float(observation.primary['score'].max().item()),
                    query_confidence=float(query['score'].max().item())))
            assert np.array_equal(tracker.state, state_after)
            assert torch.equal(tracker.templates, templates_after) and torch.equal(tracker.history, history_after)
            assert tracker.frame == frame_after and tracker.since_query == since_after
        assert rows and np.isfinite(np.asarray(features)).all()
        np.save(run / (name + '_features.npy'), np.asarray(features, dtype=np.float32))
        with (run / (name + '_labels.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        progress.append(dict(sequence=name, frames=count, sampled_states=sampled_states, labels=len(rows)))
        (run / 'progress.json').write_text(json.dumps(progress, indent=2))
        print(json.dumps(progress[-1]), flush=True)
    observation.handle.remove()
    assert all(torch.equal(before[k], v) for k, v in cpu_state(model).items())
    assert all(p.grad is None for p in model.parameters())
    check_files(ROOT, config['source_sha256'])
    check_files(ROOT, parent['input_sha256'])
    assert sha(manifest_path) == plan['manifest_sha256'] and sha(plan_path) == config['plan_sha256']
    receipt = dict(status='SMOKE_PASS' if args.smoke else 'CACHE_COMPLETE',
        sequences=len(progress), frames=sum(p['frames'] for p in progress), labels=sum(p['labels'] for p in progress),
        optimizer_updates=0, backward_calls=0, frozen_state_tensors=len(before),
        observer_parent_trajectory_equal=True if args.smoke else None, teacher_preserves_tracker_state=True,
        checkpoint_sha256=parent['checkpoint_sha256'], plan_sha256=config['plan_sha256'],
        wall_seconds=time.time()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), test_eligible=False,
        artifacts_sha256={p.name: sha(p) for p in run.iterdir() if p.is_file()},
        scope='Frozen-parent train-only sampled query features/labels; no trained model or benchmark accuracy.')
    (run / ('SMOKE.json' if args.smoke else 'COMPLETE.json')).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
