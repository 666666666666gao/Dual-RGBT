"""Observe a frozen dev trajectory and evaluate every query at sampled states."""
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
from benchmark_inputs import load_manifest, sha
from dual_rgbt.data import read_pair
from dual_rgbt_evidence.readout import pixel_xywh
from dual_rgbt_evidence.regions import crop_region, execute_boxes, pixel_iou
import dual_rgbt_global.candidate_tracker as tracker_module
from joint_evaluation_contract import check_files, evaluation_plan
from run_benchmark_tracking import build_model, build_tracker, freeze
from train_global_evidence_policy import cpu_state


class Observation:
    """Return native outputs unchanged; observe the executed template crop."""

    def __init__(self, model):
        self.enabled = False
        self.model = model
        self.original_preview = model.preview_state
        self.original_gains = model.candidate_gains
        self.original_crop = tracker_module.crop_pair
        model.preview_state = self.preview
        model.candidate_gains = self.gains
        tracker_module.crop_pair = self.crop

    def preview(self, *args):
        result = self.original_preview(*args)
        if self.enabled:
            self.preview_output = result
        return result

    def gains(self, primary, preview, modality, main, broad, queries, kinds):
        result = self.original_gains(primary, preview, modality, main, broad, queries, kinds)
        if self.enabled:
            self.gain_calls += 1
            self.primary, self.modality = primary, modality
            self.main, self.broad = main, broad
            self.queries, self.kinds, self.predictions = queries, kinds, result
        return result

    def crop(self, *args, **kwargs):
        result = self.original_crop(*args, **kwargs)
        if self.enabled:
            self.template_crops.append(result[0].detach().clone())
        return result

    def begin(self):
        self.gain_calls = 0
        self.template_crops = []
        self.enabled = True


def snapshot(tracker):
    return dict(state=tracker.state.copy(), templates=tracker.templates.clone(),
                history=tracker.history.clone(), frame=tracker.frame,
                since_query=tracker.since_query)


def same_state(tracker, saved):
    assert np.array_equal(tracker.state, saved['state'])
    assert torch.equal(tracker.templates, saved['templates'])
    assert torch.equal(tracker.history, saved['history'])
    assert tracker.frame == saved['frame'] and tracker.since_query == saved['since_query']


def box_metrics(box, target):
    prediction = np.asarray(box, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    valid = bool(np.isfinite(truth).all() and (truth[2:] > 0).all())
    if not valid:
        return dict(valid_gt=False, iou=None, center_error=None, area_ratio=None,
                    width_ratio=None, height_ratio=None)
    left = np.maximum(prediction[:2], truth[:2])
    right = np.minimum(prediction[:2] + prediction[2:], truth[:2] + truth[2:])
    intersection = np.maximum(0, right - left).prod()
    union = prediction[2:].prod() + truth[2:].prod() - intersection
    return dict(valid_gt=True, iou=float(intersection / union),
                center_error=float(np.linalg.norm(prediction[:2] + prediction[2:] / 2 -
                                                  truth[:2] - truth[2:] / 2)),
                area_ratio=float(prediction[2:].prod() / truth[2:].prod()),
                width_ratio=float(prediction[2] / truth[2]),
                height_ratio=float(prediction[3] / truth[3]))


def template_event(observation, tracker, before, route):
    expected = before['templates'].clone()
    writes = []
    if observation.template_crops:
        assert len(observation.template_crops) == 1
        writes = [route['modality']]
        if route['action']:
            writes.append(1 - route['modality'])
        updated = observation.template_crops[0]
        for modality in writes:
            channels = slice(modality * 3, (modality + 1) * 3)
            expected[:, 1, channels] = updated[channels]
    assert torch.equal(tracker.templates, expected)
    return writes


@torch.no_grad()
def candidates(model, observation, image, before, tracker, route):
    saved = snapshot(tracker)
    size = observation.main.new_tensor([image.shape[:2]])
    main_box = execute_boxes(pixel_xywh(observation.primary['box'], observation.main), size)[0]
    costs = model.candidate_costs(observation.kinds)[0]
    results = []
    for index in range(observation.queries.shape[1]):
        kind = int(observation.kinds[0, index].item())
        geometry = observation.queries[:, index]
        pixels = crop_region(image, geometry[0].cpu().numpy(),
                             256 if kind == 3 else 128, image.mean((0, 1)))[None].cuda()
        output = model.query(before['templates'], pixels, observation.primary, observation.main,
                             geometry, observation.modality)
        box = execute_boxes(output['pixel_box'], size)[0]
        if index == route['request_index']:
            assert np.array_equal(box.cpu().numpy(), tracker.state)
        results.append(dict(request_index=index, action=kind + 1,
                            selected=int(index == route['request_index']),
                            predicted_gain=float(observation.predictions[0, index + 1].item()),
                            token_cost=float(costs[index + 1].item()),
                            box=box.cpu().tolist(),
                            confidence=float(output['score'].max().item()),
                            geometry=geometry[0].cpu().tolist()))
    means = image.mean((0, 1))
    main_image = crop_region(image, observation.main[0].cpu().numpy(), 256, means)[None].cuda()
    other = model.executor.primary(before['templates'], main_image,
                                   1 - observation.modality, tracker.window)
    other_box = execute_boxes(pixel_xywh(other['box'], observation.main), size)[0]
    same_state(tracker, saved)
    return main_box.cpu().tolist(), results, other_box.cpu().tolist(), float(other['score'].max().item())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    assert plan['mode'] == 'fixed_state_dev_diagnostic'
    assert plan['training_allowed'] is plan['test_eligible'] is False
    check_files(ROOT, plan['source_sha256'])
    parent, spec, _ = evaluation_plan(ROOT, plan['parent_evaluation_plan'])
    assert parent['checkpoint_sha256'] == plan['parent_checkpoint_sha256']
    native = json.loads((ROOT / 'runs/pipeline_tasks' / args.task / 'status.json').read_text())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['command'] == [sys.executable] + sys.argv
    assert native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    manifest_path = ROOT / spec['datasets']['LasHeR']['dev']
    assert sha(manifest_path) == plan['manifest_sha256']
    manifest = load_manifest(manifest_path)
    sequence = next(s for s in manifest['sequences'] if s['name'] == args.sequence)
    assert args.sequence in plan['sequence_names']
    if not args.smoke:
        smoke_path = ROOT / plan['smoke_result']
        assert sha(smoke_path) == plan['smoke_result_sha256']
        smoke = json.loads(smoke_path.read_text())
        assert smoke['status'] == 'SMOKE_PASS' and smoke['observer_control_equal']
        assert smoke['checkpoint_sha256'] == plan['checkpoint_sha256']
        assert smoke['source_sha256'] == plan['source_sha256']
        smoke_native = ROOT / plan['smoke_native_status']
        assert sha(smoke_native) == plan['smoke_native_sha256']
        terminal = json.loads(smoke_native.read_text())
        assert terminal['state'] == 'COMPLETE' and terminal['exit_code'] == 0
    reference = ROOT / 'runs' / plan['reference_run']
    reference_config = json.loads((reference / 'config.json').read_text())
    assert reference_config['checkpoint_sha256'] == plan['checkpoint_sha256']
    assert reference_config['manifest_sha256'] == plan['manifest_sha256']
    assert reference_config['cost_weight'] == plan['cost_weight'] == .15
    assert reference_config['max_candidates'] == parent['max_candidates'] == 8
    reference_predictions = np.loadtxt(reference / 'predictions' / (args.sequence + '.txt'),
                                      delimiter=',', ndmin=2)
    with (reference / 'traces' / (args.sequence + '.csv')).open() as handle:
        reference_routes = list(csv.DictReader(handle))
    count = plan['smoke_frames'] if args.smoke else len(sequence['rgb'])
    assert len(reference_predictions) == len(sequence['rgb'])
    checkpoint = ROOT / parent['checkpoint']
    assert sha(checkpoint) == parent['checkpoint_sha256']
    state = torch.load(checkpoint, map_location='cpu')['net']
    if plan['overlay'] is not None:
        overlay_path = ROOT / plan['overlay']
        assert sha(overlay_path) == plan['checkpoint_sha256']
        overlay = torch.load(overlay_path, map_location='cpu')
        assert overlay['config']['parent_checkpoint_sha256'] == parent['checkpoint_sha256']
        state.update({'query_gain.' + key: value for key, value in overlay['head'].items()})
    else:
        assert plan['checkpoint_sha256'] == parent['checkpoint_sha256']
    torch.manual_seed(spec['seed'])
    np.random.seed(spec['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_model('e1', ROOT / parent['factory_checkpoint'])
    model.load_state_dict(state, strict=True)
    del state
    freeze(model)
    frozen = cpu_state(model)
    control = []
    if args.smoke:
        tracker = build_tracker('e1', model, plan['cost_weight'], 8)
        tracker.initialize(read_pair(sequence, 0), sequence['initial_box'])
        for frame in range(1, count):
            box, route = tracker.track(read_pair(sequence, frame))
            control.append((snapshot(tracker), route['action'], route['modality']))
    observation = Observation(model)
    tracker = build_tracker('e1', model, plan['cost_weight'], 8)
    directory = ROOT / 'runs' / args.run
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(plan=plan, plan_sha256=sha(plan_path), native_command=native['command'],
                  sequence=args.sequence, smoke=args.smoke, reference_config_sha256=sha(reference / 'config.json'),
                  reference_predictions_sha256=sha(reference / 'predictions' / (args.sequence + '.txt')),
                  reference_trace_sha256=sha(reference / 'traces' / (args.sequence + '.csv')))
    (directory / 'config.json').write_text(json.dumps(config, indent=2))
    tracker.initialize(read_pair(sequence, 0), sequence['initial_box'])
    started = time.time()
    sampled_states, query_count, writes = 0, 0, 0
    with (directory / 'frames.jsonl').open('w') as frames_out, \
            (directory / 'queries.jsonl').open('w') as query_out:
        for frame in range(1, count):
            image = read_pair(sequence, frame)
            before = snapshot(tracker)
            observation.begin()
            prediction, route = tracker.track(image)
            observation.enabled = False
            assert observation.gain_calls == 1 and observation.modality == route['modality']
            assert np.array_equal(np.asarray([float('%.6f' % v) for v in prediction]),
                                  reference_predictions[frame])
            saved_route = reference_routes[frame - 1]
            assert int(saved_route['frame']) == frame
            for key in ('action', 'modality', 'request_index', 'candidate_count'):
                assert route[key] == int(saved_route[key])
            if args.smoke:
                same_state(tracker, control[frame - 1][0])
                assert (route['action'], route['modality']) == control[frame - 1][1:]
            write_modalities = template_event(observation, tracker, before, route)
            writes += bool(write_modalities)
            sampled = frame % plan['label_stride'] == 0
            primary_box = execute_boxes(pixel_xywh(observation.primary['box'], observation.main),
                                        observation.main.new_tensor([image.shape[:2]]))[0].cpu().tolist()
            if sampled:
                primary_box, queries, other_box, other_confidence = candidates(
                    model, observation, image, before, tracker, route)
            # GT is read only after all predictions and candidate executions are fixed.
            target = sequence['boxes'][frame]
            primary_metrics = box_metrics(primary_box, target)
            row = dict(frame=frame, action=route['action'], modality=route['modality'],
                       primary_box=primary_box, final_box=prediction.tolist(),
                       primary=primary_metrics, final=box_metrics(prediction, target),
                       quality=observation.preview_output['quality'][0].cpu().tolist(),
                       template_write_modalities=write_modalities, sampled=sampled)
            if sampled:
                sampled_states += 1
                row.update(other_primary_box=other_box, other_primary=box_metrics(other_box, target),
                           other_primary_confidence=other_confidence,
                           primary_confidence=float(observation.primary['score'].max().item()))
                for query in queries:
                    metrics = box_metrics(query['box'], target)
                    valid = metrics['valid_gt']
                    query.update(frame=frame, modality=route['modality'], metrics=metrics,
                                 primary_iou=primary_metrics['iou'],
                                 gain=metrics['iou'] - primary_metrics['iou'] if valid else None,
                                 utility_prediction=query['predicted_gain'] -
                                 plan['cost_weight'] * query['token_cost'],
                                 recovery=bool(valid and primary_metrics['iou'] < .1 and metrics['iou'] >= .5),
                                 harm=bool(valid and primary_metrics['iou'] >= .5 and metrics['iou'] < .1))
                    query_out.write(json.dumps(query) + '\n')
                    query_count += 1
            frames_out.write(json.dumps(row) + '\n')
            if frame % 100 == 0 or frame == count - 1:
                frames_out.flush()
                query_out.flush()
                progress = dict(frame=frame, total_frames=count, sampled_states=sampled_states,
                                query_executions=query_count, template_writes=writes,
                                wall_seconds=time.time() - started)
                (directory / 'progress.json').write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
    assert all(torch.equal(frozen[k], v) for k, v in cpu_state(model).items())
    assert all(parameter.grad is None for parameter in model.parameters())
    check_files(ROOT, plan['source_sha256'])
    check_files(ROOT, parent['source_sha256'])
    assert sha(plan_path) == config['plan_sha256']
    result = dict(status='SMOKE_PASS' if args.smoke else 'DIAGNOSTIC_COMPLETE',
                  checkpoint_sha256=plan['checkpoint_sha256'], source_sha256=plan['source_sha256'],
                  frames=count, sequence=args.sequence, sampled_states=sampled_states,
                  query_executions=query_count, template_writes=writes, frozen_state_tensors=len(frozen),
                  optimizer_updates=0, backward_calls=0, observer_control_equal=True if args.smoke else None,
                  saved_predictions_and_routes_equal=True, diagnostic_preserves_live_state=True,
                  wall_seconds=time.time() - started, test_eligible=False, training_allowed=False,
                  artifacts_sha256={p.name: sha(p) for p in directory.iterdir() if p.is_file()},
                  scope='Same-state dev diagnosis; sampled candidate population and alternate primary. '
                        'No future trajectory counterfactual, trained model or benchmark FPS.')
    (directory / ('SMOKE.json' if args.smoke else 'COMPLETE.json')).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
