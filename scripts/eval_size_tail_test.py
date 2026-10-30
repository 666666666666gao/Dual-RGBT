"""Evaluate the unchanged parent or size-tail weights on an explicit split."""
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
from benchmark_inputs import groundtruth, load_manifest, sha
from cached_gain_evaluation_contract import scope
from dual_rgbt.data import read_pair
from joint_evaluation_contract import check_files, evaluation_plan
from run_benchmark_tracking import build_model, build_tracker, freeze
from train_global_evidence_policy import cpu_state
from query_evidence_capture import FIELDS as EVIDENCE_FIELDS, QueryEvidenceCapture


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--smoke-frames', type=int, choices=(0, 11), default=0)
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text())
    expected_sequences, expected_frames, exploratory_test = scope(
        args.split, args.smoke_frames, plan.get('exploratory_test') is True,
        plan.get('full_evaluation_allowed') is True)
    if exploratory_test:
        assert plan['mode'] in ('parent_control', 'size_tail_diagnostic')
        review_path = ROOT / plan['source_review']
        assert sha(review_path) == plan['source_review_sha256']
        review = json.loads(review_path.read_text())
        assert review['status'] == 'SOURCE_PASS'
        evaluator_source = Path(__file__).relative_to(ROOT).as_posix()
        assert review['source_sha256'][evaluator_source] == plan['source_sha256'][evaluator_source]
    check_files(ROOT, plan['source_sha256'])
    parent, spec, _ = evaluation_plan(ROOT, plan['parent_evaluation_plan'])
    assert parent['checkpoint_sha256'] == plan['parent_checkpoint_sha256']
    native = json.loads((ROOT / 'runs/pipeline_tasks' / args.task / 'status.json').read_text())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['command'] == [sys.executable] + sys.argv
    assert native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID' and os.environ['OPENCV_FOR_THREADS_NUM'] == '1'
    checkpoint = ROOT / plan['checkpoint']
    assert sha(checkpoint) == plan['checkpoint_sha256']
    parent_state = torch.load(ROOT / parent['checkpoint'], map_location='cpu')['net']
    state = torch.load(checkpoint, map_location='cpu')
    assert set(state['net']) == set(parent_state)
    changed = [k for k in parent_state if not torch.equal(parent_state[k], state['net'][k])]
    control = plan['mode'] == 'parent_control'
    assert plan['mode'] in ('parent_control', 'size_tail_diagnostic')
    if control:
        assert not changed and plan['checkpoint_sha256'] == parent['checkpoint_sha256']
    else:
        assert state['stage'] == 'size_tail_adaptation'
        assert set(changed) == {'executor.box_head.conv5_size.weight', 'executor.box_head.conv5_size.bias'}
        assert state['config']['plan']['parent_checkpoint_sha256'] == parent['checkpoint_sha256']
        training = json.loads((ROOT / 'runs/pipeline_tasks' / plan['training_task'] / 'status.json').read_text())
        assert training['state'] == 'COMPLETE' and training['exit_code'] == 0
        trained = json.loads((checkpoint.parent / 'COMPLETE.json').read_text())
        assert trained['strict_reload_bit_equal'] and trained['trainable_parameters'] == 66
        assert trained['checkpoint_sha256'] == plan['checkpoint_sha256']
    del parent_state
    reference = ROOT / 'runs' / plan['reference_run']
    reference_config = json.loads((reference / 'config.json').read_text())
    assert reference_config['checkpoint_sha256'] == parent['checkpoint_sha256']
    assert reference_config['cost_weight'] == plan['cost_weight']
    assert reference_config['max_candidates'] == parent['max_candidates']
    manifest_path = ROOT / spec['datasets']['LasHeR'][args.split]
    if args.split == 'dev':
        assert sha(manifest_path) == reference_config['manifest_sha256']
    else:
        assert sha(manifest_path) == parent['input_sha256'][manifest_path.relative_to(ROOT).as_posix()]
    manifest = load_manifest(manifest_path)
    assert (manifest['sequence_count'], manifest['frame_count']) == (expected_sequences, expected_frames)
    sequences = manifest['sequences'][:1] if args.smoke_frames else manifest['sequences']
    torch.manual_seed(spec['seed'])
    np.random.seed(spec['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_model('e1', ROOT / parent['factory_checkpoint'])
    model.load_state_dict(state['net'], strict=True)
    del state
    freeze(model)
    before = cpu_state(model)
    tracker = build_tracker('e1', model, plan['cost_weight'], parent['max_candidates'])
    evidence = QueryEvidenceCapture(model, plan['cost_weight'])
    directory = ROOT / 'runs' / args.run
    directory.mkdir(parents=True, exist_ok=False)
    for name in ('predictions', 'groundtruth', 'traces', 'evidence'):
        (directory / name).mkdir()
    config = dict(vars(args), benchmark='LasHeR', shard=0, shards=1,
        checkpoint_sha256=plan['checkpoint_sha256'], checkpoint=plan['checkpoint'],
        plan_sha256=sha(ROOT / args.plan), parent_checkpoint_sha256=parent['checkpoint_sha256'],
        cost_weight=plan['cost_weight'], max_candidates=parent['max_candidates'],
        split_sequences=expected_sequences, split_frames=expected_frames, sequences=[s['name'] for s in sequences],
        manifest_sha256=sha(manifest_path), native_command=native['command'],
        source_sha256=dict(parent['source_sha256'], **plan['source_sha256']),
        total_parameters=sum(p.numel() for p in model.parameters()),
        evaluation_class=plan['mode'], test_eligible=False, exploratory_test=exploratory_test,
        changed_state_tensors=changed,
        fps_caveat='Concurrent GPU workloads; not an isolated throughput benchmark.')
    assert config['total_parameters'] == spec['total_parameters']
    (directory / 'config.json').write_text(json.dumps(config, indent=2))
    timings, receipts = [], []
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    for sequence in sequences:
        name = sequence['name']
        count = args.smoke_frames or len(sequence['rgb'])
        predictions = [np.asarray(sequence['initial_box'], dtype=np.float32)]
        started = time.perf_counter()
        tracker.initialize(read_pair(sequence, 0), sequence['initial_box'])
        init_seconds = time.perf_counter() - started
        timings.append(init_seconds)
        routes = []
        actions, modalities = np.zeros(5, dtype=np.int64), np.zeros(2, dtype=np.int64)
        with (directory / 'traces' / (name + '.csv')).open('w', newline='') as handle, \
                (directory / 'evidence' / (name + '.csv')).open('w', newline='') as evidence_handle:
            fields = ['frame', 'modality', 'action', 'confidence', 'seconds', 'seconds_including_decode',
                'primary_tokens', 'auxiliary_tokens', 'request_index', 'candidate_rank', 'candidate_count',
                'primary_confidence', 'query_confidence', 'query_selected', 'template_updated',
                'primary_box_x', 'primary_box_y', 'primary_box_w', 'primary_box_h',
                'query_box_x', 'query_box_y', 'query_box_w', 'query_box_h',
                'final_box_x', 'final_box_y', 'final_box_w', 'final_box_h']
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            evidence_writer = csv.DictWriter(evidence_handle, EVIDENCE_FIELDS)
            evidence_writer.writeheader()
            for frame in range(1, count):
                started = time.perf_counter()
                image = read_pair(sequence, frame)
                evidence.begin(image, tracker.state)
                prediction, route = tracker.track(image)
                elapsed = time.perf_counter() - started
                evidence_row = evidence.finish(frame, route, prediction)
                assert np.isfinite(prediction).all() and (prediction[2:] > 0).all()
                predictions.append(prediction)
                timings.append(elapsed)
                routes.append((route['action'], route['modality']))
                actions[route['action']] += 1
                modalities[route['modality']] += 1
                writer.writerow(dict(frame=frame, seconds_including_decode=elapsed, **route))
                evidence_writer.writerow(evidence_row)
        output = directory / 'predictions' / (name + '.txt')
        np.savetxt(output, predictions, fmt='%.6f', delimiter=',')
        visible, _ = groundtruth(sequence, 'LasHeR')
        np.savetxt(directory / 'groundtruth' / (name + '.txt'), visible[:count], fmt='%.9g', delimiter=',')
        if control:
            old = reference / 'predictions' / (name + '.txt')
            assert output.read_text().splitlines() == old.read_text().splitlines()[:count], name
            with (reference / 'traces' / (name + '.csv')).open() as handle:
                old_routes = [(int(row['action']), int(row['modality'])) for row in csv.DictReader(handle)]
            assert routes == old_routes[:count - 1], name
        receipts.append(dict(sequence=name, frames=count, initialized_once=True,
            initialization_seconds=init_seconds, action_counts=actions.tolist(), modality_counts=modalities.tolist(),
            parent_prediction_and_route_equal=control))
        (directory / 'progress.json').write_text(json.dumps(receipts, indent=2))
        print(json.dumps(receipts[-1]), flush=True)
    assert all(torch.equal(before[k], v) for k, v in cpu_state(model).items())
    assert all(p.grad is None for p in model.parameters())
    timing = np.asarray(timings)
    assert np.isfinite(timing).all() and (timing > 0).all()
    assert len(timing) == (11 if args.smoke_frames else expected_frames)
    summary = dict(task=args.task, sequences=len(receipts), frames=len(timing),
        checkpoint_sha256=plan['checkpoint_sha256'], plan_sha256=config['plan_sha256'],
        backward_calls=0, optimizer_updates=0, frozen_state_tensors=len(before),
        total_parameters=config['total_parameters'], parent_prediction_and_route_equal=control,
        wall_seconds=time.perf_counter() - wall_started, seconds=float(timing.sum()),
        fps=len(timing)/float(timing.sum()), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), evaluation_class=plan['mode'],
        test_eligible=False, exploratory_test=exploratory_test,
        image_decoding_included=True, initialization_included=True,
        model_loading_excluded=True, fps_caveat=config['fps_caveat'])
    check_files(ROOT, config['source_sha256'])
    check_files(ROOT, parent['input_sha256'])
    assert sha(checkpoint) == plan['checkpoint_sha256']
    assert sha(ROOT / args.plan) == config['plan_sha256']
    np.save(directory / 'frame_seconds.npy', timing)
    (directory / ('SMOKE.json' if args.smoke_frames else 'COMPLETE.json')).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
