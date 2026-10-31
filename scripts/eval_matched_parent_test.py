"""Create a full-test control for the unchanged current parent checkpoint."""
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
sys.path.insert(0, str(ROOT / 'tools'))
from benchmark_inputs import groundtruth, load_manifest, sha
from dual_rgbt.data import read_pair as read_original_pair
from joint_evaluation_contract import check_files, evaluation_plan
from query_evidence_capture import FIELDS, QueryEvidenceCapture
from run_benchmark_tracking import build_model, build_tracker, freeze
from run_cached_gain_server26_test import read_pair as read_test_pair, ORIGINAL_PREFIX, LOCAL_PREFIX
from train_global_evidence_policy import cpu_state

TRAIN_ORIGINAL_PREFIX = '/data2/wangwj/dataset/LasHeR-0428/trainingset/'
TRAIN_LOCAL_PREFIX = '/data2/gb/Dual-RGBT/data/LasHeR/train/'


def read_train_pair(sequence, frame):
    pair = {}
    for key in ('rgb', 'tir'):
        path = sequence[key][frame]
        assert path.startswith(TRAIN_ORIGINAL_PREFIX)
        pair[key] = [TRAIN_LOCAL_PREFIX + path[len(TRAIN_ORIGINAL_PREFIX):]]
    return read_original_pair(pair, 0)


def admitted_scope(split, smoke_frames, exploratory_test, full_evaluation_allowed):
    assert full_evaluation_allowed is True
    if split == 'test':
        assert smoke_frames == 0 and exploratory_test is True
        return 245, 220703
    assert split == 'train_smoke' and smoke_frames == 11
    return 1, 11


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--split', choices=('train_smoke', 'test'), required=True)
    parser.add_argument('--smoke-frames', type=int, choices=(0, 11), default=0)
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    expected_sequences, expected_frames = admitted_scope(
        args.split, args.smoke_frames, plan['exploratory_test'], plan['full_evaluation_allowed'])
    assert plan['mode'] == 'matched_parent_full_test_control'
    assert plan['test_eligible'] is False
    assert plan['runtime_provider'] == dict(ssh_alias='2026', project=str(ROOT))
    check_files(ROOT, plan['source_sha256'])
    review_path = ROOT / plan['source_review']
    assert sha(review_path) == plan['source_review_sha256']
    review = json.loads(review_path.read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['source_sha256'] == plan['source_sha256']
    parent, spec, _ = evaluation_plan(ROOT, plan['parent_evaluation_plan'])
    assert plan['checkpoint'] == parent['checkpoint']
    assert plan['checkpoint_sha256'] == plan['parent_checkpoint_sha256'] == parent['checkpoint_sha256']
    assert plan['cost_weight'] == .15 and parent['max_candidates'] == 8
    native_path = ROOT / 'runs/pipeline_tasks' / args.task / 'status.json'
    native = json.loads(native_path.read_text())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['command'] == [sys.executable] + sys.argv
    assert native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID' and os.environ['OPENCV_FOR_THREADS_NUM'] == '1'
    manifest_path = ROOT / spec['datasets']['LasHeR'][args.split]
    assert sha(manifest_path) == parent['input_sha256'][manifest_path.relative_to(ROOT).as_posix()]
    manifest = load_manifest(manifest_path)
    train_mapping_path = ROOT / plan['train_mapping_receipt']
    assert sha(train_mapping_path) == plan['train_mapping_receipt_sha256']
    train_mapping = json.loads(train_mapping_path.read_text())
    assert train_mapping['status'] == 'TRAIN_SMOKE_MAPPING_VERIFIED'
    assert (train_mapping['sequences'], train_mapping['frames'], train_mapping['images']) == (1, 11, 22)
    assert train_mapping['original_prefix'] == TRAIN_ORIGINAL_PREFIX
    assert train_mapping['local_prefix'] == TRAIN_LOCAL_PREFIX
    train_manifest = ROOT / spec['datasets']['LasHeR']['train_smoke']
    assert train_mapping['manifest_sha256'] == sha(train_manifest)
    assert sha(ROOT / train_mapping['source_receipt']) == train_mapping['source_receipt_sha256']
    for relative, digest in train_mapping['files_sha256'].items():
        assert sha(Path(TRAIN_LOCAL_PREFIX) / relative) == digest
    if args.split == 'test':
        assert (manifest['sequence_count'], manifest['frame_count']) == (245, 220703)
        mapping_path = ROOT / plan['dataset_mapping_receipt']
        assert sha(mapping_path) == plan['dataset_mapping_receipt_sha256']
        mapping = json.loads(mapping_path.read_text())
        assert mapping['status'] == 'TEST_MAPPING_VERIFIED'
        assert mapping['original_prefix'] == ORIGINAL_PREFIX and mapping['local_prefix'] == LOCAL_PREFIX
        assert (mapping['sequences'], mapping['frames']) == (245, 220703)
        smoke_dir = ROOT / 'runs' / plan['smoke_run']
        smoke = json.loads((smoke_dir / 'SMOKE.json').read_text())
        smoke_config = json.loads((smoke_dir / 'config.json').read_text())
        smoke_native = json.loads((ROOT / 'runs/pipeline_tasks' / smoke['task'] / 'status.json').read_text())
        assert smoke_native['state'] == 'COMPLETE' and smoke_native['exit_code'] == 0
        assert smoke_native['command'] == smoke_config['native_command']
        assert smoke_config['split'] == 'train_smoke' and smoke_config['smoke_frames'] == 11
        assert smoke_config['run'] == plan['smoke_run'] and smoke['task'] == smoke_config['task']
        assert sha(smoke_dir / 'config.json') == smoke['config_sha256']
        assert smoke['plan_sha256'] == smoke_config['plan_sha256'] == sha(plan_path)
        assert smoke['checkpoint_sha256'] == plan['checkpoint_sha256']
        assert (smoke['sequences'], smoke['frames'], smoke['frozen_state_tensors']) == (1, 11, 265)
        assert smoke['optimizer_updates'] == smoke['backward_calls'] == 0
    sequences = manifest['sequences'][:1] if args.smoke_frames else manifest['sequences']
    read_pair = read_train_pair if args.split == 'train_smoke' else read_test_pair
    checkpoint = ROOT / parent['checkpoint']
    assert sha(checkpoint) == parent['checkpoint_sha256']
    torch.manual_seed(spec['seed'])
    np.random.seed(spec['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_model('e1', ROOT / parent['factory_checkpoint'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
    freeze(model)
    before = cpu_state(model)
    tracker = build_tracker('e1', model, plan['cost_weight'], parent['max_candidates'])
    evidence = QueryEvidenceCapture(model, plan['cost_weight'])
    directory = ROOT / 'runs' / args.run
    directory.mkdir(parents=True, exist_ok=False)
    for folder in ('predictions', 'groundtruth', 'traces', 'evidence'):
        (directory / folder).mkdir()
    config = dict(vars(args), benchmark='LasHeR', shard=0, shards=1,
        checkpoint=parent['checkpoint'], checkpoint_sha256=parent['checkpoint_sha256'],
        parent_checkpoint_sha256=parent['checkpoint_sha256'], plan_sha256=sha(plan_path),
        cost_weight=plan['cost_weight'], max_candidates=parent['max_candidates'],
        source_sha256=dict(parent['source_sha256'], **plan['source_sha256']),
        manifest_sha256=sha(manifest_path), native_command=native['command'],
        split_sequences=manifest['sequence_count'], split_frames=manifest['frame_count'],
        sequences=[s['name'] for s in sequences], total_parameters=sum(p.numel() for p in model.parameters()),
        evaluation_class=plan['mode'], exploratory_test=args.split == 'test', test_eligible=False,
        changed_state_tensors=[], evidence_schema='primary_executed_box_and_selected_gain_v1',
        intervention='None: unchanged parent weights, online tracker, template writes, K8 and cost .15.',
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
        actions, modalities = np.zeros(5, dtype=np.int64), np.zeros(2, dtype=np.int64)
        with (directory / 'traces' / (name + '.csv')).open('w', newline='') as handle, \
                (directory / 'evidence' / (name + '.csv')).open('w', newline='') as evidence_handle:
            writer = csv.DictWriter(handle, ['frame', 'modality', 'action', 'confidence', 'seconds',
                'seconds_including_decode', 'primary_tokens', 'auxiliary_tokens', 'request_index',
                'candidate_rank', 'candidate_count'])
            writer.writeheader()
            evidence_writer = csv.DictWriter(evidence_handle, FIELDS)
            evidence_writer.writeheader()
            for frame in range(1, count):
                started = time.perf_counter()
                image = read_pair(sequence, frame)
                evidence.begin(image, tracker.state)
                prediction, route = tracker.track(image)
                elapsed = time.perf_counter() - started
                evidence_writer.writerow(evidence.finish(frame, route, prediction))
                assert np.isfinite(prediction).all() and (prediction[2:] > 0).all()
                predictions.append(prediction)
                timings.append(elapsed)
                actions[route['action']] += 1
                modalities[route['modality']] += 1
                writer.writerow(dict(frame=frame, seconds_including_decode=elapsed, **route))
        np.savetxt(directory / 'predictions' / (name + '.txt'), predictions, fmt='%.6f', delimiter=',')
        visible, _ = groundtruth(sequence, 'LasHeR')
        np.savetxt(directory / 'groundtruth' / (name + '.txt'), visible[:count], fmt='%.9g', delimiter=',')
        receipts.append(dict(sequence=name, frames=count, initialized_once=True,
            initialization_seconds=init_seconds, action_counts=actions.tolist(), modality_counts=modalities.tolist()))
        (directory / 'progress.json').write_text(json.dumps(receipts, indent=2))
        print(json.dumps(receipts[-1]), flush=True)
    assert all(torch.equal(before[k], v) for k, v in cpu_state(model).items())
    assert all(p.grad is None for p in model.parameters())
    timing = np.asarray(timings)
    assert len(receipts) == expected_sequences and len(timing) == expected_frames
    assert np.isfinite(timing).all() and (timing > 0).all()
    summary = dict(task=args.task, sequences=len(receipts), frames=len(timing),
        checkpoint_sha256=parent['checkpoint_sha256'], plan_sha256=config['plan_sha256'],
        config_sha256=sha(directory / 'config.json'), optimizer_updates=0, backward_calls=0,
        frozen_state_tensors=len(before), total_parameters=config['total_parameters'],
        wall_seconds=time.perf_counter() - wall_started, seconds=float(timing.sum()),
        fps=len(timing)/float(timing.sum()), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), evaluation_class=plan['mode'],
        test_eligible=False, exploratory_test=args.split == 'test',
        image_decoding_included=True, initialization_included=True, model_loading_excluded=True,
        fps_caveat=config['fps_caveat'])
    check_files(ROOT, config['source_sha256'])
    check_files(ROOT, parent['input_sha256'])
    assert sha(checkpoint) == parent['checkpoint_sha256'] and sha(plan_path) == config['plan_sha256']
    np.save(directory / 'frame_seconds.npy', timing)
    (directory / ('SMOKE.json' if args.smoke_frames else 'COMPLETE.json')).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
