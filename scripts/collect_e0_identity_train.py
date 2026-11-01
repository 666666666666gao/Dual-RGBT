"""Collect executed dual-modal candidate sets on protected causal E0 TRAIN states."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
from dual_rgbt.data import read_pair
from dual_rgbt.e0_identity import initial_identity_anchor, primary_candidate_evidence
from joint_evaluation_contract import check_files
from lib.utils.box_ops import clip_box
from profile_crop_api import six_channel_crop


def sampled_frames(sequence, count):
    return [frame for frame in sequence['valid_frames'] if 0 < frame < count][::25]


def smoke_artifact_names(sequence_names):
    return {'config.json', 'progress.json'} | {name + '.npz' for name in sequence_names}


def bound_smoke_plan_path(root, plan, native):
    command = native['command']
    path = command[command.index('--plan') + 1]
    assert sha(root / path) == plan['smoke_plan_sha256']
    return path


def reviewed_draft_path(phase):
    assert phase in ('smoke', 'full')
    return 'reports/E0_M2_IDENTITY_%s_DRAFT_R7_20261031.json' % phase.upper()


def normalized_cxcywh(boxes, geometry):
    boxes = np.asarray(boxes, dtype=np.float32)
    left, top, side = geometry
    assert boxes.ndim == 2 and boxes.shape[1] == 4 and side > 0
    centre = (boxes[:, :2] + boxes[:, 2:] / 2 - [left, top]) / side
    return np.concatenate([centre, boxes[:, 2:] / side], 1).astype(np.float32)


def returned_box(box, geometry, height, width):
    box = np.asarray(box, dtype=np.float32)
    assert box.shape == (4,) and np.isfinite(box).all() and (box[2:] > 0).all()
    left, top, side = geometry
    xywh = np.concatenate([box[:2] * side + [left, top] - box[2:] * side / 2,
                           box[2:] * side])
    return np.asarray(clip_box(xywh.tolist(), height, width, margin=10), dtype=np.float32)


def global_windows(height, width):
    """Nine overlapping image-space crops; no annotation or model score."""
    side = max(height, width) / 2.
    size = side / 4.
    return [np.asarray([x * width - size / 2, y * height - size / 2, size, size], np.float32)
            for y in (1 / 6., .5, 5 / 6.) for x in (1 / 6., .5, 5 / 6.)]


def overlap_labels(boxes, coverage, groundtruth):
    """Label only executed candidates; ambiguous overlaps stay unlabelled."""
    gt = np.asarray(groundtruth, dtype=np.float64)
    assert gt.shape == (4,) and np.isfinite(gt).all() and (gt[2:] > 0).all()
    assert boxes.shape == (24, 4) and coverage.shape == (24, 2)
    assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
    assert np.isfinite(coverage).all() and ((coverage >= 0) & (coverage <= 1)).all()
    left = np.maximum(boxes[:, :2], gt[:2])
    right = np.minimum(boxes[:, :2] + boxes[:, 2:], gt[:2] + gt[2:])
    intersection = np.maximum(0, right - left).prod(1)
    iou = intersection / (boxes[:, 2:].prod(1) + gt[2:].prod() - intersection)
    assert np.isfinite(iou).all() and ((iou >= 0) & (iou <= 1)).all()
    observed = (coverage == 1).any(1)
    labels = np.full(25, -1, np.int8)
    labels[1:][observed & (iou < .1)] = 0
    labels[1:][observed & (iou >= .5)] = 1
    if (labels[1:] == 1).any():
        labels[0] = 0
    elif observed.any() and (iou < .1).all():
        # No trustworthy match in this executed set, not absent-target evidence.
        labels[0] = 1
    return iou, labels


@torch.no_grad()
def execute_candidates(model, templates, image, previous):
    """Use fixed initial templates; collect before consulting current TRAIN GT."""
    device = templates.device
    records, geometry_rows, queries = [], [], []

    def query(box, parent):
        search, _, geometry = six_channel_crop(image, box, 4., 256)
        search = search[None].to(device)
        primary = [model.primary(templates, search, modality) for modality in (0, 1)]
        boxes = np.stack([returned_box(result['box'][0].cpu().numpy(), geometry,
            image.shape[0], image.shape[1]) for result in primary])
        normalized = torch.from_numpy(normalized_cxcywh(boxes, geometry)).to(device)
        evidence = [primary_candidate_evidence(result['features'], normalized) for result in primary]
        descriptors = torch.stack([row[0] for row in evidence], 1).cpu().numpy()
        coverage = torch.stack([row[1] for row in evidence], 1).cpu().numpy()
        index = len(queries)
        queries.append(dict(query=index, parent=parent, crop_box=np.asarray(box).tolist()))
        geometry_rows.append(geometry)
        for modality in (0, 1):
            records.append(dict(box=boxes[modality], features=descriptors[modality],
                coverage=coverage[modality], score=float(primary[modality]['score'].max().item()),
                query=index, modality=modality))

    query(previous, -1)
    for box in global_windows(image.shape[0], image.shape[1]):
        query(box, -1)
    # Stable score ordering uses only the executed global candidates, never GT.
    scores = np.asarray([row['score'] for row in records[2:20]])
    assert np.isfinite(scores).all()
    parents = np.argsort(-scores, kind='mergesort')[:2] + 2
    for parent in parents:
        coarse = records[int(parent)]['box']
        centre = coarse[:2] + coarse[2:] / 2
        fine = np.concatenate([centre - previous[2:] / 2, previous[2:]]).astype(np.float32)
        query(fine, int(parent))
    assert len(records) == 24 and len(queries) == 12
    result = {key: np.asarray([row[key] for row in records])
              for key in ('box', 'features', 'coverage', 'score', 'query', 'modality')}
    assert result['features'].shape == (24, 2, 768)
    assert np.isfinite(result['features']).all() and np.isfinite(result['score']).all()
    result['geometry'] = np.asarray(geometry_rows, dtype=np.float32)
    result['crop_boxes'] = np.asarray([row['crop_box'] for row in queries], dtype=np.float32)
    result['refinement_parents'] = parents.astype(np.int64)
    return result


def check_admission(root, plan):
    assert plan['mode'] == 'protected_e0_identity_train_collection'
    assert plan['launch_allowed'] is True and plan['training_allowed'] is False
    assert plan['split'] == 'train' and plan['test_eligible'] is False
    assert plan['phase'] in ('smoke', 'full')
    assert plan['reviewed_draft'] == reviewed_draft_path(plan['phase'])
    assert plan['label_stride'] == 25 and plan['candidate_count'] == 24
    assert plan['sampling_rule'] == 'positive_valid_frames[::25]'
    assert plan['cost_weight'] == 0. and plan['matmul_tf32'] is plan['cudnn_tf32'] is False
    assert plan['checkpoint_sha256'] == 'ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c'
    check_files(root, plan['source_sha256'])
    check_files(root, plan['input_sha256'])
    review_path = root / plan['source_review']
    assert sha(review_path) == plan['source_review_sha256']
    review = json.loads(review_path.read_bytes())
    assert review['stage'] == 'M2_identity_collector'
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['source_sha256'] == plan['source_sha256']
    draft_path = root / plan['reviewed_draft']
    assert sha(draft_path) == review['reviewed_drafts'][plan['reviewed_draft']]
    draft = json.loads(draft_path.read_bytes())
    assert draft['phase'] == plan['phase'] and draft['launch_allowed'] is False
    variable = {'launch_allowed', 'source_review_sha256',
                'smoke_plan_sha256', 'smoke_receipt_sha256', 'smoke_native_sha256'}
    assert set(plan) == set(draft) and draft['launch_allowed'] is False
    assert all(plan[key] == value for key, value in draft.items() if key not in variable)
    if plan['phase'] == 'smoke':
        assert plan['frames_per_sequence'] == 31 and plan['shards'] == 1
        assert len(plan['sequence_names']) == 3 and plan['smoke_run'] is None
    else:
        assert plan['phase'] == 'full' and plan['frames_per_sequence'] == 0 and plan['shards'] == 8
        assert len(plan['sequence_names']) == len(set(plan['sequence_names'])) == 882
        smoke_path = root / 'runs' / plan['smoke_run'] / 'COMPLETE.json'
        assert sha(smoke_path) == plan['smoke_receipt_sha256']
        smoke = json.loads(smoke_path.read_bytes())
        directory = smoke_path.parent
        config = json.loads((directory / 'config.json').read_bytes())
        native_path = root / 'runs/pipeline_tasks' / plan['smoke_task'] / 'status.json'
        assert sha(native_path) == plan['smoke_native_sha256']
        native = json.loads(native_path.read_bytes())
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
        assert native['task'] == smoke['task'] == config['task'] == plan['smoke_task']
        assert native['command'] == config['native_command']
        smoke_plan_path = bound_smoke_plan_path(root, plan, native)
        assert native['command'] == [str(root / 'conda-envs/dual_rgbt/bin/python'),
            str(root / 'scripts/collect_e0_identity_train.py'), '--plan', smoke_plan_path,
            '--run', plan['smoke_run'], '--task', plan['smoke_task'], '--shard', '0']
        assert sha(root / plan['smoke_plan']) == config['plan_sha256'] == plan['smoke_plan_sha256']
        assert config['plan'] == json.loads((root / plan['smoke_plan']).read_bytes())
        assert config['plan']['source_sha256'] == plan['source_sha256']
        assert config['plan']['input_sha256'] == plan['input_sha256']
        assert config['plan']['source_review_sha256'] == plan['source_review_sha256']
        assert config['plan']['reviewed_draft'] == reviewed_draft_path('smoke')
        smoke_draft_path = root / config['plan']['reviewed_draft']
        assert sha(smoke_draft_path) == review['reviewed_drafts'][config['plan']['reviewed_draft']]
        smoke_draft = json.loads(smoke_draft_path.read_bytes())
        assert smoke_draft['phase'] == 'smoke' and smoke_draft['launch_allowed'] is False
        assert {key: value for key, value in config['plan'].items()
                if key not in ('launch_allowed', 'source_review_sha256')} == {
                    key: value for key, value in smoke_draft.items()
                    if key not in ('launch_allowed', 'source_review_sha256')}
        assert smoke['plan_sha256'] == config['plan_sha256'] == sha(root / plan['smoke_plan'])
        assert smoke['checkpoint_sha256'] == plan['checkpoint_sha256']
        assert smoke['status'] == 'M2_IDENTITY_CACHE_COMPLETE' and smoke['smoke_only'] is True
        assert smoke['all_online_states_bit_equal'] is True and smoke['sequences'] == 3
        expected_sets = 0
        manifest = json.loads((root / plan['manifest']).read_bytes())
        sequences = {row['name']: row for row in manifest['train']}
        assert config['selected_sequences'] == smoke_draft['sequence_names']
        assert set(smoke['artifacts_sha256']) == smoke_artifact_names(config['selected_sequences'])
        progress = json.loads((directory / 'progress.json').read_bytes())
        assert [row['sequence'] for row in progress] == config['selected_sequences']
        for row in progress:
            name = row['sequence']
            count = len(sampled_frames(sequences[name], 31))
            expected_sets += count
            relative = name + '.npz'
            assert sha(directory / relative) == smoke['artifacts_sha256'][relative]
            with np.load(directory / relative, allow_pickle=False) as arrays:
                assert arrays['frame'].tolist() == sampled_frames(sequences[name], 31)
                assert arrays['predictions'].shape == (31, 4)
                assert arrays['anchor'].shape == (2, 768)
                assert arrays['features'].shape == (count, 24, 2, 768)
                assert arrays['coverage'].shape == (count, 24, 2)
                assert arrays['labels'].shape == (count, 25)
                assert np.isfinite(arrays['features']).all()
                assert np.isin(arrays['labels'], (-1, 0, 1)).all()
        assert smoke['frames'] == 93 and smoke['candidate_sets'] == expected_sets
        assert smoke['anchor_encoder_calls'] == 6 and smoke['candidate_encoder_calls'] == 24 * expected_sets
        assert smoke['optimizer_updates'] == smoke['backward_calls'] == 0
        assert smoke['frozen_state_tensors'] == 265
        for relative, digest in smoke['artifacts_sha256'].items():
            assert sha(directory / relative) == digest


@torch.no_grad()
def main():
    from check_e0_observation_train import equal_state, snapshot
    from run_benchmark_tracking import build_model, build_tracker, freeze
    from train_global_evidence_policy import cpu_state

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--shard', required=True, type=int)
    args = parser.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_bytes())
    check_admission(ROOT, plan)
    assert 0 <= args.shard < plan['shards']
    job = plan['jobs'][args.shard]
    assert (job['run'], job['task'], job['shard']) == (args.run, args.task, args.shard)
    native = json.loads((ROOT / 'runs/pipeline_tasks' / args.task / 'status.json').read_bytes())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['command'] == [sys.executable] + sys.argv
    assert native['physical_gpu'] == job['assigned_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID' and os.environ['OPENCV_FOR_THREADS_NUM'] == '1'
    manifest = json.loads((ROOT / plan['manifest']).read_bytes())
    by_name = {row['name']: row for row in manifest['train']}
    fit, held = set(plan['fit_sequences']), set(plan['holdout_sequences'])
    assert len(by_name) == 882 and len(fit) == 706 and len(held) == 176
    assert fit.isdisjoint(held) and fit | held == set(by_name)
    assert set(by_name).isdisjoint(set(manifest['test_sequences']) | {row['name'] for row in manifest['dev']})
    if plan['phase'] == 'full':
        assert plan['sequence_names'] == [row['name'] for row in manifest['train']]
    selected = plan['sequence_names'][args.shard::plan['shards']]
    assert job['sequence_names'] == selected and set(selected) <= set(by_name)
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = build_model('e0', None)
    model.load_state_dict(torch.load(ROOT / plan['checkpoint'], map_location='cpu')['net'], strict=True)
    freeze(model)
    before = cpu_state(model)
    assert len(before) == 265 and sum(p.numel() for p in model.parameters()) == 103374215
    directory = ROOT / 'runs' / args.run
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(plan=plan, plan_sha256=sha(path), native_command=native['command'],
        task=args.task, run=args.run, shard=args.shard, selected_sequences=selected, split='train')
    (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    progress, started, total_sets = [], time.time(), 0
    for name in selected:
        sequence = by_name[name]
        count = 31 if plan['phase'] == 'smoke' else len(sequence['rgb'])
        assert len(sequence['rgb']) >= count
        trackers = [build_tracker('e0', model, 0., 0) for _ in range(2)]
        initial_image = read_pair(sequence, 0)
        for tracker in trackers:
            tracker.initialize(initial_image, sequence['boxes'][0])
        equal_state(snapshot(trackers[0]), snapshot(trackers[1]))
        templates = trackers[1].templates.clone()
        search, _, geometry = six_channel_crop(initial_image, trackers[1].state, 4., 256)
        initial_box = normalized_cxcywh(trackers[1].state[None], geometry)
        anchor = initial_identity_anchor(model, templates, search[None].cuda(),
            torch.from_numpy(initial_box).cuda())[0].cpu().numpy()
        fixed_templates = templates.cpu().clone()
        predictions = [trackers[1].state.copy()]
        samples = []
        sample_ids = sampled_frames(sequence, count)
        for frame in range(1, count):
            image = read_pair(sequence, frame)
            previous = trackers[1].state.copy()
            first_box, first_route = trackers[0].track(image)
            box, route = trackers[1].track(image)
            assert np.array_equal(box, first_box)
            assert {k: v for k, v in route.items() if k != 'seconds'} == {
                k: v for k, v in first_route.items() if k != 'seconds'}
            equal_state(snapshot(trackers[0]), snapshot(trackers[1]))
            predictions.append(box.copy())
            if frame in sample_ids:
                state_before = snapshot(trackers[1])
                sample = execute_candidates(model, templates, image, previous)
                equal_state(snapshot(trackers[1]), state_before)
                assert torch.equal(templates.cpu(), fixed_templates)
                # All actual encoder/head queries finish before current GT is read.
                iou, labels = overlap_labels(sample['box'], sample['coverage'], sequence['boxes'][frame])
                sample.update(frame=frame, iou=iou, labels=labels)
                samples.append(sample)
            if frame % 100 == 0:
                (directory / 'current.json').write_text(json.dumps(dict(
                    sequence=name, frame=frame, frames=count, candidate_sets=len(samples), updated=time.time())))
        assert [sample['frame'] for sample in samples] == sample_ids
        assert samples and np.isfinite(anchor).all()
        arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
        arrays.update(anchor=anchor, anchor_geometry=geometry, anchor_box=initial_box[0],
                      predictions=np.asarray(predictions, dtype=np.float32))
        np.savez(directory / (name + '.npz'), **arrays)
        total_sets += len(samples)
        progress.append(dict(sequence=name, partition='fit' if name in fit else 'holdout',
            frames=count, tracked_frames=count - 1, candidate_sets=len(samples),
            positive_sets=sum((sample['labels'][1:] == 1).any() for sample in samples),
            no_match_sets=sum(sample['labels'][0] == 1 for sample in samples),
            ambiguous_sets=sum(sample['labels'][0] == -1 for sample in samples),
            all_online_states_bit_equal=True))
        (directory / 'progress.json').write_text(json.dumps(progress, indent=2, default=int) + '\n')
        print(json.dumps(progress[-1], default=int), flush=True)
    assert all(torch.equal(before[key], value) for key, value in cpu_state(model).items())
    assert all(p.grad is None for p in model.parameters())
    check_files(ROOT, plan['source_sha256'])
    check_files(ROOT, plan['input_sha256'])
    assert sha(path) == config['plan_sha256']
    receipt = dict(status='M2_IDENTITY_CACHE_COMPLETE', task=args.task, split='train',
        plan_sha256=sha(path), checkpoint_sha256=plan['checkpoint_sha256'],
        smoke_only=plan['phase'] == 'smoke', sequences=len(selected),
        frames=sum(row['frames'] for row in progress), candidate_sets=total_sets,
        candidate_count=24, anchor_encoder_calls=2 * len(selected),
        candidate_encoder_calls=24 * total_sets, all_online_states_bit_equal=True,
        frozen_state_tensors=len(before), optimizer_updates=0, backward_calls=0,
        test_eligible=False, scientific_result_audit_accepted=False,
        wall_seconds=time.time() - started,
        artifacts_sha256={p.relative_to(directory).as_posix(): sha(p)
                          for p in directory.rglob('*') if p.is_file()},
        scope='TRAIN executed candidate sets with fixed initial templates/anchors and '
              'paired original E0 trajectories. Current overlap labels read only after '
              'queries; ambiguous or uncovered candidates remain unlabelled. No-match '
              'means no trusted match in this set, not absent target. Component holdout '
              'was seen upstream. No optimizer, online recovery or benchmark claim.')
    (directory / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'artifacts_sha256'}), flush=True)


if __name__ == '__main__':
    main()
