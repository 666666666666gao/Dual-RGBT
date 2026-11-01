"""Verify executed M2 TRAIN caches before standalone identity-head training."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from benchmark_inputs import sha
from collect_e0_identity_train import (
    check_admission, global_windows, normalized_cxcywh, overlap_labels, sampled_frames)


def crop_geometry(box):
    centre = box[:2] + box[2:] / 2
    side = math.sqrt(float(box[2] * box[3])) * 4.
    left, top = centre - side / 2
    return np.asarray([left, top, side], np.float32)


def image_sizes(sequence, frames):
    sizes = []
    for frame in frames:
        with Image.open(sequence['rgb'][frame]) as rgb, Image.open(sequence['tir'][frame]) as tir:
            assert rgb.size == tir.size
            width, height = rgb.size
        sizes.append((height, width))
    return sizes


def verify_arrays(arrays, sequence, count, sizes):
    """Reconstruct supervision from actual candidate boxes and TRAIN annotations."""
    frames = sampled_frames(sequence, count)
    n = len(frames)
    shapes = dict(box=(n, 24, 4), features=(n, 24, 2, 768),
        coverage=(n, 24, 2), score=(n, 24), query=(n, 24),
        modality=(n, 24), geometry=(n, 12, 3), crop_boxes=(n, 12, 4),
        refinement_parents=(n, 2), frame=(n,), iou=(n, 24),
        labels=(n, 25), anchor=(2, 768), anchor_geometry=(3,),
        anchor_box=(4,), predictions=(count, 4))
    assert set(arrays) == set(shapes) and n > 0 and len(sizes) == n
    for key, shape in shapes.items():
        assert arrays[key].shape == shape, (sequence['name'], key)
        assert np.isfinite(arrays[key]).all(), (sequence['name'], key)
    assert arrays['features'].dtype == arrays['anchor'].dtype == np.float32
    assert arrays['coverage'].dtype == arrays['predictions'].dtype == np.float32
    assert arrays['labels'].dtype == np.int8 and arrays['frame'].dtype == np.int64
    assert arrays['frame'].tolist() == frames
    assert (arrays['predictions'][:, 2:] > 0).all()
    assert np.array_equal(arrays['predictions'][0], np.asarray(sequence['boxes'][0], np.float32))
    assert np.array_equal(arrays['anchor_geometry'], crop_geometry(arrays['predictions'][0]))
    assert np.array_equal(arrays['anchor_box'], normalized_cxcywh(
        np.asarray(sequence['boxes'][0], np.float32)[None], arrays['anchor_geometry'])[0])
    assert (arrays['geometry'][:, :, 2] > 0).all()
    assert (arrays['crop_boxes'][:, :, 2:] > 0).all()
    assert np.array_equal(arrays['query'], np.tile(np.repeat(np.arange(12), 2), (n, 1)))
    assert np.array_equal(arrays['modality'], np.tile([0, 1], (n, 12)))
    for index, frame in enumerate(frames):
        parents = np.argsort(-arrays['score'][index, 2:20], kind='mergesort')[:2] + 2
        assert np.array_equal(arrays['refinement_parents'][index], parents)
        previous = arrays['predictions'][frame - 1]
        crops = [previous] + global_windows(*sizes[index])
        for parent in parents:
            coarse = arrays['box'][index, parent]
            centre = coarse[:2] + coarse[2:] / 2
            crops.append(np.concatenate([centre - previous[2:] / 2, previous[2:]]).astype(np.float32))
        assert np.array_equal(arrays['crop_boxes'][index], np.asarray(crops, np.float32))
        assert np.array_equal(arrays['geometry'][index], np.stack([crop_geometry(box) for box in crops]))
        iou, labels = overlap_labels(
            arrays['box'][index], arrays['coverage'][index], sequence['boxes'][frame])
        assert np.array_equal(arrays['iou'][index], iou)
        assert np.array_equal(arrays['labels'][index], labels)
    return dict(sequence=sequence['name'], frames=count, candidate_sets=n,
        positive_sets=int((arrays['labels'][:, 1:] == 1).any(1).sum()),
        no_match_sets=int((arrays['labels'][:, 0] == 1).sum()),
        ambiguous_sets=int((arrays['labels'][:, 0] == -1).sum()),
        all_online_states_bit_equal=True)


def load_collection(root, relative, digest):
    """Require every native shard and artifact; smoke is never full acceptance."""
    path = root / relative
    assert sha(path) == digest
    plan = json.loads(path.read_bytes())
    assert plan['phase'] == 'full' and plan['shards'] == len(plan['jobs']) == 8
    check_admission(root, plan)
    manifest = json.loads((root / plan['manifest']).read_bytes())
    sequences = {row['name']: row for row in manifest['train']}
    fit, held = set(plan['fit_sequences']), set(plan['holdout_sequences'])
    assert len(fit) == 706 and len(held) == 176 and fit.isdisjoint(held)
    assert fit | held == set(sequences) == set(plan['sequence_names'])
    assert set(sequences).isdisjoint(set(manifest['test_sequences']) |
                                    {row['name'] for row in manifest['dev']})
    used = {relative: digest, plan['source_review']: plan['source_review_sha256']}
    parts = {part: {key: [] for key in ('anchor', 'features', 'coverage', 'labels')}
             for part in ('fit', 'holdout')}
    seen, results = set(), []
    for shard, job in enumerate(plan['jobs']):
        assert job['shard'] == shard
        directory = root / 'runs' / job['run']
        native_path = root / 'runs/pipeline_tasks' / job['task'] / 'status.json'
        native = json.loads(native_path.read_bytes())
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
        assert native['task'] == job['task'] and native['physical_gpu'] == job['assigned_gpu']
        command = [str(root / 'conda-envs/dual_rgbt/bin/python'),
            str(root / 'scripts/collect_e0_identity_train.py'), '--plan', relative,
            '--run', job['run'], '--task', job['task'], '--shard', str(shard)]
        assert native['command'] == command
        config = json.loads((directory / 'config.json').read_bytes())
        receipt_path = directory / 'COMPLETE.json'
        receipt = json.loads(receipt_path.read_bytes())
        names = plan['sequence_names'][shard::8]
        assert names == job['sequence_names'] == config['selected_sequences']
        assert config['native_command'] == command and config['plan'] == plan
        assert config['shard'] == shard and config['run'] == job['run']
        assert config['task'] == receipt['task'] == job['task']
        assert config['plan_sha256'] == receipt['plan_sha256'] == digest
        assert config['split'] == receipt['split'] == 'train'
        assert receipt['status'] == 'M2_IDENTITY_CACHE_COMPLETE'
        assert receipt['smoke_only'] is False and receipt['test_eligible'] is False
        assert receipt['checkpoint_sha256'] == plan['checkpoint_sha256']
        assert receipt['all_online_states_bit_equal'] is True
        assert receipt['optimizer_updates'] == receipt['backward_calls'] == 0
        assert receipt['frozen_state_tensors'] == 265 and receipt['candidate_count'] == 24
        expected = {'config.json', 'progress.json', 'current.json'} | {
            name + '.npz' for name in names}
        assert set(receipt['artifacts_sha256']) == expected
        assert {p.name for p in directory.glob('*.npz')} == {name + '.npz' for name in names}
        for artifact, artifact_sha in receipt['artifacts_sha256'].items():
            assert sha(directory / artifact) == artifact_sha
            used[(directory / artifact).relative_to(root).as_posix()] = artifact_sha
        for artifact in (native_path, receipt_path):
            used[artifact.relative_to(root).as_posix()] = sha(artifact)
        progress = json.loads((directory / 'progress.json').read_bytes())
        assert [row['sequence'] for row in progress] == names
        shard_results = []
        for name, row in zip(names, progress):
            assert name not in seen
            seen.add(name)
            part = 'fit' if name in fit else 'holdout'
            with np.load(directory / (name + '.npz'), allow_pickle=False) as arrays:
                count = len(sequences[name]['rgb'])
                sizes = image_sizes(sequences[name], sampled_frames(sequences[name], count))
                result = verify_arrays(arrays, sequences[name], count, sizes)
                assert row == dict(result, partition=part,
                                   tracked_frames=result['frames'] - 1)
                count = result['candidate_sets']
                parts[part]['anchor'].append(np.repeat(arrays['anchor'][None], count, axis=0))
                for key in ('features', 'coverage', 'labels'):
                    parts[part][key].append(arrays[key].copy())
            shard_results.append(result)
        assert receipt['sequences'] == len(names)
        assert receipt['frames'] == sum(row['frames'] for row in shard_results)
        assert receipt['candidate_sets'] == sum(row['candidate_sets'] for row in shard_results)
        assert receipt['anchor_encoder_calls'] == 2 * len(names)
        assert receipt['candidate_encoder_calls'] == 24 * receipt['candidate_sets']
        results.extend(shard_results)
    assert seen == set(sequences) and len(results) == 882
    assert sum(row['frames'] for row in results) == plan['expected_frames'] == 459435
    assert sum(row['candidate_sets'] for row in results) == plan['expected_candidate_sets'] == 18711
    data = {part: {key: torch.from_numpy(np.concatenate(values))
                   for key, values in fields.items()} for part, fields in parts.items()}
    report = dict(status='FULL_M2_IDENTITY_TRAIN_COLLECTION_ACCEPTED',
        split='train', test_eligible=False, scientific_result_audit_accepted=False,
        collection_plan=relative, collection_plan_sha256=digest,
        sequences=882, frames=459435, candidate_sets=18711,
        trajectories=results, inputs_sha256=used,
        full_image_pixel_hashes_verified=False,
        scope='All native shards/artifacts and every saved candidate label verified. '
              'Collector receipts attest paired online parity; it was not rerun. '
              'Overlap supervision is not semantic identity proof. Upstream E0 saw '
              'the component holdout. No online recovery or tracking improvement.')
    return data, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection-plan', required=True)
    parser.add_argument('--collection-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists()
    _, report = load_collection(ROOT, args.collection_plan, args.collection_sha256)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('trajectories', 'inputs_sha256')}), flush=True)


if __name__ == '__main__':
    main()
