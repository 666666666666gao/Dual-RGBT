"""Recompute the new M0 log schema against GT without loading neural weights."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_global_candidate_utility import clip_box, iou
from check_candidate_policy_sanity import sha


PROBE_FIELDS = ('quality', 'primary_box', 'queries', 'kinds', 'ranks', 'candidate_indices',
    'request_cell_indices', 'family_candidate_counts', 'gains', 'box', 'pixel_box', 'score', 'size', 'offset')


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'completion', 'steps', 'csv', 'manifest', 'headers', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_bytes())
    terminal = json.loads(Path(args.completion).read_bytes())
    complete, sanity, config, native = (terminal[key] for key in ('complete', 'sanity', 'config', 'native_task'))
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0 and native['physical_gpu'] == '1'
    assert native['command'] == [plan['remote_python'], plan['remote_root'] + '/scripts/train_scale_candidate_policy.py',
        '--plan', config['plan'], '--run', plan['runs']['sanity'], '--sanity']
    assert config['stage'] == complete['stage'] == 'scale_candidate_b' and config['sanity'] and complete['sanity']
    assert config['run'] == plan['runs']['sanity'] and config['plan_sha256'] == sha(args.plan)
    assert config['source_sha256'] == plan['source_sha256'] and config['input_sha256'] == plan['input_sha256']
    assert config['candidate_config'] == complete['candidate_config'] == plan['candidate_config']
    assert complete['status'] == 'COMPLETE' and complete['updates'] == sanity['optimizer_updates'] == 4
    assert complete['samples'] == 32 and complete['labelled_queries'] == 64
    assert complete['frozen_state_tensors'] == sanity['frozen_state_tensors'] == 257
    assert len(complete['changed_state_tensors']) == len(sanity['changed_state_tensors']) == 8
    assert complete['changed_state_tensors'] == sanity['changed_state_tensors']
    assert len(set(complete['changed_state_tensors'])) == 8
    assert all(name.startswith(('executor.quality.', 'query_gain.')) for name in complete['changed_state_tensors'])
    assert sanity['status'] == 'PASS' and sanity['strict_reload_prediction_bit_equal']
    assert sanity['saved_candidate_configuration_restored'] and sanity['four_request_legacy_bit_equal']
    assert tuple(sanity['probe_fields']) == PROBE_FIELDS
    assert sha(args.csv) == complete['private_labels_sha256']
    assert sha(args.manifest) == plan['input_sha256']['data/manifest.json']
    manifest = json.loads(Path(args.manifest).read_bytes())
    sequences = manifest['train']
    by_name = {row['name']: row for row in sequences}
    assert len(sequences) == len(by_name) == 882
    expected = []
    for index in range(32):
        rng = np.random.RandomState(plan['seed'] + index)
        sequence = sequences[int(rng.randint(len(sequences)))]
        valid = np.asarray(sequence['valid_frames'])
        frame = int(valid[int(rng.randint(2, len(valid)))])
        expected.append((sequence['name'], frame))
    assert complete['seen_train_sequences'] == sorted({name for name, frame in expected})
    headers = json.loads(Path(args.headers).read_bytes())
    assert headers['nn_calls'] == 0 and len(headers['items']) == 32
    sizes = {(name, frame): (height, width) for name, frame, width, height in headers['items']}
    assert len(sizes) == 32 and set(sizes) == set(expected)
    with Path(args.steps).open(newline='', encoding='utf-8') as handle:
        steps = list(csv.DictReader(handle))
    assert len(steps) == 4
    for index, row in enumerate(steps):
        assert int(row['epoch']) == 1 and int(row['step']) == index + 1 and int(row['kind']) == index
        values = [float(row[key]) for key in ('loss', 'quality_loss', 'gain_loss')]
        assert np.isfinite(values).all()
        assert values[0] == float(np.float32(np.float32(values[1]) + np.float32(values[2])))
    with Path(args.csv).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 64
    errors = dict(main_clip=0., query_clip=0., main_iou=0., query_iou=0., float32_gain=0.)
    counts = {}
    capacity = plan['candidate_config']['max_candidates']
    for ordinal, row in enumerate(rows):
        step, modality, kind = (int(row[key]) for key in ('step', 'modality', 'kind'))
        assert int(row['epoch']) == 1 and step == ordinal // 16 + 1
        assert modality == (ordinal % 16) // 8 and kind == step - 1
        name, frame = row['sequence'], int(row['frame'])
        assert (name, frame) == expected[(step - 1) * 8 + ordinal % 8]
        assert np.isfinite([float(row[key]) for key in ('main_iou', 'query_iou', 'delta_from_stop')]).all()
        target = np.asarray(by_name[name]['boxes'][frame], dtype=np.float32)
        main_raw, query_raw, main, query, main_geo, query_geo = (
            np.asarray(json.loads(row[key]), dtype=np.float32)
            for key in ('main_raw_box', 'query_raw_box', 'main_box', 'query_box', 'main_geometry', 'query_geometry'))
        assert all(np.isfinite(value).all() for value in (target, main_raw, query_raw, main, query, main_geo, query_geo))
        assert (target[2:] > 0).all() and main_geo[2] > 0
        assert query_geo[2] == main_geo[2] / (1 if kind == 3 else 2)
        height, width = sizes[name, frame]
        errors['main_clip'] = max(errors['main_clip'], float(np.abs(clip_box(main_raw, height, width) - main).max()))
        errors['query_clip'] = max(errors['query_clip'], float(np.abs(clip_box(query_raw, height, width) - query).max()))
        errors['main_iou'] = max(errors['main_iou'], abs(iou(main, target) - float(row['main_iou'])))
        errors['query_iou'] = max(errors['query_iou'], abs(iou(query, target) - float(row['query_iou'])))
        expected_gain = float(np.float32(np.float32(row['query_iou']) - np.float32(row['main_iou'])))
        errors['float32_gain'] = max(errors['float32_gain'], abs(expected_gain - float(row['delta_from_stop'])))
        rank, cell, family_count, unique_count = (int(row[key]) for key in
            ('candidate_rank', 'request_cell', 'family_candidate_count', 'unique_candidate_count'))
        assert 1 <= unique_count <= 2 * capacity
        if kind < 2:
            assert rank == cell == -1 and family_count == 1
        else:
            assert 0 <= rank < family_count <= capacity and family_count <= unique_count
            assert 0 <= cell < 1024
            centre = ((np.asarray([cell % 32, cell // 32], dtype=np.float32) + np.float32(.5)) /
                      np.float32(32) * np.float32(max(height, width)))
            assert 0 <= centre[0] < width and 0 <= centre[1] < height
            expected_geometry = np.concatenate([centre - query_geo[2] / np.float32(2), query_geo[2:3]])
            assert np.array_equal(query_geo, expected_geometry)
        key = '%d:%d' % (modality, kind)
        counts[key] = counts.get(key, 0) + 1
    assert counts == complete['route_samples'] == {'%d:%d' % (role, kind): 8 for role in range(2) for kind in range(4)}
    assert max(errors.values()) <= 1e-6
    causal = complete['causal_probe']
    assert causal['tracked_frames'] == 10 and causal['selected_execution_bit_equal']
    assert causal['frozen_model_state_tensors'] == 265
    assert causal['actual_auxiliary_queries'] == causal['replay_auxiliary_queries']
    result = dict(status='SCALE_SANITY_CPU_CHECK_PASS', samples=32, labelled_queries=64, updates=4,
        nn_calls=0, errors=errors, checked_family_samples=counts, candidate_config=plan['candidate_config'],
        input_sha256={name: sha(getattr(args, name)) for name in ('plan', 'completion', 'steps', 'csv', 'manifest', 'headers')},
        analyzer_sha256=sha(__file__),
        scope='GT, image-clipped boxes, finite scalar labels, float32 gains/loss sums and new cell/count geometry are recomputed. '
              'Native execution/gradients/optimizer/frozen weights/reload/replay remain supplied runtime assertions. '
              'This checker does not authenticate provenance or establish NN performance, metrics or latency.')
    output = Path(args.output)
    assert not output.exists()
    output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
