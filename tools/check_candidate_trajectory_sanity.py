"""Recompute causal controller M0 query labels using dataset GT and actual image headers."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_global_candidate_utility import clip_box, iou
from check_candidate_policy_sanity import sha


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'completion', 'steps', 'csv', 'manifest', 'headers', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_bytes())
    terminal = json.loads(Path(args.completion).read_bytes())
    complete, sanity, config, native = (terminal[key] for key in ('complete', 'sanity', 'config', 'native_task'))
    assert plan['status'] == 'READY' and native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert config['stage'] == 'candidate_c' and config['sanity']
    assert config['plan_sha256'] == sha(args.plan) and config['source_sha256'] == plan['source_sha256']
    assert config['input_sha256'] == plan['input_sha256'] and complete['sanity']
    assert {key: complete[key] for key in plan['sanity_expected_counts']} == plan['sanity_expected_counts']
    assert complete['selected_execution_bit_equal'] and complete['teacher_preserves_tracker_state']
    assert complete['frozen_state_tensors'] == sanity['frozen_state_tensors'] == 257
    assert len(complete['changed_state_tensors']) == len(sanity['changed_state_tensors']) == 8
    assert sanity['strict_reload_prediction_bit_equal'] and sanity['four_request_legacy_bit_equal']
    assert tuple(sanity['probe_fields']) == ('quality', 'primary_box', 'queries', 'kinds', 'ranks', 'candidate_indices',
                                           'gains', 'box', 'pixel_box', 'score', 'size', 'offset')
    assert sha(args.csv) == complete['private_labels_sha256']
    assert sha(args.manifest) == plan['input_sha256']['data/manifest.json']
    manifest = json.loads(Path(args.manifest).read_bytes())
    train = {row['name']: row for row in manifest['train']}
    assert len(train) == 882 and set(train) == set(plan['sequence_names'])
    sequence = train[plan['sequence_names'][0]]
    expected_frames = [frame for frame in sequence['valid_frames'] if frame > 0 and frame % plan['label_stride'] == 0][:4]
    assert expected_frames == [5, 10, 15, 20]
    headers = json.loads(Path(args.headers).read_bytes())
    assert headers['nn_calls'] == 0
    sizes = {(name, frame): (height, width) for name, frame, width, height in headers['items']}
    assert len(sizes) == len(headers['items']) == 4
    assert set(sizes) == {(sequence['name'], frame) for frame in expected_frames}
    with Path(args.steps).open(newline='', encoding='utf-8') as handle:
        steps = list(csv.DictReader(handle))
    with Path(args.csv).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(steps) == plan['sanity_updates'] == 4
    assert len(rows) == complete['labelled_queries'] == 8 + complete['extra_selected_query_labels']
    errors = dict(main_clip=0., query_clip=0., main_iou=0., query_iou=0., float32_gain=0.)
    counts, extra_queries = {}, 0
    for index, step_row in enumerate(steps):
        step, frame = index + 1, expected_frames[index]
        assert int(step_row['step']) == step and int(step_row['round']) == 1
        assert step_row['sequence'] == sequence['name'] and int(step_row['frame']) == frame
        assert np.isfinite([float(step_row[key]) for key in ('loss', 'quality_loss', 'gain_loss')]).all()
        expected_loss = float(np.float32(np.float32(step_row['quality_loss']) + np.float32(step_row['gain_loss'])))
        assert float(step_row['loss']) == expected_loss
        sampled_kind = int(step_row['sampled_kind'])
        assert sampled_kind == index
        group = [row for row in rows if int(row['step']) == step]
        assert [row['purpose'] for row in group[:2]] == ['sampled', 'sampled']
        assert [int(row['modality']) for row in group[:2]] == [0, 1]
        extra = step_row['extra_selected_query'] == 'True'
        assert step_row['extra_selected_query'] in ('True', 'False') and len(group) == 2 + int(extra)
        if extra:
            assert group[2]['purpose'] == 'selected_extra'
            assert int(group[2]['modality']) == int(step_row['chosen_modality'])
            assert int(group[2]['kind']) + 1 == int(step_row['chosen_action'])
            assert int(group[2]['candidate_rank']) == int(step_row['chosen_rank'])
            matched = group[int(step_row['chosen_modality'])]
            assert (int(matched['kind']), int(matched['candidate_rank'])) != (int(group[2]['kind']), int(group[2]['candidate_rank']))
            for field in ('main_geometry', 'main_raw_box', 'main_box', 'main_iou'):
                assert group[2][field] == matched[field]
            extra_queries += 1
        elif int(step_row['chosen_action']):
            matched = group[int(step_row['chosen_modality'])]
            assert int(matched['kind']) + 1 == int(step_row['chosen_action'])
            assert int(matched['candidate_rank']) == int(step_row['chosen_rank'])
        else:
            assert int(step_row['chosen_rank']) == -1
        target = np.asarray(sequence['boxes'][frame], dtype=np.float32)
        height, width = sizes[sequence['name'], frame]
        for row in group:
            assert int(row['round']) == 1 and row['sequence'] == sequence['name'] and int(row['frame']) == frame
            assert np.isfinite([float(row[key]) for key in ('main_iou', 'query_iou', 'delta_from_stop')]).all()
            role, kind, rank, count = (int(row[key]) for key in ('modality', 'kind', 'candidate_rank', 'candidate_count'))
            assert role in (0, 1) and kind in (0, 1, 2, 3) and 1 <= count <= plan['max_candidates']
            assert rank == -1 if kind < 2 else 0 <= rank < count
            if row['purpose'] == 'sampled':
                assert kind == sampled_kind
                family = '%d:%d' % (role, kind)
                counts[family] = counts.get(family, 0) + 1
            main_raw, query_raw, main, query, main_geo, query_geo = (
                np.asarray(json.loads(row[key]), dtype=np.float32)
                for key in ('main_raw_box', 'query_raw_box', 'main_box', 'query_box', 'main_geometry', 'query_geometry'))
            assert all(np.isfinite(value).all() for value in (target, main_raw, query_raw, main, query, main_geo, query_geo))
            assert query_geo[2] == main_geo[2] / (1 if kind == 3 else 2)
            errors['main_clip'] = max(errors['main_clip'], float(np.abs(clip_box(main_raw, height, width) - main).max()))
            errors['query_clip'] = max(errors['query_clip'], float(np.abs(clip_box(query_raw, height, width) - query).max()))
            errors['main_iou'] = max(errors['main_iou'], abs(iou(main, target) - float(row['main_iou'])))
            errors['query_iou'] = max(errors['query_iou'], abs(iou(query, target) - float(row['query_iou'])))
            expected_gain = float(np.float32(np.float32(row['query_iou']) - np.float32(row['main_iou'])))
            errors['float32_gain'] = max(errors['float32_gain'], abs(expected_gain - float(row['delta_from_stop'])))
    assert counts == complete['sampled_family_counts'] == {'%d:%d' % (role, kind): 1 for role in range(2) for kind in range(4)}
    assert complete['sampled_query_labels'] == 8 and extra_queries == complete['extra_selected_query_labels']
    assert max(errors.values()) <= 1e-6
    result = dict(status='REAL_CAUSAL_SANITY_CPU_PASS', tracked_frames=20, updates=4, labelled_queries=len(rows),
        extra_selected_query_labels=extra_queries, nn_calls=0, errors=errors, sampled_family_counts=counts,
        input_sha256={name: sha(getattr(args, name)) for name in ('plan', 'completion', 'steps', 'csv', 'manifest', 'headers')},
        analyzer_sha256=sha(__file__),
        scope='Dataset GT, image-clipped boxes and signed float32 query benefits are recomputed from actual C M0 records. '+
              'Causal ordering, selected replay, gradients, frozen executor and strict reload remain native runtime/source assertions. '+
              'No final accuracy or inference latency claim.')
    output = Path(args.output)
    assert not output.exists()
    output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
