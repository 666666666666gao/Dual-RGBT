"""Verify real sampled controller labels against dataset GT and image headers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_global_candidate_utility import clip_box, iou


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'completion', 'csv', 'manifest', 'headers', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_bytes())
    terminal = json.loads(Path(args.completion).read_bytes())
    complete, sanity = terminal['complete'], terminal['sanity']
    assert terminal['native_task']['state'] == 'COMPLETE' and terminal['native_task']['exit_code'] == 0
    assert terminal['config']['plan_sha256'] == sha(args.plan) and terminal['config']['source_sha256'] == plan['source_sha256']
    assert complete['sanity'] and complete['updates'] == 4 and complete['samples'] == 32
    assert complete['frozen_state_tensors'] == 257 and len(complete['changed_state_tensors']) == 8
    assert sanity['strict_reload_prediction_bit_equal'] and sanity['four_request_legacy_bit_equal']
    assert tuple(sanity['probe_fields']) == ('quality', 'primary_box', 'queries', 'kinds', 'ranks', 'candidate_indices',
                                           'gains', 'box', 'pixel_box', 'score', 'size', 'offset')
    assert sha(args.csv) == complete['private_labels_sha256']
    assert sha(args.manifest) == plan['input_sha256']['data/manifest.json']
    manifest = json.loads(Path(args.manifest).read_bytes())
    sequences = manifest['train']
    assert len(sequences) == 882
    by_name = {row['name']: row for row in sequences}
    headers = json.loads(Path(args.headers).read_bytes())
    assert headers['nn_calls'] == 0 and len(headers['items']) == 32
    sizes = {(name, frame): (height, width) for name, frame, width, height in headers['items']}
    expected = []
    for index in range(32):
        rng = np.random.RandomState(plan['seed'] + index)
        sequence = sequences[int(rng.randint(len(sequences)))]
        valid = np.asarray(sequence['valid_frames'])
        frame = int(valid[int(rng.randint(2, len(valid)))])
        expected.append((sequence['name'], frame))
    with Path(args.csv).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 64
    errors = dict(main_clip=0., query_clip=0., main_iou=0., query_iou=0., float32_gain=0.)
    counts = {}
    for ordinal, row in enumerate(rows):
        step, modality, kind = int(row['step']), int(row['modality']), int(row['kind'])
        assert step == ordinal // 16 + 1 and modality == (ordinal % 16) // 8 and kind == step - 1
        name, frame = row['sequence'], int(row['frame'])
        assert (name, frame) == expected[(step - 1) * 8 + ordinal % 8]
        height, width = sizes[name, frame]
        target = np.asarray(by_name[name]['boxes'][frame], dtype=np.float32)
        main_raw = np.asarray(json.loads(row['main_raw_box']), dtype=np.float32)
        query_raw = np.asarray(json.loads(row['query_raw_box']), dtype=np.float32)
        main = np.asarray(json.loads(row['main_box']), dtype=np.float32)
        query = np.asarray(json.loads(row['query_box']), dtype=np.float32)
        geometry = np.asarray(json.loads(row['query_geometry']), dtype=np.float32)
        main_geometry = np.asarray(json.loads(row['main_geometry']), dtype=np.float32)
        assert all(np.isfinite(box).all() for box in (target, main_raw, query_raw, main, query, geometry, main_geometry))
        errors['main_clip'] = max(errors['main_clip'], float(np.abs(clip_box(main_raw, height, width) - main).max()))
        errors['query_clip'] = max(errors['query_clip'], float(np.abs(clip_box(query_raw, height, width) - query).max()))
        errors['main_iou'] = max(errors['main_iou'], abs(iou(main, target) - float(row['main_iou'])))
        errors['query_iou'] = max(errors['query_iou'], abs(iou(query, target) - float(row['query_iou'])))
        expected_gain = float(np.float32(np.float32(row['query_iou']) - np.float32(row['main_iou'])))
        errors['float32_gain'] = max(errors['float32_gain'], abs(expected_gain - float(row['delta_from_stop'])))
        assert geometry[2] == main_geometry[2] / (1 if kind == 3 else 2)
        rank, count = int(row['candidate_rank']), int(row['candidate_count'])
        assert 1 <= count <= plan['max_candidates']
        assert rank == -1 if kind < 2 else 0 <= rank < count
        key = str(modality) + ':' + str(kind)
        counts[key] = counts.get(key, 0) + 1
    assert counts == complete['route_samples'] and all(value == 8 for value in counts.values())
    assert max(errors.values()) <= 1e-6
    causal = complete['causal_probe']
    assert causal['tracked_frames'] == 10 and causal['selected_execution_bit_equal'] and causal['frozen_model_state_tensors'] == 265
    assert causal['actual_auxiliary_queries'] == causal['replay_auxiliary_queries']
    result = dict(status='REAL_SANITY_CPU_PASS', samples=32, labelled_queries=64, nn_calls=0,
        errors=errors, actual_route_counts=counts, causal_probe=causal,
        input_sha256={name: sha(getattr(args, name)) for name in ('plan', 'completion', 'csv', 'manifest', 'headers')},
        analyzer_sha256=sha(__file__),
        scope='Actual M0 labels, dataset GT, raw/image-clipped boxes and float32 signed IoU are independently recomputed. '+
              'GPU gradient/optimizer/freeze/reload/causal replay facts remain native runtime assertions; no final performance claim.')
    output = Path(args.output)
    assert not output.exists()
    output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
