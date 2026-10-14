"""Recompute fixed-state global ROI coverage; ranking scores remain native-derived."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for field in ('plan', 'completion', 'manifest', 'content_csv', 'headers', 'before', 'after', 'output'):
        parser.add_argument('--' + field.replace('_', '-'), required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    terminal = json.loads(Path(args.completion).read_text(encoding='utf-8'))
    assert terminal['native_task']['state'] == 'COMPLETE' and terminal['native_task']['exit_code'] == 0
    assert terminal['config']['plan_sha256'] == sha(args.plan)
    assert terminal['config']['input_checkpoint_sha256'] == plan['input_checkpoint_sha256']
    assert not terminal['complete']['sanity'] and terminal['complete']['updates'] == 6250
    assert terminal['complete']['samples'] == 50000 and terminal['complete']['sequences_seen'] == 882
    assert terminal['complete']['frozen_state_tensors'] == 259
    assert len(terminal['complete']['changed_state_tensors']) == 6
    assert terminal['complete']['strict_reload_probe_bit_equal'] and len(terminal['complete']['probe_fields']) == 7
    assert sha(args.manifest) == plan['manifest_sha256']
    assert sha(args.content_csv) == plan['diagnostic_csv_sha256']
    assert sha(args.headers) == '03b13b68e9332eae630f88a291f879f8523dbbb7c8b9ed727bdc3123f337b943'
    sequences = {row['name']: row for row in json.loads(Path(args.manifest).read_text(encoding='utf-8'))['train']}
    with Path(args.content_csv).open(newline='', encoding='utf-8') as handle:
        original = [row for row in csv.DictReader(handle) if row['region'] == 'GLOBAL_WIDE' and row['content'] == 'TRUE']
    headers = json.loads(Path(args.headers).read_bytes())
    assert headers['nn_calls'] == 0 and len(headers['items']) == len(original) == 1752
    result = dict(status='CPU_ANALYSIS_PASS', states=1752, input_sha256={},
                  scope='Paired fixed train-state proposal coverage only. Main-outside labels and score/rank values remain native-derived; no new tracking trajectory, official metrics, generalization, test selection or inference-cost claim.',
                  per_checkpoint={})
    maximum_error = 0.
    before_points, after_points = [], []
    for label, path in (('before', args.before), ('after', args.after)):
        assert sha(path) == terminal['complete'][label + '_ranking']['private_sha256']
        rows = json.loads(Path(path).read_text(encoding='utf-8'))
        assert len(rows) == 1752
        for index, (row, old, header) in enumerate(zip(rows, original, headers['items'])):
            name, frame, width, height = header
            assert row['state'] == int(old['state']) == index
            assert (row['sequence'], row['frame']) == (name, frame) == (old['sequence'], int(old['frame']))
            assert row['auxiliary_modality'] == 1 - int(old['modality'])
            assert row['native_main_outside'] == (old['main_centre_inside'] == 'False')
            peak = np.asarray(row['peak'], dtype=np.float32)
            assert peak.shape == (2,) and np.isfinite(peak).all()
            side = float(max(width, height))
            axis = (np.arange(32, dtype=np.float32) + .5) * side / 32
            assert all(float(np.abs(axis - point).min()) < 1e-3 for point in peak)
            assert 0 <= peak[0] < width and 0 <= peak[1] < height
            valid_count = int((axis < width).sum() * (axis < height).sum())
            assert isinstance(row['target_nearest_cell_rank'], int)
            assert 0 <= row['target_nearest_cell_rank'] < valid_count
            assert np.isfinite(row['score_margin']) and row['score_margin'] <= 0
            assert (row['target_nearest_cell_rank'] == 0) == (row['score_margin'] == 0)
            geo = np.asarray(json.loads(old['query_geometry']), dtype=np.float32)
            if label == 'before':
                assert np.max(np.abs(peak - (geo[:2] + geo[2] / 2))) < 1e-3
            target = np.asarray(sequences[name]['boxes'][frame], dtype=np.float32)
            centre = target[:2] + target[2:] / 2
            for region, budget in (('local', geo[2] / 2), ('wide', geo[2])):
                left = peak - budget / 2
                inside = bool(((centre >= left) & (centre < left + budget)).all())
                coverage = float(np.maximum(0, np.minimum(target[:2] + target[2:], left + budget) - np.maximum(target[:2], left)).prod() / target[2:].prod())
                assert row[region + '_centre_inside'] == inside
                error = abs(row[region + '_coverage'] - coverage)
                maximum_error = max(maximum_error, error)
                assert error <= 1e-6
            (before_points if label == 'before' else after_points).append(peak)
        groups = {}
        for scope, selected in (('all', rows), ('native_main_outside', [row for row in rows if row['native_main_outside']])):
            assert len(selected) == (1752 if scope == 'all' else 239)
            group = dict(states=len(selected),
                         mean_nearest_cell_rank=float(np.mean([row['target_nearest_cell_rank'] for row in selected])),
                         median_nearest_cell_rank=float(np.median([row['target_nearest_cell_rank'] for row in selected])),
                         mean_score_margin=float(np.mean([row['score_margin'] for row in selected])))
            for region in ('local', 'wide'):
                group[region] = dict(centre_inside=sum(row[region + '_centre_inside'] for row in selected),
                                     mean_gt_coverage=float(np.mean([row[region + '_coverage'] for row in selected])))
            assert group == terminal['complete'][label + '_ranking']['groups'][scope]
            groups[scope] = group
        result['per_checkpoint'][label] = groups
    for field in ('plan', 'completion', 'manifest', 'content_csv', 'headers', 'before', 'after'):
        result['input_sha256'][field] = sha(getattr(args, field))
    result['input_sha256']['analyzer'] = sha(__file__)
    result['coverage_recomputed_max_absolute_error'] = maximum_error
    result['original_before_proposals_reproduced'] = True
    result['changed_proposal_states'] = int((np.asarray(before_points) != np.asarray(after_points)).any(1).sum())
    output = Path(args.output)
    assert not output.exists()
    output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

