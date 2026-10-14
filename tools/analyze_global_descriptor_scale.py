"""Recompute frozen descriptor-scale ROI coverage without image or model execution."""
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
    for name in ('plan', 'completion', 'manifest', 'content_csv', 'headers', 'baseline', 'comparisons', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    terminal = json.loads(Path(args.completion).read_text(encoding='utf-8'))
    complete = terminal['complete']
    assert terminal['native_task']['state'] == 'COMPLETE' and terminal['native_task']['exit_code'] == 0
    assert terminal['config']['plan_sha256'] == sha(args.plan)
    assert not complete['sanity'] and complete['states'] == 1752 and complete['comparisons'] == 3504
    assert complete['frozen_state_tensors'] == 265 and complete['optimizer_updates'] == complete['backward_calls'] == 0
    for field, remote in (('manifest', 'data/manifest.json'), ('content_csv', plan['content_csv']),
                          ('baseline', plan['baseline_ranking'])):
        assert sha(getattr(args, field)) == plan['input_sha256'][remote]
    assert sha(args.comparisons) == complete['private_comparisons_sha256']
    assert sha(args.headers) == '03b13b68e9332eae630f88a291f879f8523dbbb7c8b9ed727bdc3123f337b943'
    sequences = {r['name']: r for r in json.loads(Path(args.manifest).read_text(encoding='utf-8'))['train']}
    with Path(args.content_csv).open(newline='', encoding='utf-8') as handle:
        original = [r for r in csv.DictReader(handle) if r['region'] == 'GLOBAL_WIDE' and r['content'] == 'TRUE']
    headers = json.loads(Path(args.headers).read_bytes())
    baseline = json.loads(Path(args.baseline).read_bytes())
    rows = json.loads(Path(args.comparisons).read_bytes())
    assert headers['nn_calls'] == 0 and len(headers['items']) == len(original) == len(baseline) == 1752
    assert len(rows) == 3504
    groups, maximum_error = {}, 0.
    for resolution in (64, 32):
        selected = rows[0 if resolution == 64 else 1::2]
        for index, (row, old, previous, header) in enumerate(zip(selected, original, baseline, headers['items'])):
            name, frame, width, height = header
            assert row['state'] == int(old['state']) == previous['state'] == index
            assert (row['sequence'], row['frame']) == (name, frame) == (old['sequence'], int(old['frame']))
            assert row['resolution'] == resolution and row['auxiliary_modality'] == 1 - int(old['modality'])
            assert row['native_main_outside'] == previous['native_main_outside'] == (old['main_centre_inside'] == 'False')
            point = np.asarray(row['peak'], dtype=np.float32)
            axis = (np.arange(32, dtype=np.float32) + .5) * max(width, height) / 32
            assert point.shape == (2,) and np.isfinite(point).all()
            assert all(float(np.abs(axis - p).min()) < 1e-3 for p in point)
            assert 0 <= point[0] < width and 0 <= point[1] < height
            assert isinstance(row['target_nearest_cell_rank'], int)
            assert 0 <= row['target_nearest_cell_rank'] < (axis < width).sum() * (axis < height).sum()
            assert np.isfinite(row['score_margin']) and row['score_margin'] <= 0
            if resolution == 64:
                for key in ('peak', 'target_nearest_cell_rank', 'score_margin', 'local_centre_inside',
                            'local_coverage', 'wide_centre_inside', 'wide_coverage'):
                    assert row[key] == previous[key], key
            target = np.asarray(sequences[name]['boxes'][frame], dtype=np.float32)
            centre = target[:2] + target[2:] / 2
            geo = np.asarray(json.loads(old['query_geometry']), dtype=np.float32)
            for region, side in (('local', geo[2] / 2), ('wide', geo[2])):
                left = point - side / 2
                inside = bool(((centre >= left) & (centre < left + side)).all())
                coverage = float(np.maximum(0, np.minimum(target[:2] + target[2:], left + side) - np.maximum(target[:2], left)).prod() / target[2:].prod())
                assert row[region + '_centre_inside'] == inside
                error = abs(row[region + '_coverage'] - coverage)
                maximum_error = max(maximum_error, error)
                assert error <= 1e-6
        groups[str(resolution)] = {}
        for scope in ('all', 'native_main_outside'):
            subset = [r for r in selected if scope == 'all' or r['native_main_outside']]
            assert len(subset) == (1752 if scope == 'all' else 239)
            group = dict(states=len(subset),
                         mean_nearest_cell_rank=float(np.mean([r['target_nearest_cell_rank'] for r in subset])),
                         mean_score_margin=float(np.mean([r['score_margin'] for r in subset])))
            for region in ('local', 'wide'):
                group[region] = dict(centre_inside=sum(r[region + '_centre_inside'] for r in subset),
                                     mean_gt_coverage=float(np.mean([r[region + '_coverage'] for r in subset])))
            assert group == complete['per_resolution'][str(resolution)][scope]
            groups[str(resolution)][scope] = group
    result = dict(status='CPU_ANALYSIS_PASS', states=1752, comparisons=3504, nn_calls=0,
                  baseline64_rows_reproduced=True, coverage_recomputed_max_absolute_error=maximum_error,
                  per_resolution=groups, input_sha256={name: sha(getattr(args, name)) for name in
                  ('plan', 'completion', 'manifest', 'content_csv', 'headers', 'baseline', 'comparisons')},
                  analyzer_sha256=sha(__file__),
                  scope='Paired frozen train-state query geometry only. Main-outside membership and CNN ranks/margins remain native-derived. Resolution changes receptive-field context too; no isolation of scale alone, query execution, official metrics or latency claim.')
    output = Path(args.output)
    assert not output.exists()
    output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
