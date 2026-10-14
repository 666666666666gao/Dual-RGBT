"""Reconstruct greedy candidates and single-ROI oracle geometry on CPU."""
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
    for key in ('plan', 'completion', 'manifest', 'content_csv', 'headers', 'baseline', 'candidates', 'scores', 'output'):
        parser.add_argument('--' + key.replace('_', '-'), required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    terminal = json.loads(Path(args.completion).read_text(encoding='utf-8'))
    complete = terminal['complete']
    assert terminal['native_task']['state'] == 'COMPLETE' and terminal['native_task']['exit_code'] == 0
    assert terminal['config']['plan_sha256'] == sha(args.plan)
    assert not complete['sanity'] and complete['states'] == 1752
    assert complete['frozen_state_tensors'] == 265 and complete['optimizer_updates'] == complete['backward_calls'] == 0
    for field, remote in (('manifest', 'data/manifest.json'), ('content_csv', plan['content_csv']), ('baseline', plan['baseline_ranking'])):
        assert sha(getattr(args, field)) == plan['input_sha256'][remote]
    assert sha(args.candidates) == complete['private_candidates_sha256']
    assert sha(args.scores) == complete['private_scores_sha256']
    assert sha(args.headers) == '03b13b68e9332eae630f88a291f879f8523dbbb7c8b9ed727bdc3123f337b943'
    sequences = {r['name']: r for r in json.loads(Path(args.manifest).read_text(encoding='utf-8'))['train']}
    with Path(args.content_csv).open(newline='', encoding='utf-8') as handle:
        original = [r for r in csv.DictReader(handle) if r['region'] == 'GLOBAL_WIDE' and r['content'] == 'TRUE']
    headers = json.loads(Path(args.headers).read_bytes())
    baseline = json.loads(Path(args.baseline).read_bytes())
    rows = json.loads(Path(args.candidates).read_bytes())
    scores = np.load(args.scores, allow_pickle=False)
    assert headers['nn_calls'] == 0 and len(original) == len(baseline) == len(rows) == len(headers['items']) == 1752
    assert scores.dtype == np.float32 and scores.shape == (1752, 1024)
    maximum_error = 0.
    for index, (row, old, previous, header) in enumerate(zip(rows, original, baseline, headers['items'])):
        name, frame, width, height = header
        assert row['state'] == int(old['state']) == index
        assert (row['sequence'], row['frame']) == (old['sequence'], int(old['frame'])) == (name, frame)
        assert row['auxiliary_modality'] == 1 - int(old['modality'])
        assert row['native_main_outside'] == previous['native_main_outside'] == (old['main_centre_inside'] == 'False')
        cells = np.arange(1024)
        coordinates = (np.stack([cells % 32, cells // 32], 1).astype(np.float32) + .5) / 32 * max(width, height)
        valid = (coordinates[:, 0] < width) & (coordinates[:, 1] < height)
        assert np.isfinite(scores[index, valid]).all() and np.isneginf(scores[index, ~valid]).all()
        assert (np.abs(scores[index, valid]) <= 1.000001).all()
        geometry = np.asarray(json.loads(old['query_geometry']), dtype=np.float32)
        local_side = float(geometry[2] / 2)
        remaining, indices = scores[index].copy(), []
        for _ in range(8):
            peak = int(remaining.argmax())
            if not np.isfinite(remaining[peak]):
                break
            indices.append(peak)
            distance = ((coordinates - coordinates[peak]) ** 2).sum(1)
            remaining[distance < (local_side / 2) ** 2] = -np.inf
        assert row['indices'] == indices and row['candidate_count'] == len(indices)
        points = coordinates[indices]
        assert np.max(np.abs(points - np.asarray(row['points'], dtype=np.float32))) == 0
        assert np.max(np.abs(points[0] - np.asarray(previous['peak'], dtype=np.float32))) == 0
        target = np.asarray(sequences[name]['boxes'][frame], dtype=np.float32)
        centre = target[:2] + target[2:] / 2
        for k in (1, 2, 4, 8):
            for region, side in (('local', local_side), ('wide', float(geometry[2]))):
                left = points[:k] - side / 2
                inside = bool(((centre >= left) & (centre < left + side)).all(1).any())
                coverage = np.maximum(0, np.minimum(target[:2] + target[2:], left + side) - np.maximum(target[:2], left)).prod(1) / target[2:].prod()
                native = row['by_k'][str(k)][region]
                assert native['centre_inside'] == inside and native['oracle_index'] == int(coverage.argmax())
                error = abs(native['best_gt_coverage'] - float(coverage.max()))
                maximum_error = max(maximum_error, error)
                assert error <= 1e-6
    groups = {}
    for scope in ('all', 'native_main_outside'):
        selected = [r for r in rows if scope == 'all' or r['native_main_outside']]
        assert len(selected) == (1752 if scope == 'all' else 239)
        group = dict(states=len(selected), candidate_count_histogram={str(n): sum(r['candidate_count'] == n for r in selected)
            for n in range(1, 9)}, by_k={})
        for k in (1, 2, 4, 8):
            group['by_k'][str(k)] = {region: dict(
                centre_inside=sum(r['by_k'][str(k)][region]['centre_inside'] for r in selected),
                mean_best_gt_coverage=float(np.mean([r['by_k'][str(k)][region]['best_gt_coverage'] for r in selected])))
                for region in ('local', 'wide')}
        assert group == complete['per_scope'][scope]
        groups[scope] = group
    result = dict(status='CPU_ANALYSIS_PASS', states=1752, nn_calls=0,
        greedy_candidates_reconstructed=True, baseline_top1_reproduced=True,
        geometry_max_absolute_error=maximum_error, per_scope=groups,
        input_sha256={field: sha(getattr(args, field)) for field in
            ('plan', 'completion', 'manifest', 'content_csv', 'headers', 'baseline', 'candidates', 'scores')},
        analyzer_sha256=sha(__file__),
        scope='Frozen training-state best single-ROI oracle geometry only. Candidate choices reconstructed from native CNN score maps; main-outside membership remains native-derived. No deployable selection, auxiliary execution, joint coverage, tracking upper bound, official metrics or latency claim.')
    output = Path(args.output)
    assert not output.exists()
    output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
