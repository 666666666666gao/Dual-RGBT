"""Post-hoc CPU geometric opportunity control with equal paired candidate counts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def random_peaks(priority, points, radius):
    indices = []
    for _ in range(8):
        peak = int(priority.argmax())
        if not np.isfinite(priority[peak]):
            break
        indices.append(peak)
        priority[((points - points[peak]) ** 2).sum(1) < radius ** 2] = -np.inf
    return indices


def opportunity(points, target, side):
    left = points - side / 2
    centre = target[:2] + target[2:] / 2
    inside = bool(((centre >= left) & (centre < left + side)).all(1).any())
    coverage = np.maximum(0, np.minimum(target[:2] + target[2:], left + side) - np.maximum(target[:2], left)).prod(1) / target[2:].prod()
    return inside, float(coverage.max())


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'manifest', 'content_csv', 'headers', 'candidates', 'completion', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_bytes())
    assert plan['seeds'] == list(range(2027, 2037)) and plan['candidate_limits'] == [1, 2, 4, 8]
    assert plan['analyzer_sha256'] == sha(__file__)
    for field in ('manifest', 'content_csv', 'headers', 'candidates', 'completion'):
        assert sha(getattr(args, field)) == plan['input_sha256'][field]
    complete = json.loads(Path(args.completion).read_bytes())['complete']
    assert not complete['sanity'] and complete['states'] == 1752
    assert sha(args.candidates) == complete['private_candidates_sha256']
    manifest = json.loads(Path(args.manifest).read_bytes())
    sequences = {r['name']: r for r in manifest['train']}
    with Path(args.content_csv).open(newline='', encoding='utf-8') as handle:
        original = [r for r in csv.DictReader(handle) if r['region'] == 'GLOBAL_WIDE' and r['content'] == 'TRUE']
    headers = json.loads(Path(args.headers).read_bytes())['items']
    rows = json.loads(Path(args.candidates).read_bytes())
    assert len(original) == len(headers) == len(rows) == 1752
    per_seed = []
    for seed in plan['seeds']:
        rng = np.random.RandomState(seed)
        totals = {scope: {str(k): dict(states=0, effective_count_histogram={},
            cnn={r: dict(centre_inside=0, coverage_sum=0.) for r in ('local', 'wide')},
            random={r: dict(centre_inside=0, coverage_sum=0.) for r in ('local', 'wide')})
            for k in plan['candidate_limits']} for scope in ('all', 'native_main_outside')}
        for index, (row, old, header) in enumerate(zip(rows, original, headers)):
            name, frame, width, height = header
            assert row['state'] == int(old['state']) == index
            assert (row['sequence'], row['frame']) == (name, frame) == (old['sequence'], int(old['frame']))
            assert row['native_main_outside'] == (old['main_centre_inside'] == 'False')
            cells = np.arange(1024)
            grid = (np.stack([cells % 32, cells // 32], 1).astype(np.float32) + .5) / 32 * max(width, height)
            valid = (grid[:, 0] < width) & (grid[:, 1] < height)
            wide_side = float(np.asarray(json.loads(old['query_geometry']), dtype=np.float32)[2])
            priority = rng.random_sample(1024).astype(np.float32)
            priority[~valid] = -np.inf
            # Random proposals depend on geometry and seeded priority, never target annotations.
            random_points = grid[random_peaks(priority, grid, wide_side / 4)]
            cnn_points = np.asarray(row['points'], dtype=np.float32)
            target = np.asarray(sequences[name]['boxes'][frame], dtype=np.float32)
            for k in plan['candidate_limits']:
                count = min(k, len(cnn_points), len(random_points))
                observations = {label: {region: opportunity(points[:count], target, side)
                    for region, side in (('local', wide_side / 2), ('wide', wide_side))}
                    for label, points in (('cnn', cnn_points), ('random', random_points))}
                for scope in ('all', 'native_main_outside'):
                    if scope == 'native_main_outside' and not row['native_main_outside']:
                        continue
                    group = totals[scope][str(k)]
                    group['states'] += 1
                    key = str(count)
                    group['effective_count_histogram'][key] = group['effective_count_histogram'].get(key, 0) + 1
                    for label in ('cnn', 'random'):
                        for region in ('local', 'wide'):
                            inside, coverage = observations[label][region]
                            group[label][region]['centre_inside'] += inside
                            group[label][region]['coverage_sum'] += coverage
        for scope, groups in totals.items():
            for group in groups.values():
                assert group['states'] == (1752 if scope == 'all' else 239)
                for label in ('cnn', 'random'):
                    for region in ('local', 'wide'):
                        value = group[label][region]
                        value['mean_best_gt_coverage'] = value.pop('coverage_sum') / group['states']
        per_seed.append(dict(seed=seed, groups=totals))
    result = dict(status='CPU_CONTROL_COMPLETE', post_hoc=True, states=1752, nn_calls=0,
        plan_sha256=sha(args.plan), analyzer_sha256=sha(__file__), per_seed=per_seed,
        scope=plan['scope'], input_sha256=plan['input_sha256'])
    output = Path(args.output)
    assert not output.exists()
    output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    summary = {scope: {str(k): {label: {region: dict(
        mean_centre_inside=float(np.mean([r['groups'][scope][str(k)][label][region]['centre_inside'] for r in per_seed])),
        mean_best_gt_coverage=float(np.mean([r['groups'][scope][str(k)][label][region]['mean_best_gt_coverage'] for r in per_seed])))
        for region in ('local', 'wide')} for label in ('cnn', 'random')}
        for k in plan['candidate_limits']} for scope in ('all', 'native_main_outside')}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
