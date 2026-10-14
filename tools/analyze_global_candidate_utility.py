"""Recompute executed query benefits from private boxes and dataset annotations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iou(box, target):
    intersection = np.maximum(0, np.minimum(box[:2] + box[2:], target[:2] + target[2:]) - np.maximum(box[:2], target[:2])).prod()
    return float(intersection / (box[2:].prod() + target[2:].prod() - intersection))


def clip_box(box, height, width):
    x, y, w, h = [float(value) for value in box]
    right, bottom = x + w, y + h
    left, top = min(max(0, x), width - 10), min(max(0, y), height - 10)
    right, bottom = min(max(10, right), width), min(max(10, bottom), height)
    return np.asarray([left, top, max(10, right - left), max(10, bottom - top)], dtype=np.float32)


def coverage(target, geometry):
    centre = target[:2] + target[2:] / 2
    inside = bool(((centre >= geometry[:2]) & (centre < geometry[:2] + geometry[2])).all())
    overlap = np.maximum(0, np.minimum(target[:2] + target[2:], geometry[:2] + geometry[2]) - np.maximum(target[:2], geometry[:2])).prod()
    return inside, float(overlap / target[2:].prod())


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'completion', 'manifest', 'csv', 'headers', 'candidates', 'reference_content', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_bytes())
    terminal = json.loads(Path(args.completion).read_bytes())
    complete = terminal['complete']
    assert terminal['native_task']['state'] == 'COMPLETE' and terminal['native_task']['exit_code'] == 0
    assert terminal['config']['plan_sha256'] == sha(args.plan)
    assert not complete['sanity'] and complete['labelled_states'] == 1752 and complete['tracked_frames'] == 8854
    assert complete['optimizer_updates'] == complete['backward_calls'] == 0
    assert complete['reference_frozen_tensors'] == complete['fitted_frozen_tensors'] == 265
    assert sha(args.csv) == complete['raw_csv_sha256']
    for field, name in (('manifest', 'data/manifest.json'), ('candidates', plan['candidates_json']), ('reference_content', plan['content_csv'])):
        assert sha(getattr(args, field)) == plan['input_sha256'][name]
    assert sha(args.headers) == '03b13b68e9332eae630f88a291f879f8523dbbb7c8b9ed727bdc3123f337b943'
    sequences = {r['name']: r for r in json.loads(Path(args.manifest).read_bytes())['train']}
    candidates = json.loads(Path(args.candidates).read_bytes())
    headers = json.loads(Path(args.headers).read_bytes())['items']
    with Path(args.reference_content).open(newline='', encoding='utf-8') as handle:
        old_rows = list(csv.DictReader(handle))
    reference = [r for r in old_rows if r['region'] == 'GLOBAL_WIDE' and r['content'] == 'TRUE']
    main_reference = {(int(r['state']), r['region']): r for r in old_rows
                      if r['region'] in ('LOCAL_MAIN', 'LOCAL_ALTERNATE') and r['content'] == 'TRUE'}
    assert len(main_reference) == 3504
    assert len(candidates) == len(headers) == len(reference) == 1752
    with Path(args.csv).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == plan['expected_formal_auxiliary_queries'] == complete['auxiliary_query_calls'] == 31384
    states, errors = {}, dict(iou=0., main_iou=0., clip=0., coverage=0., delta=0.)
    for row in rows:
        for name in ('frame', 'state', 'modality', 'candidate_rank', 'candidate_count', 'auxiliary_tokens'):
            row[name] = int(row[name])
        for name in ('iou', 'main_iou', 'delta_from_stop', 'query_gt_coverage'):
            row[name] = float(row[name])
        for name in ('main_centre_inside', 'query_centre_inside'):
            assert row[name] in ('True', 'False')
            row[name] = row[name] == 'True'
        index = row['state']
        name, frame, width, height = headers[index]
        old, proposals = reference[index], candidates[index]
        assert (row['sequence'], row['frame']) == (name, frame) == (old['sequence'], int(old['frame']))
        assert row['modality'] == int(old['modality']) == 1 - proposals['auxiliary_modality']
        assert row['candidate_count'] == proposals['candidate_count']
        target = np.asarray(sequences[name]['boxes'][frame], dtype=np.float32)
        main_geo = np.asarray(json.loads(row['main_geometry']), dtype=np.float32)
        main_box = np.asarray(json.loads(row['main_box']), dtype=np.float32)
        query_geo = np.asarray(json.loads(row['query_geometry']), dtype=np.float32)
        raw = np.asarray(json.loads(row['raw_box']), dtype=np.float32)
        executed = np.asarray(json.loads(row['executed_box']), dtype=np.float32)
        assert all(np.isfinite(v).all() for v in (target, main_geo, main_box, query_geo, raw, executed))
        assert (target[2:] > 0).all() and (executed[2:] >= 10).all()
        errors['clip'] = max(errors['clip'], float(np.abs(clip_box(raw, height, width) - executed).max()))
        errors['iou'] = max(errors['iou'], abs(iou(executed, target) - row['iou']))
        errors['main_iou'] = max(errors['main_iou'], abs(iou(main_box, target) - row['main_iou']))
        assert row['main_iou'] == float(old['main_iou'])
        assert coverage(target, main_geo)[0] == row['main_centre_inside'] == (old['main_centre_inside'] == 'True')
        inside, covered = coverage(target, query_geo)
        assert inside == row['query_centre_inside']
        errors['coverage'] = max(errors['coverage'], abs(covered - row['query_gt_coverage']))
        errors['delta'] = max(errors['delta'], abs(row['iou'] - row['main_iou'] - row['delta_from_stop']))
        assert row['auxiliary_tokens'] == (385 if row['region'] == 'GLOBAL_WIDE' else 193)
        if row['region'].startswith('GLOBAL_'):
            point = np.asarray(proposals['points'][row['candidate_rank']], dtype=np.float32)
            side = main_geo[2] if row['region'] == 'GLOBAL_WIDE' else main_geo[2] / 2
            assert np.max(np.abs(query_geo - np.r_[point - side / 2, side])) < 1e-4
        else:
            assert row['region'] in ('LOCAL_MAIN', 'LOCAL_ALTERNATE') and row['candidate_rank'] == -1
            old_query = main_reference[index, row['region']]
            assert np.array_equal(query_geo, np.asarray(json.loads(old_query['query_geometry']), dtype=np.float32))
            assert np.array_equal(executed, np.asarray(json.loads(old_query['executed_box']), dtype=np.float32))
            assert row['iou'] == float(old_query['iou'])
        states.setdefault(index, []).append(row)
    assert len(states) == 1752 and errors['clip'] <= 1e-5
    assert all(errors[name] <= 1e-6 for name in ('iou', 'main_iou', 'coverage', 'delta'))
    for index, group in states.items():
        count = candidates[index]['candidate_count']
        assert len(group) == 2 + 2 * count
        expected = {('LOCAL_MAIN', -1), ('LOCAL_ALTERNATE', -1)} | {
            (region, rank) for rank in range(count) for region in ('GLOBAL_LOCAL', 'GLOBAL_WIDE')}
        assert {(r['region'], r['candidate_rank']) for r in group} == expected
    token_sum = sum(r['auxiliary_tokens'] for r in rows)
    assert token_sum == complete['auxiliary_token_sum']
    summary = {}
    for scope in ('all', 'main_centre_outside'):
        selected = [group for group in states.values() if scope == 'all' or not group[0]['main_centre_inside']]
        assert len(selected) == (1752 if scope == 'all' else 239)
        values = dict(states=len(selected), by_k={}, mean_main_iou=float(np.mean([g[0]['main_iou'] for g in selected])))
        for k in (1, 2, 4, 8):
            values['by_k'][str(k)] = {}
            for region in ('GLOBAL_LOCAL', 'GLOBAL_WIDE', 'ALL_ACTIONS'):
                pools = [[r for r in group if r['candidate_rank'] < k and (region == 'ALL_ACTIONS' or r['region'] == region)] for group in selected]
                best = [pool[int(np.argmax([r['iou'] for r in pool]))] for pool in pools]
                geometric = [pool[int(np.argmax([r['query_gt_coverage'] for r in pool]))] for pool in pools]
                gains = np.asarray([r['delta_from_stop'] for r in best])
                values['by_k'][str(k)][region] = dict(mean_best_query_iou=float(np.mean([r['iou'] for r in best])),
                    mean_best_signed_gain=float(gains.mean()), mean_stop_inclusive_oracle_gain=float(np.maximum(gains, 0).mean()),
                    positive=int((gains > 0).sum()), negative=int((gains < 0).sum()), zero=int((gains == 0).sum()),
                    mean_coverage_choice_iou=float(np.mean([r['iou'] for r in geometric])),
                    coverage_choice_equals_best=sum((a['region'], a['candidate_rank']) == (b['region'], b['candidate_rank']) for a, b in zip(best, geometric)))
        assert values == complete['per_scope'][scope]
        summary[scope] = values
    result = dict(status='CPU_ANALYSIS_PASS', states=1752, auxiliary_queries=len(rows), nn_calls=0,
        per_scope=summary, recomputation_max_absolute_errors=errors,
        input_sha256={n: sha(getattr(args, n)) for n in ('plan', 'completion', 'manifest', 'csv', 'headers', 'candidates', 'reference_content')},
        analyzer_sha256=sha(__file__), scope=plan['scope'])
    output = Path(args.output)
    assert not output.exists()
    output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
