"""Recompute private content comparisons; export only aggregate evidence."""
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
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--completion', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    terminal = json.loads(Path(args.completion).read_text(encoding='utf-8'))
    assert terminal['native_task']['state'] == 'COMPLETE' and terminal['native_task']['exit_code'] == 0
    assert sha(args.csv) == terminal['private_csv_sha256']
    assert sha(args.plan) == terminal['config']['plan_sha256']
    training = json.loads((Path(args.plan).parent.parent / plan['training_plan']).read_text(encoding='utf-8'))
    assert sha(args.manifest) == training['manifest_sha256']
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    sequences = {row['name']: row for row in manifest['train']}
    expected = [(name, frame) for name in plan['sequence_names']
                for frame in range(1, len(sequences[name]['boxes']))
                if frame % plan['label_stride'] == 0 and frame in sequences[name]['valid_frames']]
    assert len(expected) == plan['expected_labelled_states'] == 1752
    with Path(args.csv).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == plan['expected_formal_comparisons'] == 31536
    for row in rows:
        for field in ('frame', 'state', 'modality', 'auxiliary_tokens'):
            row[field] = int(row[field])
        for field in ('iou', 'main_iou', 'delta_from_stop', 'query_gt_coverage'):
            row[field] = float(row[field])
        for field in ('main_centre_inside', 'query_centre_inside'):
            assert row[field] in ('True', 'False')
            row[field] = row[field] == 'True'
        assert np.isfinite([row['iou'], row['main_iou'], row['delta_from_stop'], row['query_gt_coverage']]).all()
        assert row['delta_from_stop'] == row['iou'] - row['main_iou']
    prediction = np.asarray([json.loads(row['executed_box']) for row in rows], dtype=np.float32)
    geometry = np.asarray([json.loads(row['query_geometry']) for row in rows], dtype=np.float32)
    target = np.asarray([sequences[row['sequence']]['boxes'][row['frame']] for row in rows], dtype=np.float32)
    assert prediction.shape == target.shape == (31536, 4) and geometry.shape == (31536, 3)
    assert (target[:, 2:] > 0).all() and (prediction[:, 2:] > 0).all() and (geometry[:, 2] > 0).all()
    intersection = np.maximum(0, np.minimum(prediction[:, :2] + prediction[:, 2:], target[:, :2] + target[:, 2:]) -
                              np.maximum(prediction[:, :2], target[:, :2])).prod(1)
    computed_iou = intersection / (prediction[:, 2:].prod(1) + target[:, 2:].prod(1) - intersection)
    saved_iou = np.asarray([row['iou'] for row in rows])
    iou_error = np.abs(computed_iou.astype(np.float64) - saved_iou)
    assert iou_error.max() <= 1e-6
    centre = target[:, :2] + target[:, 2:] / 2
    inside = ((centre >= geometry[:, :2]) & (centre < geometry[:, :2] + geometry[:, 2:3])).all(1)
    assert np.array_equal(inside, np.asarray([row['query_centre_inside'] for row in rows]))
    overlap = np.maximum(0, np.minimum(target[:, :2] + target[:, 2:], geometry[:, :2] + geometry[:, 2:3]) -
                         np.maximum(target[:, :2], geometry[:, :2])).prod(1)
    computed_coverage = overlap / target[:, 2:].prod(1)
    coverage_error = np.abs(computed_coverage.astype(np.float64) - np.asarray([row['query_gt_coverage'] for row in rows]))
    assert coverage_error.max() <= 1e-6
    states = []
    keys = {(region, content) for region in plan['regions'] for content in plan['contents']}
    for index, pair in enumerate(expected):
        group = rows[index * 18:(index + 1) * 18]
        assert all(row['state'] == index and (row['sequence'], row['frame']) == pair for row in group)
        assert len({row['modality'] for row in group}) == 1 and group[0]['modality'] in (0, 1)
        assert len({row['main_iou'] for row in group}) == len({row['main_centre_inside'] for row in group}) == 1
        matrix = {(row['region'], row['content']): row for row in group}
        assert len(matrix) == 18 and set(matrix) == keys
        for region in plan['regions']:
            variants = [matrix[region, content] for content in plan['contents']]
            assert len({row['query_geometry'] for row in variants}) == 1
            assert len({row['query_centre_inside'] for row in variants}) == 1
            assert len({row['query_gt_coverage'] for row in variants}) == 1
            assert all(row['auxiliary_tokens'] == (385 if region.endswith('WIDE') else 193) for row in variants)
        states.append(matrix)
    native = terminal['complete']['per_region']
    aggregates, coverage = {}, {}
    for region in plan['regions']:
        contents, paired = {}, {}
        for content in plan['contents']:
            values = [state[region, content] for state in states]
            contents[content] = dict(rows=len(values), mean_iou=float(np.mean([row['iou'] for row in values])),
                mean_delta_from_stop=float(np.mean([row['delta_from_stop'] for row in values])))
            assert contents[content]['rows'] == native[region]['contents'][content]['rows']
            for field in ('mean_iou', 'mean_delta_from_stop'):
                assert abs(contents[content][field] - native[region]['contents'][content][field]) <= 1e-12
        for control in ('ZERO_AUX', 'SHIFT_AUX'):
            for scope in ('all', 'main_centre_outside'):
                selected = [state for state in states if scope == 'all' or not state[region, 'TRUE']['main_centre_inside']]
                delta = np.asarray([state[region, 'TRUE']['iou'] - state[region, control]['iou'] for state in selected])
                value = dict(rows=len(delta), sum_delta=float(delta.sum()), positive=int((delta > 0).sum()),
                             negative=int((delta < 0).sum()), zero=int((delta == 0).sum()))
                key = control + ':' + scope
                for field, number in value.items():
                    assert abs(number - native[region]['true_minus_control'][key][field]) <= 1e-12
                paired[key] = value
        true = [state[region, 'TRUE'] for state in states]
        outside = [row for row in true if not row['main_centre_inside']]
        coverage[region] = dict(states=len(true), query_centre_inside=sum(row['query_centre_inside'] for row in true),
            mean_query_gt_coverage=float(np.mean([row['query_gt_coverage'] for row in true])),
            native_main_centre_outside=len(outside), outside_query_centre_inside=sum(row['query_centre_inside'] for row in outside),
            outside_sum_gt_coverage=sum(row['query_gt_coverage'] for row in outside))
        aggregates[region] = dict(contents=contents, true_minus_control=paired)
    report = dict(status='CPU_ANALYSIS_PASS', sequences=20, labelled_states=1752, comparisons=31536,
        input_sha256=dict(manifest=sha(args.manifest), private_csv=sha(args.csv), plan=sha(args.plan),
                          completion=sha(args.completion), analyzer=sha(__file__)),
        recomputed_iou_max_absolute_error=float(iou_error.max()),
        recomputed_query_coverage_max_absolute_error=float(coverage_error.max()),
        query_centres_independently_checked=True, same_geometry_tokens_primary_checks=True,
        native_aggregates_reproduced=True, per_region=aggregates, query_coverage=coverage,
        selected_modality_counts=[sum(state['LOCAL_MAIN', 'TRUE']['modality'] == modality for state in states) for modality in range(2)],
        scope='Within-state TRUE/constant/shifted auxiliary search content comparisons on the B primary-only train20 reference. Correct coordinates/templates/previews remain available. Not tracking metrics or latency.',
        limitations=['Main executed boxes and main search geometries are absent from CSV: main IoUs/centre-outside flags remain native-derived, not independently reconstructed.',
            'Query coordinates for deployable proposals are checked for cross-content identity and GT coverage, not independently regenerated from model features.',
            'GT-centred query coordinates supply a correct-location prior; content differences do not prove deployable proposal success.',
            'SHIFT_AUX is cyclic spatial displacement, not a separate distractor identity or natural sensor corruption.',
            'One seed and one visited trajectory distribution; no statistical stability, full PR/NPR/SR or efficiency claim.'])
    output = Path(args.output)
    assert not output.exists()
    output.write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps({key: value for key, value in report.items() if key not in ('per_region','query_coverage')}, indent=2))


if __name__ == '__main__':
    main()
