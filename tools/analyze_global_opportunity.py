"""Aggregate private train-state diagnostics without exporting boxes or GT."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

DEPLOYABLE = ('STOP', 'LOCAL_MAIN', 'LOCAL_ALTERNATE', 'GLOBAL_LOCAL', 'GLOBAL_WIDE')
CONTROLS = ('DENSE_SAME_SEARCH', 'RANDOM_LOCAL', 'GT_CENTER_LOCAL', 'GT_CENTER_WIDE')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(rows):
    iou = np.asarray([row['iou'] for row in rows], dtype=np.float64)
    gain = np.asarray([row['delta'] for row in rows], dtype=np.float64)
    return dict(rows=len(rows), mean_iou=float(iou.mean()), mean_delta=float(gain.mean()),
                positive=int((gain > 0).sum()), negative=int((gain < 0).sum()), zero=int((gain == 0).sum()),
                centre_inside=int(sum(row['centre_inside'] for row in rows)),
                mean_gt_coverage=float(np.mean([row['gt_coverage'] for row in rows])))


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
    expected_states = [(name, frame) for name in plan['sequence_names']
                       for frame in range(1, len(sequences[name]['boxes']))
                       if frame % plan['label_stride'] == 0 and frame in sequences[name]['valid_frames']]
    assert len(expected_states) == plan['expected_labelled_states'] == 1752
    with Path(args.csv).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == plan['expected_formal_comparisons'] == 24528
    for row in rows:
        for field in ('frame', 'state', 'reference_modality', 'modality', 'auxiliary_tokens'):
            row[field] = int(row[field])
        for field in ('iou', 'main_iou', 'delta', 'gt_coverage'):
            row[field] = float(row[field])
        assert row['centre_inside'] in ('True', 'False')
        row['centre_inside'] = row['centre_inside'] == 'True'
        assert np.isfinite([row['iou'], row['main_iou'], row['delta'], row['gt_coverage']]).all()
        assert row['delta'] == row['iou'] - row['main_iou']
    prediction = np.asarray([json.loads(row['executed_box']) for row in rows], dtype=np.float32)
    target = np.asarray([sequences[row['sequence']]['boxes'][row['frame']] for row in rows], dtype=np.float32)
    assert (target[:, 2:] > 0).all()
    intersection = np.maximum(0, np.minimum(prediction[:, :2] + prediction[:, 2:], target[:, :2] + target[:, 2:]) -
                              np.maximum(prediction[:, :2], target[:, :2])).prod(1)
    recomputed = intersection / (prediction[:, 2:].prod(1) + target[:, 2:].prod(1) - intersection)
    saved = np.asarray([row['iou'] for row in rows])
    error = np.abs(recomputed.astype(np.float64) - saved)
    assert error.max() <= 1e-6
    states, best_primary, best_query = [], [], []
    for index, expected in enumerate(expected_states):
        group = rows[index * 14:(index + 1) * 14]
        assert all(row['state'] == index and (row['sequence'], row['frame']) == expected for row in group)
        reference = group[0]['reference_modality']
        assert reference in (0, 1) and all(row['reference_modality'] == reference for row in group)
        matrix = {(row['modality'], row['action']): row for row in group}
        assert len(matrix) == 14
        assert set(matrix) == {(modality, action) for modality in range(2) for action in DEPLOYABLE} | {
            (reference, action) for action in CONTROLS}
        for modality in range(2):
            stop = matrix[modality, 'STOP']
            assert stop['iou'] == stop['main_iou'] and stop['delta'] == 0
            assert all(row['main_iou'] == stop['iou'] for row in group if row['modality'] == modality)
        chosen = {action: matrix[reference, action] for action in DEPLOYABLE + CONTROLS}
        primary = chosen['STOP']
        assert matrix[1 - reference, 'STOP']['gt_coverage'] == primary['gt_coverage']
        assert matrix[1 - reference, 'STOP']['centre_inside'] == primary['centre_inside']
        best_primary.append(max(matrix[modality, 'STOP']['iou'] for modality in range(2)))
        best_query.append(max(chosen[action]['iou'] for action in DEPLOYABLE))
        states.append(chosen)
    native = terminal['complete']['per_action']
    for action in DEPLOYABLE + CONTROLS:
        computed = statistics([row for row in rows if row['action'] == action])
        for key, value in computed.items():
            assert abs(value - native[action][key]) <= 1e-12, (action, key)
    masks = dict(all=lambda state: True,
                 main_centre_outside=lambda state: not state['STOP']['centre_inside'],
                 main_gt_disjoint=lambda state: state['STOP']['gt_coverage'] == 0.,
                 primary_iou_zero=lambda state: state['STOP']['iou'] == 0.)
    groups = {}
    for name, predicate in masks.items():
        subset = [state for state in states if predicate(state)]
        groups[name] = dict(states=len(subset), per_action={
            action: statistics([state[action] for state in subset]) for action in DEPLOYABLE + CONTROLS
        } if subset else {})
    pairs = {}
    for action, control in (('LOCAL_MAIN', 'RANDOM_LOCAL'), ('LOCAL_ALTERNATE', 'RANDOM_LOCAL'),
                            ('GLOBAL_LOCAL', 'RANDOM_LOCAL'), ('GLOBAL_WIDE', 'DENSE_SAME_SEARCH')):
        assert all(state[action]['auxiliary_tokens'] == state[control]['auxiliary_tokens'] for state in states)
        delta = np.asarray([state[action]['iou'] - state[control]['iou'] for state in states])
        pairs[action + '_minus_' + control] = dict(states=len(states), mean_iou_difference=float(delta.mean()),
            positive=int((delta > 0).sum()), negative=int((delta < 0).sum()), ties=int((delta == 0).sum()))
    primary_iou = np.asarray([state['STOP']['iou'] for state in states])
    report = dict(status='CPU_ANALYSIS_PASS', sequences=20, labelled_states=len(states), comparisons=len(rows),
        input_sha256=dict(private_csv=sha(args.csv), manifest=sha(args.manifest), plan=sha(args.plan),
                          completion=sha(args.completion), analyzer=sha(__file__)),
        numerical_checks=dict(max_recomputed_float32_iou_abs_error=float(error.max()),
                              exact_recomputed_iou_rows=int((error == 0).sum()),
                              python_delta_subtractions_exact=True, native_aggregate_fields_recomputed=True),
        groups=groups, same_token_budget_pairs=pairs,
        conditional_gt_selection=dict(selected_primary_mean_iou=float(primary_iou.mean()),
            best_observed_primary_mean_iou=float(np.mean(best_primary)),
            best_observed_selected_role_action_mean_iou=float(np.mean(best_query)),
            best_selected_role_action_mean_gain=float((np.asarray(best_query) - primary_iou).mean())),
        scope='Single seed2027, train20, primary-only visited states; state-weighted IoU, not toolkit PR/NPR/SR. '
              'GT controls and GT-selected observed actions are diagnostic only, not deployable, best-region or trajectory upper bounds. '
              'Coverage fields are native executed-query measurements, not independently reconstructed by this CPU analysis. '
              'One random query per state. Token comparisons and diagnostic elapsed time do not establish real inference latency.')
    output = Path(args.output)
    assert not output.exists()
    output.write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(dict(status=report['status'], groups={name: value['states'] for name, value in groups.items()},
                         numerical_checks=report['numerical_checks'],
                         conditional_gt_selection=report['conditional_gt_selection']), indent=2))


if __name__ == '__main__':
    main()
