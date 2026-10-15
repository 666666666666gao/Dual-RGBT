"""CPU-only descriptive comparison of two completed, matched-cost dev traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'scripts'))
from analyze_full_dev_coverage import analyse_sequence
from analyze_state_write_trajectories import future_summary

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--events', required=True)
parser.add_argument('--inventory', required=True)
args = parser.parse_args()
output, events_path, inventory_path = [Path(value) for value in (args.output, args.events, args.inventory)]
assert not output.exists() and not events_path.exists() and not inventory_path.exists()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
reference_path = root / 'reports/C20_C882_DEV_MATCHED_COST_COMPARISON.json'
assert sha(reference_path) == 'e4b788f4a3699a1041e7fe8e7c7123cb670d60cb64a47175e068e7fa4f755a21'
reference = json.loads(reference_path.read_bytes())
inputs = {str(reference_path): sha(reference_path)}
for name in ('scripts/analyze_full_dev_coverage.py', 'scripts/analyze_state_write_trajectories.py', 'dual_rgbt/tracker.py'):
    inputs[str(root / name)] = sha(root / name)
inputs[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
arms, event_rows, targets = [], [], {}
for identity in (row for row in reference['raw_metric_table'] if row['cost_weight'] == .05):
    version = identity['version']
    run = (root / identity['config_file']).parent
    config = json.loads((run / 'config.json').read_bytes())
    complete = json.loads((run / 'COMPLETE.json').read_bytes())
    for field in ('config_file', 'complete_file'):
        name = identity[field]
        assert sha(root / name) == reference['input_hashes'][name]
    assert config['split'] == 'dev' and config['smoke_frames'] == 0 and config['shards'] == 1 and config['shard'] == 0
    assert config['cost_weight'] == .05 and config['checkpoint_sha256'] == identity['checkpoint_sha256']
    names = config['sequences']
    assert len(names) == len(set(names)) == complete['sequences'] == config['split_sequences'] == 97
    assert complete['frames'] == config['split_frames'] == 54646 and complete['smoke_only'] is False
    for directory, suffix in (('predictions', '*.txt'), ('groundtruth', '*.txt'), ('traces', '*.csv')):
        assert {p.stem for p in (run / directory).glob(suffix)} == set(names)
    for path in run.rglob('*'):
        if path.is_file():
            inputs[str(path)] = sha(path)
    totals = {key: 0 for key in ('frames', 'tracked_frames', 'valid_frames', 'invalid_frames',
        'inferred_write_events_all_gt', 'inferred_write_events_valid_gt', 'inferred_modality_writes_valid_gt',
        'low_iou_frames', 'low_iou_write_events', 'bad_write_auxiliary_selected', 'bad_write_after20_low_frames',
        'low_iou_episodes', 'low_iou_episodes_with_write', 'low_iou_episodes_at_least20')}
    previous = {key: 0 for key in ('low', 'intermediate', 'good', 'invalid_or_unobserved')}
    streak_bins = {key: 0 for key in ('zero', 'one_to_four', 'five_to_nineteen', 'twenty_or_more')}
    actions = np.zeros(5, dtype=np.int64)
    bad_confidences, token_sum = [], 0
    windows = {str(horizon): [] for horizon in (5, 20, 50)}
    per_sequence = []
    for name in names:
        gt_sha = sha(run / 'groundtruth' / (name + '.txt'))
        if version == 'C20':
            targets[name] = gt_sha
        else:
            assert targets[name] == gt_sha
        base, values = analyse_sequence(run, name)
        iou, valid, confidence = [values[key] for key in ('raw_iou', 'valid', 'confidence')]
        assert ((confidence >= 0) & (confidence <= 1)).all()
        frame = np.arange(1, len(iou) + 1)
        update = (frame % 5 == 0) & (confidence >= .65)
        low, good = valid & (iou < .1), valid & (iou >= .5)
        bad_write = low & update
        indices = np.flatnonzero(bad_write)
        prior = np.zeros(len(iou), dtype=np.int64)
        streak = 0
        for index in range(len(iou)):
            prior[index] = streak
            streak = streak + 1 if low[index] else 0
        for index in indices:
            label = ('invalid_or_unobserved' if index == 0 or not valid[index - 1]
                     else 'low' if low[index - 1] else 'good' if good[index - 1] else 'intermediate')
            previous[label] += 1
            length = int(prior[index])
            streak_bins['zero' if length == 0 else 'one_to_four' if length < 5 else 'five_to_nineteen' if length < 20 else 'twenty_or_more'] += 1
            event_rows.append(dict(version=version, sequence=name, frame=int(frame[index]),
                action=int(values['action'][index]), modality=int(values['modality'][index]),
                confidence=float(confidence[index]), raw_iou=float(iou[index]), preceding_consecutive_low_frames=length))
        starts = np.flatnonzero(low & ~np.concatenate(([False], low[:-1])))
        ends = np.flatnonzero(low & ~np.concatenate((low[1:], [False])))
        assert len(starts) == len(ends)
        row = dict(sequence=name, frames=base['frames'], tracked_frames=len(iou), valid_frames=int(valid.sum()),
            invalid_frames=int((~valid).sum()), inferred_write_events_all_gt=int(update.sum()),
            inferred_write_events_valid_gt=int((update & valid).sum()),
            inferred_modality_writes_valid_gt=int(((1 + (values['action'] > 0)) * (update & valid)).sum()),
            low_iou_frames=int(low.sum()), low_iou_write_events=int(bad_write.sum()),
            bad_write_auxiliary_selected=int((bad_write & (values['action'] > 0)).sum()),
            bad_write_after20_low_frames=int((prior[indices] >= 20).sum()), low_iou_episodes=len(starts),
            low_iou_episodes_with_write=sum(bool(update[start:end + 1].any()) for start, end in zip(starts, ends)),
            low_iou_episodes_at_least20=int(sum(end - start + 1 >= 20 for start, end in zip(starts, ends))))
        per_sequence.append(row)
        for key in totals:
            totals[key] += int(row[key])
        for action in range(5):
            actions[action] += int((bad_write & (values['action'] == action)).sum())
        bad_confidences.extend(confidence[bad_write].tolist())
        token_sum += int(values['heavy_tokens'].sum())
        for horizon in (5, 20, 50):
            windows[str(horizon)].append(future_summary(indices, values, horizon))
    assert totals['frames'] == 54646 and totals['tracked_frames'] == 54549 and totals['valid_frames'] == 54289
    assert sum(previous.values()) == sum(streak_bins.values()) == sum(actions) == totals['low_iou_write_events']
    pooled_windows = {}
    for horizon, rows in windows.items():
        pooled = {key: sum(row[key] for row in rows) for key in rows[0] if key != 'mean_future_iou'}
        count = pooled['complete_valid_windows']
        pooled['mean_future_iou'] = sum(row['mean_future_iou'] * row['complete_valid_windows'] for row in rows if row['complete_valid_windows']) / count if count else None
        pooled_windows[horizon] = pooled
    arms.append(dict(version=version, checkpoint_sha256=config['checkpoint_sha256'], cost_weight=.05,
        run=str(run.resolve()), counts=totals, low_iou_write_fraction_percent=100 * totals['low_iou_write_events'] / totals['inferred_write_events_valid_gt'],
        mean_bad_write_confidence=float(np.mean(bad_confidences)),
        mean_recorded_heavy_tokens=token_sum / totals['tracked_frames'],
        bad_write_predecessor=previous, bad_write_prior_low_streak=streak_bins,
        bad_write_action_counts=actions.tolist(), bad_write_future_windows=pooled_windows, per_sequence=per_sequence))
assert [arm['version'] for arm in arms] == ['C20', 'C882']
assert arms[0]['counts']['inferred_write_events_valid_gt'] == 6430 and arms[0]['counts']['low_iou_write_events'] == 1053
assert len(targets) == 97
with events_path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(event_rows[0]))
    writer.writeheader()
    writer.writerows(event_rows)
assert all(sha(Path(path)) == digest for path, digest in inputs.items())
inventory_path.write_text(json.dumps(inputs, indent=2) + '\n', encoding='utf-8')
report = dict(scope='Original E0 C20 versus E0 C882, cost0.05, same completed97 dev trajectories. Descriptive post-hoc associations only; not selected test-cost0, E1, full test or a causal intervention.',
    evaluation_type='real_gt', neural_execution=False, ssh_calls=0, gpu_queries=0, task_launches=0,
    protocol=dict(sequences=97, total_frames_per_arm=54646, initialization_excluded=True,
        raw_geometry='Continuous float32 box intersection/union, float64 arrays and reductions; not legacy rounded toolkit PR/NPR/SR.',
        valid_gt='positive width and height', inferred_write_rule='noninitial frame%5==0 and confidence>=0.65',
        low_iou_threshold=.1, predecessor_good_iou_threshold=.5, future_horizons=[5, 20, 50],
        invalid_gt_resets_streak=True, gt_files_byte_equal=True,
        frame_weighted_diagnostics=True, identical_cost_coefficient_is_not_identical_executed_compute=True),
    arms=arms, delta_c882_minus_c20=dict(counts={key: arms[1]['counts'][key] - arms[0]['counts'][key] for key in arms[0]['counts']},
        low_iou_write_fraction_percentage_points=arms[1]['low_iou_write_fraction_percent'] - arms[0]['low_iou_write_fraction_percent']),
    input_inventory_sha256=sha(inventory_path), input_inventory_count=len(inputs), private_event_records=len(event_rows),
    private_events_sha256=sha(events_path), producer_sha256=sha(Path(__file__).resolve()),
    limitations=[
        'One trained checkpoint and one archived dev run per arm. Training sequence scope and update count both differ.',
        'Observed states and selected actions differ between trajectories. They are not paired identical-state interventions or matched real compute.',
        'Write events are inferred from the frozen tracker rule, not explicit write logs; an event writes one modality on STOP and two on a query.',
        'Raw diagnostic validity excludes nonpositive width/height and initialization, unlike the unchanged legacy official scoring protocol.',
        'A bad selected-query output is not evidence of negative auxiliary gain: main-only and rejected-query counterfactuals were not recorded.',
        'Predecessors, streaks and overlapping future windows describe temporal association, not independent error probabilities or causal template harm.',
        'Future reaches0.5 is one crossing within a valid window, not stable recovery. Invalid or sequence-short windows are excluded explicitly.',
        'No threshold tuning, test-set adaptation, new NN execution, selection change or achieved target claim.'])
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps(dict(arms=[{key: value for key, value in arm.items() if key not in ('per_sequence', 'bad_write_future_windows')} for arm in arms],
    delta=report['delta_c882_minus_c20'], input_inventory_count=len(inputs), output_sha256=sha(output)), indent=2))
