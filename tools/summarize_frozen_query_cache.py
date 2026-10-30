"""Verify a completed feature cache and summarize same-state query outcomes."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = ROOT / 'runs' / args.run
    config = json.loads((run / 'config.json').read_text())
    plan = config['plan']
    native_path = ROOT / 'runs/pipeline_tasks' / args.task / 'status.json'
    native = json.loads(native_path.read_text())
    receipt_path = run / ('SMOKE.json' if config['smoke'] else 'COMPLETE.json')
    receipt = json.loads(receipt_path.read_text())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert native['command'] == config['native_command'] and native['task'] == args.task
    assert receipt['status'] == ('SMOKE_PASS' if config['smoke'] else 'CACHE_COMPLETE')
    assert receipt['optimizer_updates'] == receipt['backward_calls'] == 0
    assert receipt['teacher_preserves_tracker_state'] and config['split'] == 'train'
    assert receipt['checkpoint_sha256'] == config['checkpoint_sha256'] == plan['checkpoint_sha256']
    assert receipt['plan_sha256'] == config['plan_sha256']
    assert all(sha(run / name) == digest for name, digest in receipt['artifacts_sha256'].items())
    summaries, totals = {}, []
    for name in config['selected_sequences']:
        features = np.load(run / (name + '_features.npy'), allow_pickle=False)
        with (run / (name + '_labels.csv')).open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert features.shape == (len(rows), 955) and features.dtype == np.float32
        assert np.isfinite(features).all()
        states = defaultdict(list)
        for row in rows:
            assert row['sequence'] == name
            assert row['partition'] == ('fit' if name in plan['fit_sequences'] else 'holdout')
            for key in ('primary_iou', 'query_iou', 'gain', 'predicted_gain', 'sampling_probability'):
                row[key] = float(row[key])
                assert np.isfinite(row[key])
            assert 0 <= row['primary_iou'] <= 1 and 0 <= row['query_iou'] <= 1
            assert 0 < row['sampling_probability'] <= 1
            assert abs(row['query_iou'] - row['primary_iou'] - row['gain']) < 1e-6
            states[int(row['frame'])].append(row)
        result = dict(partition=rows[0]['partition'], states=len(states), labels=len(rows),
            lost_states=0, sampled_recoverable_states=0, recoverable_but_stopped=0,
            deployed_queries=0, deployed_recoveries=0, deployed_destroys_good=0)
        for frame, group in states.items():
            assert frame > 0 and frame % plan['label_stride'] == 0
            assert len({r['primary_iou'] for r in group}) == len({r['deployed_action'] for r in group}) == 1
            assert len({r['request_index'] for r in group}) == len(group)
            sampled = [r for r in group if r['purpose'] == 'sampled']
            assert sorted(int(r['kind']) for r in sampled) == [0, 1, 2, 3]
            assert all(r['purpose'] in ('sampled', 'selected_extra') for r in group)
            main_iou, action = group[0]['primary_iou'], int(group[0]['deployed_action'])
            selected = [r for r in group if r['selected'] == '1']
            assert len(selected) == int(action != 0)
            utilities = [r['predicted_gain'] - plan['cost_weight'] * (1 if int(r['kind']) == 3 else 193/385) for r in group]
            if selected:
                chosen = selected[0]
                assert int(chosen['kind']) + 1 == action
                chosen_utility = chosen['predicted_gain'] - plan['cost_weight'] * (1 if action == 4 else 193/385)
                assert chosen_utility >= -1e-6 and chosen_utility >= max(utilities)-1e-6
                result['deployed_queries'] += 1
                result['deployed_recoveries'] += int(main_iou < .1 and chosen['query_iou'] >= .5)
                result['deployed_destroys_good'] += int(main_iou >= .5 and chosen['query_iou'] < .1)
            else:
                assert max(utilities) <= 1e-6
            recoverable = main_iou < .1 and any(r['query_iou'] >= .5 for r in sampled)
            result['lost_states'] += int(main_iou < .1)
            result['sampled_recoverable_states'] += int(recoverable)
            result['recoverable_but_stopped'] += int(recoverable and action == 0)
        summaries[name] = result
        totals.extend(rows)
    assert len(totals) == receipt['labels'] and len(summaries) == receipt['sequences']
    partitions = {}
    for partition in ('fit', 'holdout'):
        group = [s for s in summaries.values() if s['partition'] == partition]
        partitions[partition] = {k: sum(s[k] for s in group) for k in result if k != 'partition'}
        partitions[partition]['sequences'] = len(group)
    output = ROOT / args.output
    assert not output.exists()
    report = dict(status='VERIFIED_CACHE_DIAGNOSTIC', smoke=config['smoke'], run=args.run,
        checkpoint_sha256=receipt['checkpoint_sha256'], native_sha256=sha(native_path),
        receipt_sha256=sha(receipt_path), source_sha256=sha(Path(__file__)),
        partitions=partitions, sequences=summaries,
        scope='Frozen-parent train-state sampled query availability and deployed decisions. Not exhaustive candidates, dev accuracy, calibration fit or future trajectory improvement.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(partitions, indent=2))


if __name__ == '__main__':
    main()
