"""Read native-complete training labels; no fitting or model changes."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_label(group, main, query, kind, stopped):
    delta = query - main
    cost = 1.0 if kind == 3 else 193.0 / 385
    group['labels'] += 1
    group['helpful_gt_0p01'] += delta > .01
    group['harmful_lt_minus0p01'] += delta < -.01
    group['near_zero_gain'] += abs(delta) <= .01
    group['main_lost'] += main < .1
    group['recovers_lost_to_0p5'] += main < .1 and query >= .5
    group['destroys_good_to_0p1'] += main >= .5 and query < .1
    group['stopped'] += stopped
    group['stopped_recovery'] += stopped and main < .1 and query >= .5
    for penalty in (0, .05, .1, .15):
        group['positive_utility_at_' + str(penalty)] += delta - penalty * cost > 1e-5
    group['gain_sum'] += delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    training_path = ROOT / plan['training_plan']
    training = json.loads(training_path.read_text())
    directory = ROOT / 'runs' / training['runs']['formal']
    native_path = ROOT / 'runs/pipeline_tasks' / training['tasks']['formal'] / 'status.json'
    assert sha(native_path) == plan['input_sha256'][native_path.relative_to(ROOT).as_posix()]
    native = json.loads(native_path.read_text())
    complete_path, config_path = directory / 'COMPLETE.json', directory / 'config.json'
    complete, config = json.loads(complete_path.read_text()), json.loads(config_path.read_text())
    assert native['state'] == complete['status'] == 'COMPLETE' and native['exit_code'] == 0
    assert not complete['sanity'] and not config['sanity']
    assert config['run'] == plan['candidate_id'] == training['runs']['formal']
    assert sha(config_path) == plan['training_config_sha256']
    assert sha(training_path) == config['plan_sha256']
    assert complete['checkpoint_sha256'] == plan['checkpoint_sha256']
    assert complete['selected_execution_bit_equal'] and complete['teacher_preserves_tracker_state']
    steps_path, labels_path = directory / 'trajectory_steps.csv', directory / 'query_labels.csv'
    inputs = [plan_path, training_path, native_path, complete_path, config_path, steps_path, labels_path]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    with steps_path.open(newline='') as handle:
        steps = {int(row['step']): row for row in csv.DictReader(handle)}
    assert sorted(steps) == list(range(1, complete['updates'] + 1))
    assert set(row['sequence'] for row in steps.values()) == set(training['sequence_names'])
    groups, families, seen = defaultdict(Counter), Counter(), set()
    chosen_samples = extras = total = 0
    with labels_path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            step, role, kind = int(row['step']), int(row['modality']), int(row['kind'])
            source = steps[step]
            assert all(row[k] == source[k] for k in ('round', 'sequence', 'frame'))
            assert role in (0, 1) and kind in range(4)
            main_iou, query_iou, delta = (float(row[k]) for k in ('main_iou', 'query_iou', 'delta_from_stop'))
            assert all(math.isfinite(x) for x in (main_iou, query_iou, delta))
            assert 0 <= main_iou <= 1 and 0 <= query_iou <= 1
            assert abs(query_iou - main_iou - delta) < 1e-6
            key = (step, role, row['purpose'])
            assert key not in seen
            seen.add(key)
            chosen = role == int(source['chosen_modality'])
            stopped = source['chosen_action'] == '0'
            total += 1
            if row['purpose'] == 'sampled':
                assert kind == int(source['sampled_kind'])
                families[str(role) + ':' + str(kind)] += 1
                for name in ('all_sampled', 'kind_' + str(kind), 'round_' + row['round']):
                    count_label(groups[name], main_iou, query_iou, kind, stopped)
                if chosen:
                    chosen_samples += 1
                    for name in ('chosen_modality_sampled', 'chosen_round_' + row['round'], 'chosen_kind_' + str(kind)):
                        count_label(groups[name], main_iou, query_iou, kind, stopped)
            else:
                assert row['purpose'] == 'selected_extra' and source['extra_selected_query'] == 'True'
                assert chosen and kind + 1 == int(source['chosen_action'])
                extras += 1
                count_label(groups['selected_extra'], main_iou, query_iou, kind, False)
    assert total == complete['labelled_queries']
    assert chosen_samples == complete['updates'] and sum(families.values()) == complete['sampled_query_labels']
    assert dict(families) == complete['sampled_family_counts']
    assert extras == complete['extra_selected_query_labels']
    assert all(sha(ROOT / name) == value for name, value in hashes.items())
    for group in groups.values():
        group['mean_gain'] = group.pop('gain_sum') / group['labels']
    result = dict(status='VERIFIED_TRAIN_LABEL_DIAGNOSTIC', candidate_id=plan['candidate_id'],
        checkpoint_sha256=plan['checkpoint_sha256'], updates=chosen_samples, labelled_queries=total,
        sampled_family_counts=dict(families), groups=dict(groups), input_sha256=hashes,
        script_sha256=sha(Path(__file__)),
        scope='Historical changing-model training states, two rounds, one sampled query per modality per labelled step. Not final-checkpoint dev recall, exhaustive candidate coverage, or future trajectory benefit. STOP comparisons use chosen_modality_sampled only. Labels newly hashed for this audit; native metadata and counts matched, not a prior labels hash attestation.')
    output = ROOT / args.output
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'updates', 'labelled_queries', 'groups')}, indent=2))


if __name__ == '__main__':
    main()
