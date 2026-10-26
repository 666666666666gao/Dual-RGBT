"""Join saved training decisions to their same-state counterfactual labels."""
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    run = ROOT / 'runs/global_candidate_rankstop_trajectory_formal_20261018'
    labels, steps = run / 'query_labels.csv', run / 'trajectory_steps.csv'
    assert digest(labels) == '6ddfdb92feae3294ea85211a672fe6bf1b07cdcdd5a8abc1dfdf38b908bbfa47'
    assert digest(steps) == '2f22103097fdcaeb4483841cf5456160938a2ba0baadf393a7d70317f0a14e50'
    groups = defaultdict(lambda: defaultdict(int))
    with labels.open(newline='') as label_file, steps.open(newline='') as step_file:
        label_groups = itertools.groupby(csv.DictReader(label_file), key=lambda row: int(row['step']))
        count = 0
        for decision, (step, values) in zip(csv.DictReader(step_file), label_groups):
            values = list(values)
            assert int(decision['step']) == step == count + 1
            assert all(all(row[key] == decision[key] for key in ('round', 'sequence', 'frame')) for row in values)
            sampled = {int(row['modality']): row for row in values if row['purpose'] == 'sampled'}
            assert set(sampled) == {0, 1}
            role, action = int(decision['chosen_modality']), int(decision['chosen_action'])
            q = [float(sampled[m]['main_iou']) for m in (0, 1)]
            regret = max(q) - q[role]
            sampled_gain = float(sampled[role]['delta_from_stop'])
            if action:
                actual = [row for row in values if row['purpose'] == 'selected_extra']
                if not actual:
                    actual = [sampled[role]]
                assert len(actual) == 1
                actual = actual[0]
                assert int(actual['modality']) == role and int(actual['kind']) + 1 == action
                assert actual['candidate_rank'] == decision['chosen_rank']
                gain = float(actual['delta_from_stop'])
                assert abs(float(actual['query_iou']) - q[role] - gain) < 1e-6
            else:
                assert not any(row['purpose'] == 'selected_extra' for row in values)
                gain = 0.0
            for key in ('all', 'round' + decision['round']):
                item = groups[key]
                item['updates'] += 1
                item['tir_primary'] += role == 1
                item['sum_chosen_primary_iou'] += q[role]
                item['sum_best_primary_iou'] += max(q)
                item['sum_primary_routing_regret'] += regret
                item['other_primary_better_over_0p01'] += regret > .01
                item['other_primary_better_over_0p1'] += regret > .1
                item['queries'] += bool(action)
                item['stop'] += not action
                item['query_harmful_below_minus0p01'] += gain < -.01
                item['query_helpful_over_0p01'] += gain > .01
                item['sum_executed_query_gain'] += gain
                item['sum_final_iou'] += q[role] + gain
                item['stop_with_helpful_sampled_same_role_query'] += not action and sampled_gain > .01
            count += 1
        assert next(label_groups, None) is None
    assert count == 182182
    for item in groups.values():
        item['mean_gain_per_executed_query'] = item['sum_executed_query_gain'] / item['queries']
        for name in list(item):
            if name.startswith('sum_'):
                item['mean_' + name[4:]] = item.pop(name) / item['updates']
    report = dict(status='SAVED_TRAIN_DECISION_JOIN_PASS', groups=dict(groups),
                  input_sha256=dict(labels=digest(labels), steps=digest(steps)),
                  scope='All182182 sampled update states during evolving TRAIN C882, not final fixed-weight '
                        'dev/test or all917106 tracked frames. Routing regret compares the two saved primary '
                        'predictions at identical history. Executed-query gains are reconstructed from exact '
                        'selected-extra or matching sampled action labels. STOP opportunities consider only '
                        'one sampled same-role request, not the full candidate pool or a trajectory upper bound. '
                        'Training GT is diagnostic only; no runtime GT rule or neural execution.')
    output = ROOT / 'reports/RANKSTOP_TRAIN_DECISION_DIAGNOSTIC_20261026.json'
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
