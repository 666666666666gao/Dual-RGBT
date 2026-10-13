"""Analyze completed frozen train-only action diagnostics, without NN execution."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


METRICS = ('clipped', 'uncensored', 'executed')
ACTION_COST = (0., 193. / 385., 193. / 385., 193. / 385., 1.)


def quantiles(values):
    if not values:
        return None
    ordered = sorted(values)
    result = []
    for fraction in (0., .01, .5, .99, 1.):
        position = (len(ordered) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        result.append(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))
    return result


def pair_summary(rows):
    comparisons = {}
    for before, after in (('clipped', 'uncensored'), ('uncensored', 'executed'), ('clipped', 'executed')):
        differences = [row[after + '_gain'] - row[before + '_gain'] for row in rows]
        flips = [min(abs(row[before + '_gain']), abs(row[after + '_gain'])) for row in rows
                 if row[before + '_gain'] * row[after + '_gain'] < 0]
        comparisons[before + '_to_' + after] = dict(
            raw_sign_flip_pairs=len(flips), flipped_min_abs_gain_quantiles=quantiles(flips),
            mean_signed_gain_difference=statistics.mean(differences),
            mean_absolute_gain_difference=statistics.mean(map(abs, differences)))
    return dict(action_pairs=len(rows), comparisons=comparisons,
                positive_gain_pairs={metric: sum(row[metric + '_gain'] > 0 for row in rows) for metric in METRICS},
                positive_full_image_iou_gain_pairs=sum(row['full_iou_gain'] > 0 for row in rows),
                predicted_gain_mae={metric: statistics.mean(abs(row['predicted_gain'] - row[metric + '_gain'])
                                                            for row in rows) for metric in METRICS})


def state_summary(states, cost):
    mismatch, stop_to_query, query_to_stop = 0, 0, 0
    clipped_regrets, policy_regrets, chosen_gains, iou_gains = [], [], [], []
    chosen_counts, clipped_counts, executed_counts = [0] * 5, [0] * 5, [0] * 5
    primary_regrets, primary_choice_mismatches = [], 0
    for rows in states:
        modality, action = rows[0]['chosen_modality'], rows[0]['chosen_action']
        selected = [row for row in rows if row['modality'] == modality]
        gains = {metric: [0.] + [row[metric + '_gain'] for row in selected] for metric in METRICS}
        utilities = {metric: [gain - cost * ACTION_COST[index] for index, gain in enumerate(values)]
                     for metric, values in gains.items()}
        best = {metric: max(range(5), key=lambda index: values[index]) for metric, values in utilities.items()}
        clipped, executed = best['clipped'], best['executed']
        mismatch += clipped != executed
        stop_to_query += clipped == 0 and executed != 0
        query_to_stop += clipped != 0 and executed == 0
        chosen_counts[action] += 1
        clipped_counts[clipped] += 1
        executed_counts[executed] += 1
        clipped_regrets.append(utilities['executed'][executed] - utilities['executed'][clipped])
        policy_regrets.append(utilities['executed'][executed] - utilities['executed'][action])
        chosen_gains.append(gains['executed'][action])
        iou_gains.append(0. if action == 0 else selected[action - 1]['full_iou_gain'])
        main_losses = [rows[0]['executed_main_loss'], rows[4]['executed_main_loss']]
        primary_regrets.append(main_losses[modality] - min(main_losses))
        primary_choice_mismatches += main_losses[modality] > min(main_losses)
    count = len(states)
    return dict(labeled_states=count, clipped_vs_executed_budget_action_mismatches=mismatch,
                clipped_stop_executed_query_states=stop_to_query, clipped_query_executed_stop_states=query_to_stop,
                chosen_action_counts=chosen_counts, clipped_oracle_action_counts=clipped_counts,
                executed_oracle_action_counts=executed_counts,
                chosen_mean_auxiliary_tokens=sum(n * (0 if action == 0 else 385 if action == 4 else 193)
                                                  for action, n in enumerate(chosen_counts)) / count,
                chosen_mean_executed_localization_gain=statistics.mean(chosen_gains),
                chosen_mean_full_image_iou_gain=statistics.mean(iou_gains),
                mean_executed_utility_regret_of_clipped_oracle=statistics.mean(clipped_regrets),
                mean_executed_utility_regret_of_actual_policy=statistics.mean(policy_regrets),
                selected_primary_worse_than_other_states=primary_choice_mismatches,
                mean_executed_primary_loss_regret=statistics.mean(primary_regrets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    assert not output.exists()
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    complete = json.loads((run / 'COMPLETE.json').read_text(encoding='utf-8'))
    plan = config['plan']
    assert not config['sanity'] and not complete['sanity'] and complete['status'] == 'PASS'
    assert complete['optimizer_updates'] == 0 and complete['input_checkpoint_unchanged']
    assert complete['frozen_state_tensors'] == 265
    assert complete['checkpoint_sha256'] == plan['input_checkpoint_sha256']
    assert complete['sequences'] == plan['sequences'] == 20
    assert complete['tracked_frames'] == plan['expected_tracked_frames'] == 8854
    assert complete['labeled_states'] == plan['expected_labeled_states'] == 1752
    assert complete['action_pairs'] == plan['expected_action_pairs'] == 14016
    assert complete['actual_chosen_execution_bit_equal_checks'] == complete['labeled_states']
    groups = {}
    with (run / 'action_rewards.csv').open(newline='', encoding='utf-8') as handle:
        for record in csv.DictReader(handle):
            row = dict(sequence=record['sequence'])
            for name, value in record.items():
                if name in ('frame', 'modality', 'action', 'chosen_modality', 'chosen_action'):
                    row[name] = int(value)
                elif name in ('target_centre_inside', 'target_corners_inside'):
                    assert value in ('True', 'False')
                    row[name] = value == 'True'
                elif name != 'sequence':
                    row[name] = float(value)
                    assert math.isfinite(row[name]), (row['sequence'], name)
            assert row['sequence'] in plan['sequence_names']
            assert row['frame'] > 0 and row['frame'] % plan['label_stride'] == 0
            assert not row['target_corners_inside'] or row['target_centre_inside']
            assert row['chosen_modality'] in (0, 1) and row['chosen_action'] in range(5)
            for metric in METRICS:
                assert row[metric + '_gain'] == row[metric + '_main_loss'] - row[metric + '_action_loss']
            assert row['full_iou_gain'] == row['action_full_iou'] - row['main_full_iou']
            assert 0 <= row['main_full_iou'] <= 1 and 0 <= row['action_full_iou'] <= 1
            groups.setdefault((row['sequence'], row['frame']), []).append(row)
    states = list(groups.values())
    assert len(states) == complete['labeled_states']
    assert sum(map(len, states)) == complete['action_pairs']
    assert {key[0] for key in groups} == set(plan['sequence_names'])
    for rows in states:
        assert [(row['modality'], row['action']) for row in rows] == [(m, a) for m in range(2) for a in range(1, 5)]
        for name in ('chosen_modality', 'chosen_action', 'target_centre_inside', 'target_corners_inside'):
            assert len({row[name] for row in rows}) == 1
        for modality in range(2):
            for name in tuple(metric + '_main_loss' for metric in METRICS) + ('main_full_iou',):
                assert len({row[name] for row in rows if row['modality'] == modality}) == 1
    partitions = dict(all=states,
                      target_corners_inside=[rows for rows in states if rows[0]['target_corners_inside']],
                      centre_inside_corners_outside=[rows for rows in states if rows[0]['target_centre_inside']
                                                    and not rows[0]['target_corners_inside']],
                      target_centre_outside=[rows for rows in states if not rows[0]['target_centre_inside']])
    assert sum(len(v) for k, v in partitions.items() if k != 'all') == len(states)
    assert len(partitions['target_centre_outside']) == complete['outside_centre_states']
    all_rows = [row for rows in states for row in rows]
    summary = pair_summary(all_rows)
    assert summary['comparisons']['clipped_to_executed']['raw_sign_flip_pairs'] == complete['raw_sign_flip_pairs']
    report = dict(status='PASS', input_sha256={name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                                              for name in ('config.json', 'COMPLETE.json', 'action_rewards.csv')},
                  checkpoint_sha256=complete['checkpoint_sha256'], cost_weight=plan['cost_weight'],
                  partitions={name: dict(pairs=pair_summary([row for rows in selected for row in rows]),
                                         states=state_summary(selected, plan['cost_weight'])) if selected else
                              dict(pairs=0, states=0) for name, selected in partitions.items()},
                  by_modality_action={str(modality) + '_' + str(action): pair_summary(
                      [row for row in all_rows if row['modality'] == modality and row['action'] == action])
                      for modality in range(2) for action in range(1, 5)},
                  per_sequence={name: state_summary([rows for (seq, _), rows in groups.items() if seq == name],
                                                     plan['cost_weight']) for name in plan['sequence_names']},
                  scope='Frozen B train-only paired counterfactuals at the actually visited states; state-weighted means. '
                        'Oracle actions are diagnostic, not deployable or trajectory upper bounds. No NN/optimization, '
                        'LasHeR test accuracy, speedup or causal improvement claim.',
                  arithmetic='Oracle utility uses saved float scalars and rational token costs in Python float64, '
                             'ties choose STOP then action order. Actual policy action is read from the saved route; '
                             'this is not bitwise reproduction of GPU float32 policy argmax.',
                  sign_flip_definition='Strict gain product<0, no near-zero filtering. Quantiles [0,.01,.5,.99,1].')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), labeled_states=len(states), action_pairs=len(all_rows),
                         clipped_vs_executed_sign_flip_pairs=complete['raw_sign_flip_pairs'])))


if __name__ == '__main__':
    main()
