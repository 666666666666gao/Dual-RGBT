"""Read archived CSVs and TRAIN text annotations only; no producer import or Torch."""
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
import numpy as np

ROOT = Path('/data2/gb/Dual-RGBT')
RUN = ROOT / 'runs/global_candidate_rankstop_trajectory_formal_20261018'
checks = Counter()
failures = Counter()
examples = defaultdict(list)

def check(name, ok, evidence):
    checks[name] += 1
    if not ok:
        failures[name] += 1
        if len(examples[name]) < 6:
            examples[name].append(evidence)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

original = json.loads((ROOT / 'inputs/server25/manifest.json').read_text())
migrated = json.loads((ROOT / 'data/manifest.json').read_text())
config = json.loads((RUN / 'config.json').read_text())
complete = json.loads((RUN / 'COMPLETE.json').read_text())
check('original_manifest_bound_to_training_config', sha(ROOT / 'inputs/server25/manifest.json') == config['input_sha256']['data/manifest.json'], {})
check('train_sequence_order_preserved', [s['name'] for s in original['train']] == [s['name'] for s in migrated['train']], {})
check('unique_882_train_sequences', len({s['name'] for s in migrated['train']}) == len(migrated['train']) == 882, {})
truth = {}
valid = {}
gt_records = []
annotation_frames = 0
for old, new in zip(original['train'], migrated['train']):
    name = new['name']
    check('train_boxes_and_valid_frames_preserved', old['boxes'] == new['boxes'] and old['valid_frames'] == new['valid_frames'], name)
    p = Path(new['rgb'][0]).parent.parent / 'init.txt'
    raw = np.loadtxt(str(p), delimiter=',', dtype=np.float32)
    boxes = np.asarray(old['boxes'], dtype=np.float32)
    check('raw_gt_matches_original_manifest_fp32', raw.shape == boxes.shape and np.array_equal(raw, boxes, equal_nan=True), name)
    check('gt_shape_and_frame_lengths', raw.shape == (len(new['rgb']), 4) and len(new['tir']) == len(new['rgb']), name)
    truth[name] = raw
    valid[name] = set(new['valid_frames'])
    annotation_frames += len(raw)
    gt_records.append(dict(sequence=name, path=str(p), frames=len(raw), sha256=sha(p)))

with (RUN / 'trajectory_steps.csv').open(newline='') as f:
    step_rows = list(csv.DictReader(f))
n = len(step_rows)
check('step_ids_exact_contiguous', [int(r['step']) for r in step_rows] == list(range(1, n + 1)), n)
check('expected_step_count', n == complete['updates'] == 182182, n)
keys = [(int(r['round']), r['sequence'], int(r['frame'])) for r in step_rows]
check('no_duplicate_round_sequence_frame', len(keys) == len(set(keys)), n)
rounds = np.asarray([int(r['round']) for r in step_rows], dtype=np.int32)
roles = np.asarray([int(r['chosen_modality']) for r in step_rows], dtype=np.int32)
actions = np.asarray([int(r['chosen_action']) for r in step_rows], dtype=np.int32)
check('decision_domain', np.isin(roles, [0, 1]).all() and np.isin(actions, range(5)).all(), {})
observed_order = {}
for iteration in sorted(set(rounds.tolist())):
    relevant = [r for r in step_rows if int(r['round']) == iteration]
    order = [name for name, group in itertools.groupby(r['sequence'] for r in relevant)]
    observed_order[str(iteration)] = order
    expected = [(iteration, name, frame) for name in order for frame in sorted(valid[name]) if frame > 0 and frame % 5 == 0]
    actual = [(iteration, r['sequence'], int(r['frame'])) for r in relevant]
    check('each_round_all_882_once', len(order) == len(set(order)) == 882 and set(order) == set(truth), int(iteration))
    check('exact_stride5_valid_frame_coverage_and_order', actual == expected, int(iteration))
check('same_sequence_order_both_rounds', observed_order['1'] == observed_order['2'], {})
check('tracked_frame_count_from_manifest', 2 * sum(len(v) - 1 for v in truth.values()) == complete['tracked_frames'] == 917106, {})
check('initialize_count_from_manifest', 2 * len(truth) == complete['initializes'] == 1764, {})
loss_error = 0.0
for r in step_rows:
    values = [float(r[k]) for k in ('loss', 'quality_loss', 'gain_loss')]
    check('finite_logged_losses', all(math.isfinite(x) for x in values), int(r['step']))
    loss_error = max(loss_error, abs(values[0] - values[1] - values[2]))

primary = np.full((n, 2), np.nan, dtype=np.float64)
executed_gain = np.full(n, np.nan, dtype=np.float64)
sampled_same_role_gain = np.full(n, np.nan, dtype=np.float64)
outcome_type = np.zeros(n, dtype=np.int8)
conditional = defaultdict(lambda: dict(labels=0, positive=0, negative=0, near_zero=0, query_iou_at_least_0p5=0, gains=[]))
families = Counter()
purposes = Counter()
label_count = 0
label_groups = 0
arithmetic_max_error = 0.0
gt_error = dict(main=0.0, query=0.0, gain=0.0)
gt_nonexact = Counter()
gt_above_1e_6 = Counter()
gt_checked = 0
gt_buffers = {key: [] for key in ('main', 'query', 'target', 'main_iou', 'query_iou', 'gain', 'line')}

def iou(prediction, target):
    left = np.maximum(prediction[:, :2], target[:, :2])
    right = np.minimum(prediction[:, :2] + prediction[:, 2:], target[:, :2] + target[:, 2:])
    side = np.maximum(right - left, np.float32(0))
    intersection = side[:, 0] * side[:, 1]
    union = prediction[:, 2] * prediction[:, 3] + target[:, 2] * target[:, 3] - intersection
    return intersection / union

def flush_gt():
    global gt_checked
    if not gt_buffers['main']:
        return
    target = np.asarray(gt_buffers['target'], dtype=np.float32)
    main = iou(np.asarray(gt_buffers['main'], dtype=np.float32), target)
    query = iou(np.asarray(gt_buffers['query'], dtype=np.float32), target)
    for name, computed, saved in [('main', main, 'main_iou'), ('query', query, 'query_iou'), ('gain', query - main, 'gain')]:
        differences = np.abs(computed.astype(np.float64) - np.asarray(gt_buffers[saved], dtype=np.float64))
        gt_error[name] = max(gt_error[name], float(differences.max()))
        gt_nonexact[name] += int((differences != 0).sum())
        gt_above_1e_6[name] += int((differences > 1e-6).sum())
        bad = np.flatnonzero(differences > 1e-6)
        check('gt_' + name + '_iou_or_gain_within_1e_6', not len(bad), [{'csv_line': gt_buffers['line'][int(j)], 'difference': float(differences[j])} for j in bad[:3]])
    gt_checked += len(target)
    for values in gt_buffers.values():
        values.clear()

with (RUN / 'query_labels.csv').open(newline='') as f:
    for step_id, values_iter in itertools.groupby(csv.DictReader(f), key=lambda r: int(r['step'])):
        values = list(values_iter)
        label_groups += 1
        check('label_group_ids_exact_contiguous', step_id == label_groups, step_id)
        d = step_rows[step_id - 1]
        expected_key = (d['round'], d['sequence'], d['frame'])
        check('every_label_same_round_sequence_frame', all(tuple(r[k] for k in ('round', 'sequence', 'frame')) == expected_key for r in values), step_id)
        sampled = [r for r in values if r['purpose'] == 'sampled']
        extras = [r for r in values if r['purpose'] == 'selected_extra']
        check('exact_two_unique_sampled_roles', len(sampled) == 2 and {int(r['modality']) for r in sampled} == {0, 1}, step_id)
        check('only_known_label_purposes', len(sampled) + len(extras) == len(values), step_id)
        check('at_most_one_extra_and_flag_alignment', len(extras) <= 1 and bool(extras) == (d['extra_selected_query'] == 'True') and d['extra_selected_query'] in ('True', 'False'), step_id)
        sample_by_role = {int(r['modality']): r for r in sampled}
        check('same_primary_geometry_across_roles', sample_by_role[0]['main_geometry'] == sample_by_role[1]['main_geometry'], step_id)
        check('sampled_kind_cycle', int(d['sampled_kind']) == (step_id - 1) % 4 and all(int(r['kind']) == int(d['sampled_kind']) for r in sampled), step_id)
        for r in values:
            label_count += 1
            purposes[r['purpose']] += 1
            m, q, g = [float(r[k]) for k in ('main_iou', 'query_iou', 'delta_from_stop')]
            check('finite_and_bounded_label_scores', all(math.isfinite(x) for x in (m, q, g)) and 0 <= m <= 1 and 0 <= q <= 1 and -1 <= g <= 1, label_count + 1)
            arithmetic_max_error = max(arithmetic_max_error, abs(q - m - g))
            check('gain_arithmetic_within_1e_6', abs(q - m - g) < 1e-6, label_count + 1)
            decoded = {key: json.loads(r[key]) for key in ('main_geometry', 'query_geometry', 'main_raw_box', 'query_raw_box', 'main_box', 'query_box')}
            check('finite_saved_geometry_and_boxes', all(math.isfinite(x) for v in decoded.values() for x in v), label_count + 1)
            check('valid_target_frame', int(r['frame']) in valid[r['sequence']], label_count + 1)
            gt_buffers['main'].append(decoded['main_box'])
            gt_buffers['query'].append(decoded['query_box'])
            gt_buffers['target'].append(truth[r['sequence']][int(r['frame'])])
            gt_buffers['main_iou'].append(m)
            gt_buffers['query_iou'].append(q)
            gt_buffers['gain'].append(g)
            gt_buffers['line'].append(label_count + 1)
            if len(gt_buffers['main']) == 8192:
                flush_gt()
            regime = 'primary_lt_0p1' if m < .1 else 'primary_0p1_to_lt_0p5' if m < .5 else 'primary_ge_0p5'
            item = conditional[(r['purpose'], regime, int(r['kind']))]
            item['labels'] += 1
            item['positive'] += g > .01
            item['negative'] += g < -.01
            item['near_zero'] += -.01 <= g <= .01
            item['query_iou_at_least_0p5'] += q >= .5
            item['gains'].append(g)
            if r['purpose'] == 'sampled':
                families[r['modality'] + ':' + r['kind']] += 1
        role, action = int(d['chosen_modality']), int(d['chosen_action'])
        selected_sample = sample_by_role[role]
        primary[step_id - 1] = [float(sample_by_role[m]['main_iou']) for m in (0, 1)]
        sampled_same_role_gain[step_id - 1] = float(selected_sample['delta_from_stop'])
        if action == 0:
            check('stop_has_no_extra_and_rank_minus1', not extras and int(d['chosen_rank']) == -1, step_id)
            executed_gain[step_id - 1] = 0.0
        else:
            actual = extras[0] if extras else selected_sample
            check('executed_query_role_kind_rank_alignment', int(actual['modality']) == role and int(actual['kind']) == action - 1 and int(actual['candidate_rank']) == int(d['chosen_rank']), step_id)
            check('extra_exact_same_primary', all(actual[k] == selected_sample[k] for k in ('main_iou', 'main_box', 'main_raw_box', 'main_geometry', 'candidate_count')), step_id)
            pair_matches = (int(selected_sample['kind']), int(selected_sample['candidate_rank'])) == (action - 1, int(d['chosen_rank']))
            check('extra_iff_selected_not_sampled', bool(extras) != pair_matches, step_id)
            executed_gain[step_id - 1] = float(actual['delta_from_stop'])
            outcome_type[step_id - 1] = 1 if extras else 2
flush_gt()
check('full_steps_and_labels_exhausted', label_groups == len(step_rows), dict(groups=label_groups, steps=len(step_rows)))
check('expected_label_counts', label_count == complete['labelled_queries'] == 399401 and purposes['sampled'] == complete['sampled_query_labels'] == 364364 and purposes['selected_extra'] == complete['extra_selected_query_labels'] == 35037, dict(purposes))
check('sampled_family_counts_match_complete', dict(families) == complete['sampled_family_counts'], dict(families))
check('finite_complete_reconstruction', np.isfinite(primary).all() and np.isfinite(executed_gain).all() and np.isfinite(sampled_same_role_gain).all(), {})
chosen = primary[np.arange(n), roles]
best = primary.max(axis=1)
regret = best - chosen
groups = {}
outcome_groups = {}
for name, mask in [('all', np.ones(n, dtype=bool)), ('round1', rounds == 1), ('round2', rounds == 2)]:
    queried = mask & (actions != 0)
    stopped = mask & (actions == 0)
    groups[name] = dict(updates=int(mask.sum()), tir_primary=int((mask & (roles == 1)).sum()),
        other_primary_better_over_0p01=int((mask & (regret > .01)).sum()),
        other_primary_better_over_0p1=int((mask & (regret > .1)).sum()),
        queries=int(queried.sum()), stop=int(stopped.sum()),
        query_harmful_below_minus0p01=int((queried & (executed_gain < -.01)).sum()),
        query_helpful_over_0p01=int((queried & (executed_gain > .01)).sum()),
        stop_with_helpful_sampled_same_role_query=int((stopped & (sampled_same_role_gain > .01)).sum()),
        mean_gain_per_executed_query=float(executed_gain[queried].sum(dtype=np.float64) / queried.sum()),
        mean_chosen_primary_iou=float(chosen[mask].mean()), mean_best_primary_iou=float(best[mask].mean()),
        mean_primary_routing_regret=float(regret[mask].mean()), mean_executed_query_gain=float(executed_gain[mask].mean()),
        mean_final_iou=float((chosen + executed_gain)[mask].mean()))
    outcome_groups[name] = {}
    for tag, code in [('selected_extra', 1), ('reused_sampled', 2), ('stop', 0)]:
        selected = mask & (outcome_type == code)
        outcome_groups[name][tag] = dict(count=int(selected.sum()), harmful=int((selected & (executed_gain < -.01)).sum()),
            helpful=int((selected & (executed_gain > .01)).sum()), neutral=int((selected & (np.abs(executed_gain) <= .01)).sum()),
            mean_gain=float(executed_gain[selected].mean()), sum_gain=float(executed_gain[selected].sum(dtype=np.float64)))

conditional_rows = []
for (purpose, regime, kind), item in sorted(conditional.items()):
    values = item.pop('gains')
    conditional_rows.append(dict(purpose=purpose, primary_iou_regime=regime, kind=kind, mean_gain=math.fsum(values) / len(values), **item))
inputs = {name: sha(RUN / name) for name in ('query_labels.csv', 'trajectory_steps.csv', 'COMPLETE.json', 'config.json')}
inputs.update({name: sha(ROOT / name) for name in ('data/manifest.json', 'inputs/server25/manifest.json')})
result = dict(status='PASS' if not failures else 'FAIL', python=sys.version, numpy=np.__version__,
    inputs_sha256=inputs, checks=dict(checks), failure_counts=dict(failures), failure_examples=dict(examples),
    counts=dict(updates=n, label_groups=label_groups, labels=label_count, purposes=dict(purposes),
                train_sequences=len(truth), train_annotation_frames=annotation_frames, rounds=dict(Counter(rounds.tolist()))),
    gt_recompute=dict(rows=gt_checked, iou_values=2 * gt_checked, gain_values=gt_checked,
        dtype='numpy.float32', absolute_tolerance=1e-6, max_absolute_error=gt_error,
        nonexact_counts=dict(gt_nonexact), above_tolerance_counts=dict(gt_above_1e_6),
        label_arithmetic_max_absolute_error=arithmetic_max_error,
        raw_gt_manifest_fp32_direct_comparison='no coordinate transform; original-image xywh'),
    groups=groups, execution_outcomes=outcome_groups, conditional_groups=conditional_rows,
    sampled_family_counts=dict(families), loss_sum_max_error=loss_error,
    sequence_order_sha256=hashlib.sha256(json.dumps(observed_order, separators=(',', ':')).encode()).hexdigest(),
    gt_file_records=gt_records,
    no_torch_import='torch' not in sys.modules,
    scope='CPU archived TRAIN CSV and original TRAIN init.txt audit only; no images, checkpoint loading, neural execution, active logs, GPU resources, or processes.')
print(json.dumps(result, indent=2))
