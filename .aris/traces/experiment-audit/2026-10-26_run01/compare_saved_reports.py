import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
TRACE = ROOT / '.aris/traces/experiment-audit/2026-10-26_run01'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
recomputed = read(TRACE / 'independent_recompute.json')
target = read(ROOT / 'reports/RANKSTOP_TRAIN_DECISION_DIAGNOSTIC_20261026.json')
conditional = read(ROOT / 'reports/RANKSTOP_CONDITIONAL_QUERY_DIAGNOSTIC_20261026.json')
completion_audit = read(ROOT / 'reports/RANKSTOP_C882_COMPLETION_AUDIT_20261026.json')
metadata = read(TRACE / 'remote_metadata.json')
plan_path = ROOT / 'reports/global_candidate_rankstop_trajectory_plan_20261018.json'
plan = read(plan_path)
differences = []
numerical_cells = 0
for group, values in target['groups'].items():
    assert set(values) == set(recomputed['groups'][group])
    for key, value in values.items():
        numerical_cells += 1
        if value != recomputed['groups'][group][key]:
            differences.append(dict(group=group, key=key, saved=value, actual=recomputed['groups'][group][key]))
conditional_index = {(r['purpose'], r['primary_iou_regime'], r['kind']): r for r in recomputed['conditional_groups']}
conditional_differences = []
conditional_max_mean_error = 0.0
conditional_count_cells = 0
for row in conditional['groups']:
    actual = conditional_index[(row['purpose'], row['primary_iou_regime'], row['kind'])]
    for key in ('labels', 'positive', 'negative', 'near_zero', 'query_iou_at_least_0p5'):
        conditional_count_cells += 1
        if row[key] != actual[key]:
            conditional_differences.append(dict(group=actual, key=key, saved=row[key]))
    conditional_max_mean_error = max(conditional_max_mean_error, abs(row['mean_gain'] - actual['mean_gain']))
expected_order = {str(i): plan['sequence_names'] for i in range(1, plan['rounds'] + 1)}
expected_order_sha = hashlib.sha256(json.dumps(expected_order, separators=(',', ':')).encode()).hexdigest()
source_resolution = {}
for name in ('scripts/train_candidate_rankstop_trajectory.py', 'scripts/train_candidate_evidence_policy.py'):
    current = (ROOT / name).read_text(encoding='utf-8')
    historic_path = TRACE / ('historical_' + Path(name).name)
    historic = historic_path.read_text(encoding='utf-8')
    transformed = historic.replace("sha('/home/gaob/CADTrack_ep0014.pth.tar')", "sha(ROOT / 'checkpoints/CADTrack_ep0014.pth.tar')")
    source_resolution[name] = dict(current_sha256=sha(ROOT / name), historical_sha256=sha(historic_path),
        training_config_sha256=metadata['config']['source_sha256'][name],
        historical_sha_exact_match=sha(historic_path) == metadata['config']['source_sha256'][name],
        historical_checkpoint_path_occurrences=historic.count("sha('/home/gaob/CADTrack_ep0014.pth.tar')"),
        exact_text_after_two_checkpoint_path_substitutions_and_newline_normalization=current == transformed)
files = ['tools/analyze_rankstop_training_decisions.py', 'scripts/train_candidate_rankstop_trajectory.py',
         'scripts/train_candidate_evidence_policy.py', 'dual_rgbt_global/candidate_tracker.py',
         'dual_rgbt/data.py', 'dual_rgbt_evidence/regions.py',
         'reports/RANKSTOP_TRAIN_DECISION_DIAGNOSTIC_20261026.json',
         'reports/RANKSTOP_C882_COMPLETION_AUDIT_20261026.json',
         'reports/RANKSTOP_CONDITIONAL_QUERY_DIAGNOSTIC_20261026.json',
         'reports/SERVER26_TRAIN_DEV_MIGRATION_20261026.json',
         'reports/global_candidate_rankstop_trajectory_plan_20261018.json']
all_group = recomputed['groups']['all']
complete = metadata['complete']
result = dict(
    primary_report=dict(group_count=len(target['groups']), numeric_cells=numerical_cells,
        all_exact=not differences, differences=differences),
    conditional_report=dict(group_count=len(conditional['groups']), count_cells=conditional_count_cells,
        count_differences=conditional_differences, mean_cells=len(conditional['groups']),
        max_absolute_mean_error=conditional_max_mean_error,
        all_means_within_1e_15=conditional_max_mean_error < 1e-15,
        total_labels=sum(r['labels'] for r in conditional['groups']),
        metadata_count_matches=conditional['labelled_queries'] == recomputed['counts']['labels'],
        metadata_label_sha_matches=conditional['label_sha256'] == recomputed['inputs_sha256']['query_labels.csv'],
        checkpoint_receipt_matches_only=conditional['checkpoint_sha256'] == complete['checkpoint_sha256']),
    original_plan=dict(path=str(plan_path), sha256=sha(plan_path), config_sha256=metadata['config']['plan_sha256'],
        exact_hash_bound=sha(plan_path) == metadata['config']['plan_sha256'],
        expected_sequence_order_sha256=expected_order_sha,
        observed_sequence_order_sha256=recomputed['sequence_order_sha256'],
        exact_sequence_order_matches=expected_order_sha == recomputed['sequence_order_sha256'],
        seed=plan['seed'], rounds=plan['rounds'], label_stride=plan['label_stride'], cost_weight=plan['cost_weight'],
        max_candidates=plan['max_candidates']),
    source_resolution=source_resolution,
    audited_local_input_sha256={name: sha(ROOT / name) for name in files},
    producer_local_remote_byte_equal=sha(ROOT / files[0]) == metadata['sources'][files[0]]['sha256'],
    completion_count_receipts_match={key: completion_audit['counts'][key] == complete[key] == plan['expected_formal_counts'][key]
                                    for key in ('updates', 'initializes', 'tracked_frames')},
    completion_artifact_hashes_match={name: value == recomputed['inputs_sha256'][Path(name).name]
                                    for name, value in completion_audit['artifacts_sha256'].items()},
    completion_native_status_unknown_preserved=completion_audit['actual_process_exit_code'] is None and not completion_audit['current_native_status_available'],
    full_route_count_internal_consistency=dict(tracked_frames=sum(complete['route_frames'].values()),
        query_frames=sum(v for k, v in complete['route_frames'].items() if k.split(':')[1] != '0'),
        saved_tracking_auxiliary_queries=complete['tracking_auxiliary_queries']),
    rates_with_denominators={
        'other_primary_better_over_0p01': dict(numerator=59246, denominator=182182, fraction=59246 / 182182),
        'other_primary_better_over_0p1': dict(numerator=29479, denominator=182182, fraction=29479 / 182182),
        'harmful_among_executed_queries': dict(numerator=11057, denominator=44443, fraction=11057 / 44443),
        'helpful_among_executed_queries': dict(numerator=14535, denominator=44443, fraction=14535 / 44443),
        'helpful_sampled_query_among_stop_states': dict(numerator=28630, denominator=137739, fraction=28630 / 137739)},
    audit_code_sha256={p.name: sha(p) for p in TRACE.glob('*.py')})
(TRACE / 'local_comparison.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
