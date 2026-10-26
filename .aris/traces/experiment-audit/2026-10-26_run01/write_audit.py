import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
TRACE_REL = '.aris/traces/experiment-audit/2026-10-26_run01'
TRACE = ROOT / TRACE_REL
AGENT = '/root/audit_rankstop_training_decisions_20261026'
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
recomputed = read(TRACE / 'independent_recompute.json')
comparison = read(TRACE / 'local_comparison.json')
metadata = read(TRACE / 'remote_metadata.json')
assert recomputed['status'] == 'PASS' and not recomputed['failure_counts']
assert comparison['primary_report']['all_exact']
assert not comparison['conditional_report']['count_differences']
assert comparison['conditional_report']['max_absolute_mean_error'] == 0
assert comparison['original_plan']['exact_hash_bound'] and comparison['original_plan']['exact_sequence_order_matches']
assert all(record['historical_sha_exact_match'] and record['exact_text_after_two_checkpoint_path_substitutions_and_newline_normalization'] for record in comparison['source_resolution'].values())
assert all(sha(ROOT / name) == value for name, value in comparison['audited_local_input_sha256'].items())
assert all(metadata['artifacts'][name]['sha256'] == recomputed['inputs_sha256'][name] for name in metadata['artifacts'])
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
markdown = '''# RankSTOP saved TRAIN decision audit

Date: 2026-10-26. **Overall verdict: PASS within the saved TRAIN diagnostic scope; 0 blocking issues.**

Fresh delegated experiment-audit review. Requested reviewer: `gpt-6-astra`, reasoning `max`; actual model and effort route: **UNATTESTED**. Review independence: **same-family**. Acceptance: **provisional**. Reviewer owns this report and the independent recount; it did not modify or rerun the diagnostic producer. The deterministic arithmetic checks below are separate from semantic reviewer acceptance.

## Verified results

All **45/45 numerical fields** in `RANKSTOP_TRAIN_DECISION_DIAGNOSTIC_20261026.json` match the independently vectorized recount exactly. All **24 conditional groups**, including **120 counts and 24 means**, match `RANKSTOP_CONDITIONAL_QUERY_DIAGNOSTIC_20261026.json` exactly.

The census covers **182,182 update states**, **399,401 label rows**, **882 TRAIN sequences**, two rounds of **91,091** updates, seed **2027**, and valid frames at stride **5**. Both rounds match the hash-bound plan's exact sequence order and complete valid stride-5 frame order. There are 364,364 sampled labels and 35,037 selected-extra labels. The archived manifest has 459,435 TRAIN annotation frames; its expected two-round tracking count is 917,106 and initialization count is 1,764.

All 882 installed original TRAIN `init.txt` files match every original-manifest annotation after float32 conversion, without coordinate transformation. Independently computed **798,802 IoUs** and **399,401 gains** match saved values **exactly** (maximum absolute error 0). The verifier uses the original-image xywh intersection/union formula in NumPy float32, without importing Torch. Checking the saved float32 delta using float64 subtraction produces at most 2.980232238769531e-08 rounding residual; this is not a GT mismatch.

| Quantity | All | Round 1 | Round 2 |
|---|---:|---:|---:|
| Update states | 182182 | 91091 | 91091 |
| Executed queries | 44443 | 23260 | 21183 |
| Selected-extra outcomes | 35037 | 18249 | 16788 |
| Reused sampled outcomes | 9406 | 5011 | 4395 |
| STOP states | 137739 | 67831 | 69908 |
| Helpful executed queries, gain > .01 | 14535 | 7553 | 6982 |
| Harmful executed queries, gain < -.01 | 11057 | 5849 | 5208 |
| STOP with helpful sampled same-role alternative | 28630 | 14324 | 14306 |
| Mean primary routing regret | .04924217114643316 | .05064848304219745 | .04783585925066887 |
| Mean gain per executed query | .016101530431867832 | .014529244591109765 | .017827979407746258 |
| Mean reconstructed final IoU | .5211649696710093 | .5211670728809975 | .5211628664610211 |

The executed-query split is exact: selected-extra contributes 8,522 harmful / 11,110 helpful / 15,405 neutral outcomes; reused-sampled contributes 2,535 / 3,425 / 3,446. Every selected-extra flag, chosen modality, action kind, rank, same-role primary box/geometry/IoU, and reused-sampled match agrees. Both CSVs are fully exhausted, with contiguous step IDs and no duplicate round/sequence/frame keys. Every sampled pair has exactly one role-0 and one role-1 label and identical primary geometry.

## A–F checks

**A. Ground-truth provenance — PASS.** TRAIN rows come from `data/manifest.json` (`scripts/train_candidate_rankstop_trajectory.py:285`). Frame-0 GT initializes each trajectory (`:167`); current-frame GT is supplied to the teacher only after `tracker.track(image)` (`:173`, `:181`). `sampled_targets` computes executed primary and query IoU against that target (`scripts/train_candidate_evidence_policy.py:57`, `:72`, `:77`); `write_label` records those results (`scripts/train_candidate_rankstop_trajectory.py:141`). The original manifest SHA matches the training config, migrated TRAIN arrays/order/valid frames are identical, and all raw TRAIN text annotations and all saved IoUs/gains were independently checked. `dual_rgbt_evidence/regions.py:43` uses uncensored original-image GT.

**B. Score normalization — PASS.** Standard IoU uses geometric intersection divided by union (`dual_rgbt_evidence/regions.py:45`). There is no rescaling by prediction extrema or prediction-score statistics. The diagnostic divides gain by executed-query count for `mean_gain_per_executed_query`, and all other sums by update count (`tools/analyze_rankstop_training_decisions.py:72`, `:75`). Regret uses the better of two same-state primary IoUs as a comparison, not as a normalization denominator (`:36`, `:37`).

**C. Result existence and numbers — PASS for the audited diagnostic.** Both raw CSVs, config, and COMPLETE receipt exist and match their reported SHA-256 values. All 45 diagnostic values and 144 conditional-group statistics match exactly. `COMPLETE` counts and the older completion audit agree with the CSV/manifest census. The older completion audit's checkpoint/tensor/optimizer assertions were not re-audited because checkpoint loading was excluded; its missing native exit receipt remains unknown (`reports/RANKSTOP_C882_COMPLETION_AUDIT_20261026.json:36`). Full-frame route totals are internally consistent (917,106 frames and 223,466 queries), but this audit does not possess an independent per-frame runtime log for them.

**D. Dead code and derivation — PASS within the inspected paths.** The producer invokes `main` (`tools/analyze_rankstop_training_decisions.py:90`), reaches every reported aggregation, and its outputs are independently reproduced. The training path invokes sampled targets, selected replay, optimizer update, and label writing in order (`scripts/train_candidate_rankstop_trajectory.py:184`, `:186`, `:191`, `:202`, `:209`). Labels describe the pre-update decision even though they are written after that update. No original producer invocation receipt was supplied; numerical reproduction and static reachability do not attest that historical process invocation.

**E. Scope — PASS only as explicitly qualified TRAIN diagnostics.** The output itself restricts its claim (`reports/RANKSTOP_TRAIN_DECISION_DIAGNOSTIC_20261026.json:60`). This is one seed and one training trajectory schedule over two evolving rounds, with update-weighted, temporally dependent observations. It is neither final fixed-weight performance nor official PR/NPR/SR or an estimate over all tracked frames. No learning-improvement claim follows from round differences alone.

**F. Evaluation type — PASS, `real_gt`.** Dataset-provided TRAIN annotations determine the target; model outputs are predictions. This classification does not turn in-training descriptive measurements into held-out performance evidence.

## Denominators and permitted interpretation

- Primary routing comparisons use all 182,182 update states: the other primary is better by > .01 on 59,246 (32.520227%) and by > .1 on 29,479 (16.181072%). Mean same-state routing regret is .04924217114643316 IoU.
- Harmful/helpful executed-query fractions use 44,443 queries: 11,057/44,443 = 24.879059%, and 14,535/44,443 = 32.704813%. Mean gain is .016101530431867832 per executed query; .003927941931604122 averages the same gain, including STOP zeros, across all updates.
- Helpful sampled STOP alternatives use 137,739 STOP states: 28,630/137,739 = 20.785689%. This examines one sampled request for the chosen role, not every available query.
- Same-state primary comparisons, selected-query gains, and sampled STOP alternatives are descriptive GT oracle diagnostics. They do not establish a deployable GT gate, a causal module benefit, a full candidate-pool oracle, a trajectory upper bound, or a reason to disable a query family based on its unconditional average.

## Preserved provenance findings and limits

1. **Current-source mismatch resolved, not concealed.** The current training scripts differ from config hashes. Exact historical evidence-policy bytes were recovered from local Git history; trajectory bytes were recovered from commit `4898736ce0ee77f829d8014d2b4faf8b1671f079` with the historical CRLF representation. Their hashes exactly match the training config. Compared as text, each current file differs only in two protected-checkpoint path substitutions from `/home/gaob/CADTrack_ep0014.pth.tar` to the project-local checkpoint path. The label, decision, training-order, and arithmetic logic is unchanged. Source-history evidence is preserved in the trace. No Git state was changed.
2. **Same-history correspondence is source-backed.** Pre-track state/templates/history are copied at `scripts/train_candidate_rankstop_trajectory.py:172`; teachers use those saved values; selected replay asserts route/geometry/executed-box equality (`:47`, `:54`, `:64`, `:67`, `:82`); teacher-state preservation is asserted (`:188`, `:189`). Saved keys, primary geometries, and selected-role boxes align for every update. The CSVs do not archive per-step templates/history/weights, and this audit did not rerun neural replay, so it does not independently attest those runtime tensors or assertions.
3. **Historical terminal status remains unknown.** No checkpoint was loaded, no native exit status invented, and no prior frozen-tensor/optimizer claim was silently upgraded to a fresh verification. The new PASS is for the saved decision diagnostic, not a replacement for the older completion audit.
4. **Scope stays TRAIN only.** No images, active dev results/logs, resource/PID queries, neural runs, checkpoint loads, or job operations occurred. Source files and raw artifacts were read only. Review routing is UNATTESTED and same-family/provisional.

No code fix is required for these saved diagnostics. Keep these qualifiers whenever reporting the numbers; final fixed-weight evaluation requires separate evidence.

## Hash-bound evidence

- Audited diagnostic: `2dc047b90a5003104e3b35ed3614726f050bca93c8f9e07859c79f68a3b51d5c`.
- Diagnostic producer: `cc0530cb1d35c7aeb51d3047f062be46ab0ac30d2b5a9c3b6269ec5c9c6ae9b5` (local and server bytes equal).
- Raw labels: `6ddfdb92feae3294ea85211a672fe6bf1b07cdcdd5a8abc1dfdf38b908bbfa47`.
- Raw decisions: `2f22103097fdcaeb4483841cf5456160938a2ba0baadf393a7d70317f0a14e50`.
- COMPLETE: `fd2c8c534a5b22107971aa90133db0866db1fbc863494678769c36fcb6d17171`.
- Config: `5b8baf3f1a6d29c9f7d17ea06086fadb4673e4cc8584c81facfb9efe66f9c190`.
- Original manifest: `389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc`; migrated manifest: `dfc64432210b053fc9db517bae0771fdeb750a5d373b732949ef43e8b65b1a7a`.
- Plan: `a9a537a734511d0e7639ac050f01f97698af45881d29c38aca839ff59b875c08`.

Full inputs, original-GT file hashes, check counts, reconstructed groups, source-history comparisons, and audit-source hashes: `.aris/traces/experiment-audit/2026-10-26_run01/`. Machine-readable findings: `reports/RANKSTOP_TRAIN_DECISION_AUDIT_20261026.json`. Audit-output hashes are sealed in the trace's `output_seals.json` to avoid self-referential hashes.
'''
md_rel = 'reports/RANKSTOP_TRAIN_DECISION_AUDIT_20261026.md'
json_rel = 'reports/RANKSTOP_TRAIN_DECISION_AUDIT_20261026.json'
with (ROOT / md_rel).open('x', encoding='utf-8', newline='\n') as f:
    f.write(markdown)
evidence_files = ['independent_recompute.py', 'independent_recompute.json', 'remote_metadata.json',
                  'remote_manifest_metadata.json', 'source_history.json', 'local_comparison.json',
                  'historical_train_candidate_rankstop_trajectory.py', 'historical_train_candidate_evidence_policy.py',
                  'compare_saved_reports.py', 'check_source_history.py', 'write_audit.py']
checks = {
    'A_ground_truth_provenance': dict(status='PASS', evidence=['scripts/train_candidate_rankstop_trajectory.py:167', 'scripts/train_candidate_rankstop_trajectory.py:181', 'scripts/train_candidate_rankstop_trajectory.py:285', 'scripts/train_candidate_evidence_policy.py:57', 'scripts/train_candidate_evidence_policy.py:72', 'dual_rgbt_evidence/regions.py:43'], details='All 882 raw TRAIN GT files match original manifest in FP32. All saved IoUs and gains independently recompute exactly.'),
    'B_score_normalization': dict(status='PASS', evidence=['dual_rgbt_evidence/regions.py:45', 'tools/analyze_rankstop_training_decisions.py:72', 'tools/analyze_rankstop_training_decisions.py:75'], details='Geometric IoU and explicitly distinguished update/query count denominators; no prediction-statistic normalization.'),
    'C_result_existence': dict(status='PASS', evidence=['reports/RANKSTOP_TRAIN_DECISION_DIAGNOSTIC_20261026.json:3', 'reports/RANKSTOP_CONDITIONAL_QUERY_DIAGNOSTIC_20261026.json:6', 'reports/RANKSTOP_C882_COMPLETION_AUDIT_20261026.json:36'], details='45 primary diagnostic values and 144 conditional statistics exact. Raw hashes/counts match. Historical checkpoint/tensor claims excluded from fresh verification; native exit remains unknown.'),
    'D_dead_code_and_derivation': dict(status='PASS', evidence=['tools/analyze_rankstop_training_decisions.py:90', 'scripts/train_candidate_rankstop_trajectory.py:184', 'scripts/train_candidate_rankstop_trajectory.py:191', 'scripts/train_candidate_rankstop_trajectory.py:202'], details='Reachable calls and numerical reproduction verified; no original producer invocation receipt supplied.'),
    'E_scope': dict(status='PASS', evidence=['reports/RANKSTOP_TRAIN_DECISION_DIAGNOSTIC_20261026.json:60', 'reports/global_candidate_rankstop_trajectory_plan_20261018.json:3', 'reports/global_candidate_rankstop_trajectory_plan_20261018.json:951'], details='Only evolving TRAIN update states: one seed, two rounds, stride 5, 882 sequences, update-weighted dependent observations; no fixed-weight dev/test performance claim.'),
    'F_evaluation_type': dict(status='PASS', classification='real_gt', evidence=['dual_rgbt_evidence/regions.py:43', 'scripts/train_candidate_evidence_policy.py:57'], details='Dataset TRAIN annotations, not model-generated reference; held-out generalization is outside scope.')}
limitations = [
    'Only evolving TRAIN update states, not final fixed-weight dev/test or all 917106 tracked frames; no PR/NPR/SR, accuracy improvement, temporal independence, or cross-seed evidence.',
    'Same-history relation is supported by exact alignment and hash-bound source replay/state-preservation assertions, not independently verified per-step history/template/weight snapshots or NN replay.',
    'STOP alternatives are one sampled same-role request. Routing regret and query gains are same-state descriptive GT diagnostics, not deployable GT rules, full candidate-pool or trajectory bounds, or causal isolated module effects.',
    'Previous completion receipt has unknown native terminal status/exit code. Checkpoint SHA is receipt consistency only; checkpoint tensors, optimizer states, and frozen-state facts were not reloaded or freshly verified.',
    'Full-frame route counts are internally consistent receipt totals and manifest-derived expectations; an independent complete per-frame runtime log was not supplied.',
    'Original producer invocation has no supplied execution receipt. The producer source is hash-bound now and every reported numeric field is independently reproduced.',
    'Reviewer actual model/effort route is UNATTESTED; semantic acceptance is same-family/provisional.']
raw_hashes = {}
for name, value in recomputed['inputs_sha256'].items():
    raw_hashes[('/data2/gb/Dual-RGBT/' + name) if '/' in name else ('/data2/gb/Dual-RGBT/runs/global_candidate_rankstop_trajectory_formal_20261018/' + name)] = value
report = dict(audit_skill='experiment-audit', status='SCOPED_TRAIN_DECISION_AUDIT_PASS', verdict='PASS',
    overall_verdict='PASS', integrity_status='pass', reason_code='saved_train_decision_arithmetic_and_source_join_supported',
    summary='All saved TRAIN decision groups, conditional groups, raw GT IoUs/gains, selected-extra/reused-sampled alignment and denominators pass independent CPU recomputation; only evolving TRAIN diagnostic interpretation is supported.',
    generated_at=now, date='2026-10-26', agent_id=AGENT, verdict_id=AGENT + ':2026-10-26:1',
    reviewer_model_requested='gpt-6-astra', reviewer_reasoning_requested='max', reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED',
    reviewer_family='openai', review_independence='same-family', acceptance_status='provisional', actual_route_attestation='UNATTESTED',
    audit_ownership=dict(reviewer=AGENT, executor_interpretation_used=False, author_outputs_modified=False,
        producer_rerun=False, independent_recompute_written_by_reviewer=True, scope='read-only on archived inputs/source; new local reports and trace only'),
    deterministic_checks=dict(status='PASS', acceptance_status='accepted-for-reported-arithmetic-only',
        implementation='Independent stdlib/NumPy verifier; no imports of producer or training modules',
        python=recomputed['python'], numpy=recomputed['numpy'], no_torch_import=recomputed['no_torch_import']),
    blocking_issues=0, blockers=[], eval_type='real_gt', checks=checks,
    counts=recomputed['counts'], gt_recompute=recomputed['gt_recompute'], recomputed_groups=recomputed['groups'],
    selected_extra_and_reused_sampled_outcomes=recomputed['execution_outcomes'], recomputed_conditional_groups=recomputed['conditional_groups'],
    raw_alignment_check_counts=recomputed['checks'], raw_alignment_failure_counts=recomputed['failure_counts'],
    saved_report_comparison=comparison['primary_report'], conditional_report_comparison=comparison['conditional_report'],
    plan_binding=comparison['original_plan'], source_provenance_resolution=comparison['source_resolution'],
    rates_with_denominators=comparison['rates_with_denominators'],
    previous_completion_audit=dict(native_exit_code=None, native_status_available=False,
        current_scope='Counts/hash/internal consistency only; previous checkpoint/tensor/optimizer facts not freshly audited.',
        count_receipts_match=comparison['completion_count_receipts_match'], artifact_hashes_match=comparison['completion_artifact_hashes_match'],
        full_route_count_internal_consistency=comparison['full_route_count_internal_consistency']),
    claims=[dict(id='saved_train_decision_statistics', impact='supported'),
        dict(id='source_backed_same_state_pairing', impact='supported_with_runtime_evidence_limit'),
        dict(id='fixed_weight_dev_test_or_target_attainment', impact='unsupported_by_this_diagnostic'),
        dict(id='trajectory_oracle_or_causal_module_gain', impact='unsupported'),
        dict(id='round_difference_proves_learning_improvement', impact='unsupported')],
    limitations=limitations, required_code_fixes=[],
    audited_input_hashes=dict(comparison['audited_local_input_sha256'], **raw_hashes),
    raw_gt_census=dict(file_count=len(recomputed['gt_file_records']),
        canonical_records_sha256=hashlib.sha256(json.dumps(recomputed['gt_file_records'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        full_records_path=TRACE_REL + '/independent_recompute.json#/gt_file_records'),
    evidence_files_sha256={TRACE_REL + '/' + name: sha(TRACE / name) for name in evidence_files},
    audit_markdown=dict(path=md_rel, sha256=sha(ROOT / md_rel)), trace_path=TRACE_REL,
    output_hash_seal_path=TRACE_REL + '/output_seals.json',
    prohibited_operations_performed=dict(checkpoint_load=False, neural_execution=False, torch_import=False, image_read=False,
        active_dev_result_or_log_read=False, resource_or_pid_query=False, job_launch_or_stop=False, remote_write=False))
with (ROOT / json_rel).open('x', encoding='utf-8', newline='\n') as f:
    json.dump(report, f, indent=2)
    f.write('\n')
seals = dict(generated_at=now, outputs={json_rel: sha(ROOT / json_rel), md_rel: sha(ROOT / md_rel)})
(TRACE / 'output_seals.json').write_text(json.dumps(seals, indent=2) + '\n', encoding='utf-8')
request = '''Apply experiment-audit to saved TRAIN decision diagnostic. Fresh read-only same-family/provisional review; requested gpt-6-astra/max, actual route UNATTESTED unless attested. Read skills C:/Users/gb/.codex/skills/experiment-audit/SKILL.md and shared-references/local-codex-policy.md and applicable AGENTS. Paths only, no author interpretation supplied. Local root C:/Users/gb/projects/Dual-RGBT. Files: tools/analyze_rankstop_training_decisions.py; scripts/train_candidate_rankstop_trajectory.py; scripts/train_candidate_evidence_policy.py; dual_rgbt_global/candidate_tracker.py; reports/RANKSTOP_TRAIN_DECISION_DIAGNOSTIC_20261026.json; reports/RANKSTOP_C882_COMPLETION_AUDIT_20261026.json; reports/RANKSTOP_CONDITIONAL_QUERY_DIAGNOSTIC_20261026.json. Raw input paths on SSH alias2026: /data2/gb/Dual-RGBT/runs/global_candidate_rankstop_trajectory_formal_20261018/{query_labels.csv,trajectory_steps.csv,COMPLETE.json,config.json}. CPU Python there /data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python. You may read ONLY these archived training artifacts/source via SSH, independently recompute all groups and selected-extra/reused-sampled outcomes, validate raw input SHA/counts/GT-label arithmetic, check same-state pair alignment and denominators. Do NOT rerun producer with its existing output path, modify raw inputs, load checkpoint, run Torch/NN, query active dev logs/resources/PIDs or launch/stop jobs. Verify source derivation and all reported numbers; identify limitations and whether interpretation is restricted to evolving TRAIN update states. Check A-F from skill and audit ownership, output/source hashes. Write unique reports/RANKSTOP_TRAIN_DECISION_AUDIT_20261026.json and .md with exact evidence and blockers, preserve any findings. No need to touch HANDOFF or Git. Return findings and created paths.'''
extension = '''You may extend read-only CPU audit to installed TRAIN GT, without images: /data2/gb/Dual-RGBT/data/manifest.json; /data2/gb/Dual-RGBT/inputs/server25/manifest.json; referenced original train GT text files only. Also local reports/SERVER26_TRAIN_DEV_MIGRATION_20261026.json and source dual_rgbt/data.py, dual_rgbt_evidence/regions.py for provenance/math. No active dev results/progress. Independently recompute saved absolute executed main/query boxes vs original train GT with NumPy FP32 if feasible, preserving original manifest/raw GT transformations and numerical tolerances; no Torch/NN. Record exact scope if missing inputs prevent it.'''
trace_meta = dict(skill='experiment-audit', run_id='2026-10-26_run01', generated_at=now, executor='codex',
    executor_model='UNATTESTED', executor_family='openai', reviewer_model_requested='gpt-6-astra',
    reviewer_reasoning_requested='max', actual_model='UNATTESTED', actual_reasoning='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional', project_dir=str(ROOT), agent_id=AGENT)
(TRACE / 'run.meta.json').write_text(json.dumps(trace_meta, indent=2) + '\n', encoding='utf-8')
(TRACE / '001-training-decision-audit.request.json').write_text(json.dumps(dict(call_number=1, purpose='training-decision-audit',
    tool='spawn_agent', model_requested='gpt-6-astra', reasoning_effort_requested='max', actual_route='UNATTESTED',
    prompt=request, scope_extension_messages=[extension], files_referenced=list(comparison['audited_local_input_sha256']) + list(raw_hashes)), indent=2) + '\n', encoding='utf-8')
response = '''Scoped PASS, 0 blocking issues; same-family/provisional, actual model route UNATTESTED.

- All 45 decision-report values and 144 conditional statistics match exactly.
- All 798,802 saved IoUs and 399,401 gains exactly match original TRAIN GT under independent NumPy FP32 recomputation.
- All 182,182 joins pass; 44,443 queries split into 35,037 selected-extra and 9,406 reused sampled outcomes.
- Historical source hashes are recovered exactly; current differences are only migrated checkpoint paths. No diagnostic fix is needed.
- Interpretation remains evolving TRAIN update states. Native training exit is still unknown; checkpoint/NN replay was not performed.

Created:
- `reports/RANKSTOP_TRAIN_DECISION_AUDIT_20261026.json`
- `reports/RANKSTOP_TRAIN_DECISION_AUDIT_20261026.md`
- Full trace: `.aris/traces/experiment-audit/2026-10-26_run01/`
'''
response += '\nAudit JSON SHA-256: `' + seals['outputs'][json_rel] + '`\n'
response += 'Audit Markdown SHA-256: `' + seals['outputs'][md_rel] + '`\n'
(TRACE / '001-training-decision-audit.response.md').write_text(response, encoding='utf-8')
(TRACE / '001-training-decision-audit.meta.json').write_text(json.dumps(dict(call_number=1, purpose='training-decision-audit',
    timestamp=now, agent_id=AGENT, model='UNATTESTED', reasoning='UNATTESTED', reviewer_family='openai',
    review_independence='same-family', acceptance_status='provisional', status='ok', verdict='PASS',
    report_paths=[json_rel, md_rel], output_sha256=seals['outputs']), indent=2) + '\n', encoding='utf-8')
events = ROOT / '.aris/meta/events.jsonl'
events.parent.mkdir(parents=True, exist_ok=True)
with events.open('a', encoding='utf-8', newline='\n') as f:
    f.write(json.dumps(dict(event='review_trace', skill='experiment-audit', purpose='training-decision-audit',
        agent_id=AGENT, trace_path=TRACE_REL, status='ok', verdict='PASS', generated_at=now)) + '\n')
print(json.dumps(seals, indent=2))
