"""Write this independent source review and its exact, uninstalled guard payloads."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path('C:/Users/gb/projects/Dual-RGBT')
BASE = ROOT / '.aris/traces/experiment-audit/2026-10-27_query_evidence_capture'
OUT = BASE / 'source_review'
OLD = BASE / 'pre_change'
PLAN = 'reports/QUERY_EVIDENCE_CAPTURE_ADMISSION_PLAN_20261027.json'
PROTOCOLS = [
    'reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json',
    'reports/VRAM18_EVALUATION_PROTOCOL_20261027.json',
    'reports/SERVER25_JOINT_EVALUATION_PROTOCOL_20261027.json',
]
CHANGED = [
    'scripts/query_evidence_capture.py',
    'scripts/run_joint_candidate_evaluation.py',
    'scripts/joint_evaluation_contract.py',
    'tools/check_query_evidence_capture.py',
    'tools/register_query_evidence_capture.py',
]
READ_SOURCES = CHANGED + [
    'dual_rgbt_global/candidate_tracker.py',
    'dual_rgbt_global/candidate_policy.py',
    'dual_rgbt_global/tracker.py',
    'dual_rgbt_evidence/regions.py',
    'dual_rgbt_evidence/readout.py',
    'dual_rgbt_evidence/policy.py',
    'dual_rgbt/model.py',
    'dual_rgbt/data.py',
    'scripts/run_benchmark_tracking.py',
    'tools/joint_candidate_results.py',
    'lib/utils/box_ops.py',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


now = datetime.now(timezone.utc).isoformat()
plan = json.loads((ROOT / PLAN).read_bytes())
assert sha(ROOT / PLAN) == '377a5f449481e876c737b01a91e197befdbfc7ef655fbb5e0f317399b3257ba2'
assert all(sha(ROOT / n) == h for n, h in plan['source_sha256'].items())
for name in CHANGED:
    compile(ast.parse((ROOT / name).read_bytes(), filename=name), name, 'exec')

protected_path = BASE / 'protected_training_sources.json'
protected = json.loads(protected_path.read_bytes())
training_sources = {}
for name in protected['training_plans']:
    training = json.loads((ROOT / name).read_bytes())
    for source, digest in training['source_sha256'].items():
        if source in training_sources:
            assert training_sources[source] == digest
        training_sources[source] = digest
    head = subprocess.run(['git', 'show', 'HEAD:' + name], cwd=ROOT,
                          capture_output=True, check=True).stdout
    assert (ROOT / name).read_bytes() == head, name
assert training_sources == protected['source_sha256']
assert all(sha(ROOT / n) == h for n, h in training_sources.items())
assert not set(CHANGED).intersection(training_sources)

contract = 'scripts/joint_evaluation_contract.py'
old_tree = ast.parse((OLD / contract).read_bytes())
new_tree = ast.parse((ROOT / contract).read_bytes())
old_functions = {n.name: ast.dump(n, include_attributes=False) for n in old_tree.body
                 if isinstance(n, ast.FunctionDef)}
new_functions = {n.name: ast.dump(n, include_attributes=False) for n in new_tree.body
                 if isinstance(n, ast.FunctionDef)}
assert set(old_functions) == set(new_functions)
changed_functions = [n for n in old_functions if old_functions[n] != new_functions[n]]
assert changed_functions == ['verify_runs']
for name in ['scripts/run_joint_candidate_evaluation.py', contract]:
    head = subprocess.run(['git', 'show', 'HEAD:' + name], cwd=ROOT,
                          capture_output=True, check=True).stdout
    assert (OLD / name).read_bytes() == head

protocol_checks = {}
source_delta = set(CHANGED) - {'tools/check_query_evidence_capture.py'}
for name in PROTOCOLS:
    old = json.loads((OLD / name).read_bytes())
    spec = json.loads((ROOT / name).read_bytes())
    assert {k for k in set(old) | set(spec) if old.get(k) != spec.get(k)} == {
        'source_sha256', 'scope', 'evidence_schema'}
    assert {k for k in set(old['source_sha256']) | set(spec['source_sha256'])
            if old['source_sha256'].get(k) != spec['source_sha256'].get(k)} == source_delta
    assert spec['scope'].startswith(old['scope'])
    assert all(sha(ROOT / n) == h for n, h in spec['source_sha256'].items())
    assert all(sha(ROOT / n) == h for n, h in spec['training_plan_sha256'].items())
    assert (ROOT / spec['source_review']).read_bytes() == (OLD / spec['source_review']).read_bytes()
    protocol_checks[name] = dict(
        sha256=sha(ROOT / name), source_entries=len(spec['source_sha256']),
        training_plan_entries=len(spec['training_plan_sha256']),
        changed_top_level=['source_sha256', 'scope', 'evidence_schema'],
        changed_sources=sorted(source_delta), all_source_hashes_match=True,
        all_training_plan_hashes_match=True, live_guard_unchanged=True,
        input_hash_map_unchanged=True, provider_unchanged=True,
        training_plan_hash_map_unchanged=True, selection_and_metric_contract_unchanged=True)

search = subprocess.run([
    'rg', '-l',
    '"protocol"\\s*:\\s*"reports/(JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026|VRAM18_EVALUATION_PROTOCOL_20261027|SERVER25_JOINT_EVALUATION_PROTOCOL_20261027)\\.json"',
    'reports', '--glob', '*.json'], cwd=ROOT, capture_output=True, text=True)
assert search.returncode in (0, 1), search.stderr
bound_files = search.stdout.splitlines()
model_plans = []
for name in bound_files:
    obj = json.loads((ROOT / name).read_bytes())
    if 'candidate_id' in obj and 'checkpoint' in obj:
        model_plans.append(name)
assert not model_plans
assert not (ROOT / 'reports/joint_candidate_evaluation').exists()

deterministic = dict(
    status='PASS', generated_at=now, compile_only_files=CHANGED,
    admission_plan_sha256=sha(ROOT / PLAN),
    admission_source_entries=len(plan['source_sha256']),
    admission_source_hashes_match=True,
    protected_training_plan_count=len(protected['training_plans']),
    protected_training_plans_byte_equal_git_head=True,
    independently_reconstructed_training_source_count=len(training_sources),
    independently_reconstructed_training_source_map_matches=True,
    protected_training_sources_match=True, changed_sources_disjoint_from_training=True,
    pre_change_evaluator_and_contract_equal_git_head=True,
    unchanged_contract_functions=[n for n in old_functions if n not in changed_functions],
    changed_contract_functions=changed_functions,
    protocols=protocol_checks,
    report_jsons_with_exact_protocol_field=bound_files,
    current_model_specific_evaluation_plans=model_plans,
    current_model_specific_evaluation_directory_absent=True,
    local_train_manifest_present=(ROOT / 'data/manifest.json').exists(),
    neural_network_execution=False, remote_queries=False, checkpoint_reads=False,
    limitations=['Compile-only checks do not establish runtime parity or accuracy.',
                 'The local data/manifest.json is absent; TRAIN membership and input-byte checks must execute on the registered provider.'])
save('DETERMINISTIC_CHECKS.json', deterministic)

checks = {
    'gt_provenance': dict(status='PASS', evidence=[
        'scripts/query_evidence_capture.py:15', 'scripts/query_evidence_capture.py:23',
        'scripts/run_joint_candidate_evaluation.py:127', 'scripts/run_joint_candidate_evaluation.py:140',
        'tools/check_query_evidence_capture.py:52', 'tools/check_query_evidence_capture.py:64',
        'dual_rgbt/data.py:15'], details=
        'Capture accepts only the model, cost, image geometry, prior tracker box, route and executed box. It does not receive GT. Benchmark GT is loaded after causal sequence prediction. The admission indexes the registered train split and reads real RGB/TIR files; only frame-zero dataset box initializes tracking. Later frames receive images only.'),
    'score_normalization': dict(status='PASS', evidence=[
        'scripts/query_evidence_capture.py:45', 'scripts/query_evidence_capture.py:50',
        'dual_rgbt_global/candidate_policy.py:43', 'dual_rgbt_global/candidate_policy.py:84',
        'tools/joint_candidate_results.py:45'], details=
        'Recorded selected_gain is the raw predecision signed gain at request_index+1. Cost is the existing action token_cost, and utility is gain minus configured cost_weight times cost. Internal confidence-distribution features are not reported as benchmark accuracy. Existing official metric computation is unchanged.'),
    'result_existence': dict(status='PASS', evidence=[
        'tools/check_query_evidence_capture.py:89', 'reports/QUERY_EVIDENCE_CAPTURE_ADMISSION_PLAN_20261027.json:150'], details=
        'No new accuracy, parity, FPS or completed-run result is claimed by this SOURCE review. The checker writes CAPTURE_REAL_TRAJECTORY_PASS only after all comparisons and branch checks. Runtime result validation is pending, and the local TRAIN manifest is absent.'),
    'dead_code': dict(status='PASS', evidence=[
        'scripts/run_joint_candidate_evaluation.py:82', 'scripts/run_joint_candidate_evaluation.py:127',
        'scripts/run_joint_candidate_evaluation.py:130', 'scripts/run_joint_candidate_evaluation.py:138',
        'scripts/joint_evaluation_contract.py:214', 'tools/joint_candidate_results.py:80'], details=
        'The runner instantiates capture, begins every tracked frame, finishes it and writes evidence CSV. verify_runs consumes every evidence CSV before scoring; score and freeze both use verify_runs. This is a reachable observation path, not an unused metric helper.'),
    'scope': dict(status='PASS', evidence=[
        'reports/QUERY_EVIDENCE_CAPTURE_ADMISSION_PLAN_20261027.json:138',
        'tools/check_query_evidence_capture.py:55', 'tools/check_query_evidence_capture.py:85'], details=
        'Admission is bounded to three fixed TRAIN prefixes, eleven frames each, costs zero and one hundred, each plain and observed: sixty tracked frames per mode. High cost is an explicit STOP fixture, not a deployment choice. Both STOP and QUERY must occur. This establishes only execution equivalence if actually passed, never final-model quality or general performance.'),
    'observation_only': dict(status='PASS', evidence=[
        'scripts/query_evidence_capture.py:17', 'scripts/query_evidence_capture.py:28',
        'scripts/query_evidence_capture.py:34', 'dual_rgbt_global/candidate_tracker.py:31',
        'dual_rgbt_global/candidate_tracker.py:38', 'dual_rgbt_evidence/policy.py:90'], details=
        'Each wrapper invokes the original bound method once, retains references to its result and returns the original output unchanged. No encoder, query, routing or tracker-state update is added. The original query reader consumes primary features without mutating retained box/score; gain selection and template/history transitions remain in the unchanged tracker.'),
    'primary_hann_box_and_original_coordinates': dict(status='PASS', evidence=[
        'dual_rgbt_global/tracker.py:29', 'dual_rgbt/model.py:92',
        'dual_rgbt/model.py:99', 'dual_rgbt_global/candidate_tracker.py:31',
        'scripts/query_evidence_capture.py:24', 'scripts/query_evidence_capture.py:40',
        'dual_rgbt_evidence/readout.py:49', 'dual_rgbt_evidence/regions.py:35'], details=
        'Captured box is the actual primary box after its Hann-window score and cal_bbox decode. Capture saves search_geometry from the pre-track state, uses the same pixel_xywh conversion and execute_boxes margin-ten float32 clipping against the original image size. It does not substitute an unwindowed peak, crop-clipped GT or final query box.'),
    'selected_gain_cost_and_stop': dict(status='PASS', evidence=[
        'scripts/query_evidence_capture.py:45', 'scripts/query_evidence_capture.py:48',
        'dual_rgbt_global/candidate_tracker.py:38', 'dual_rgbt_global/candidate_tracker.py:42',
        'dual_rgbt_global/candidate_tracker.py:66', 'dual_rgbt/model.py:47'], details=
        'request_index+1 recovers the original selected gain index. action=kinds+1 recovers the same token cost used for selection. STOP is the explicit gain-zero/cost-zero entry, and finish asserts request zero plus np.array_equal of reconstructed executed primary and final box. CSV verification allows only the existing six-decimal prediction serialization tolerance for this relation.'),
    'csv_completeness_and_sealing': dict(status='PASS', evidence=[
        'scripts/query_evidence_capture.py:8', 'scripts/run_joint_candidate_evaluation.py:112',
        'scripts/run_joint_candidate_evaluation.py:120', 'scripts/joint_evaluation_contract.py:188',
        'scripts/joint_evaluation_contract.py:213', 'scripts/joint_evaluation_contract.py:217',
        'scripts/joint_evaluation_contract.py:237', 'tools/joint_candidate_results.py:109',
        'tools/joint_candidate_results.py:135'], details=
        'Each sequence has one evidence row for every tracked frame (initialization intentionally excluded). Verification requires the exact sequence set and ordered complete frame IDs, finite positive primary boxes, finite evidence values, normalized token cost agreement, utility arithmetic and STOP zeros/box equality. Every evidence file is included in recursive run_artifacts_sha256; scored reports bind that map, and freeze recomputes it. Initial GT and prediction contracts remain unchanged.'),
    'existing_guards_and_training_unchanged': dict(status='PASS', evidence=[
        'scripts/joint_evaluation_contract.py:21', 'scripts/joint_evaluation_contract.py:32',
        'scripts/joint_evaluation_contract.py:70', 'scripts/joint_evaluation_contract.py:94',
        'scripts/joint_evaluation_contract.py:127', 'scripts/joint_evaluation_contract.py:258',
        'scripts/joint_evaluation_contract.py:284', 'tools/register_query_evidence_capture.py:14',
        'DETERMINISTIC_CHECKS.json'], details=
        'All ten existing contract functions except verify_runs are AST-identical to the byte-preserved pre-change source; verify_runs only adds evidence requirements. Complete-C/native success, full counts, exact tensor witness, provider-native command, smoke/reload, dev audit and freeze gates remain. All forty protected training plans equal Git HEAD; fifty-seven independently reconstructed source hashes match, with no changed-source intersection. Three protocol input/training/provider/metric/selection fields are unchanged. Current model-specific evaluation plans and their standard output directory are absent.'),
    'runtime_admission_design': dict(status='PASS', evidence=[
        'tools/check_query_evidence_capture.py:26', 'tools/check_query_evidence_capture.py:33',
        'tools/check_query_evidence_capture.py:45', 'tools/check_query_evidence_capture.py:55',
        'tools/check_query_evidence_capture.py:80', 'tools/check_query_evidence_capture.py:85',
        'tools/check_query_evidence_capture.py:90'], details=
        'The source gate binds the exact plan and source maps and verifies fixed input hashes. A native RUNNING parent and exact command/GPU zero are required. One strictly loaded historical candidate_b step-6250 non-sanity model is frozen and reused for plain then observed trajectories. Every box and non-time route is compared exactly; final templates are byte-hashed and history compared exactly for each sequence/cost. Both branches, unchanged state_dict and absent gradients are required. The result binds plan and checkpoint hashes. There is no optimizer, current-final checkpoint path, pause/kill operation or FPS measurement in this checker. Runtime success itself has not been asserted.'),
    'evaluation_type': {'benchmark': 'real_gt', 'admission': 'execution_equivalence_only',
                        'note': 'Admission is a correctness check on real TRAIN images, not an accuracy evaluation or model-output-as-GT benchmark.'},
}
common = dict(
    audit_skill='experiment-audit', audit_phase='SOURCE', status='SOURCE_PASS',
    blocking_issues=0, issues=[], verdict='PASS', overall_verdict='pass', integrity_status='pass',
    generated_at=now, date='2026-10-27',
    agent_task_name='/root/audit_query_evidence_capture_20261027',
    requested_reviewer_model='gpt-6-astra', requested_reviewer_reasoning_effort='max',
    reviewer_model='UNATTESTED', actual_model='UNATTESTED', actual_reasoning_effort='UNATTESTED',
    reviewer_family='openai', review_independence='same-family', acceptance_status='provisional',
    attribution_note='Fresh native reviewer. Astra/max was requested; actual model and effort are not independently attested by this reviewer.',
    trace_path=OUT.relative_to(ROOT).as_posix(),
    summary='SOURCE_PASS with zero blockers for exact reviewed bytes. Observation semantics and evidence sealing are correct by source inspection; actual real-TRAIN parity admission remains pending.',
    neural_network_execution=False, remote_queries=False, checkpoint_reads=False,
    source_only=True, runtime_admission_status='NOT_RUN_BY_REVIEWER',
    claim_limits=[
        'No new benchmark accuracy, query benefit or FPS result is established by this review.',
        'Primary-vs-final boxes support same-state immediate executed QUERY IoU comparison only after joining official GT offline.',
        'STOP missed opportunity, unexecuted alternatives and long-horizon counterfactual trajectories remain unobserved.',
        'High-cost admission is a branch-coverage fixture, not a selected deployment cost.',
        'Unchanged incidental evaluation timings are not dedicated profiling or evidence of speed improvement.'],
    operational_requirements=[
        'Install only payloads whose current plan/protocol and complete source maps still match; reviewer did not overwrite live guards.',
        'The exact planned real-TRAIN plain/observed runtime admission still needs to pass on its registered provider.',
        'Existing completed-C, native ownership/capacity, smoke, dev-result audit and frozen-selection gates remain necessary for future model evaluations.'])

input_names = READ_SOURCES + [PLAN] + PROTOCOLS + [protected_path.relative_to(ROOT).as_posix()]
hashes = {name: sha(ROOT / name) for name in input_names}
for name in PROTOCOLS + ['scripts/run_joint_candidate_evaluation.py', contract]:
    hashes[(OLD / name).relative_to(ROOT).as_posix()] = sha(OLD / name)
review = dict(common, checks=checks, audited_input_hashes=hashes,
              admission_plan=PLAN, admission_plan_sha256=sha(ROOT / PLAN),
              protocol_bindings={n: protocol_checks[n]['sha256'] for n in PROTOCOLS},
              deterministic_checks='DETERMINISTIC_CHECKS.json')
save('SOURCE_REVIEW.json', review)

admission_guard = dict(common, plan=PLAN, plan_sha256=sha(ROOT / PLAN),
                       source_sha256=plan['source_sha256'], checks=checks,
                       guard_note='Uninstalled reviewer payload. This is source approval only; it does not assert CAPTURE_REAL_TRAJECTORY_PASS.')
save('guard_payloads/' + Path(plan['source_review']).name, admission_guard)
for name in PROTOCOLS:
    spec = json.loads((ROOT / name).read_bytes())
    guard = dict(common, protocol=name, protocol_sha256=sha(ROOT / name),
                 source_sha256=spec['source_sha256'], checks=checks,
                 guard_note='Uninstalled reviewer payload. Exact protocol/source binding; no NN execution, active-weight access or runtime success is asserted.')
    save('guard_payloads/' + Path(spec['source_review']).name, guard)

lines = [
    '# Query evidence capture: independent source audit', '',
    '**SOURCE_PASS; blocking_issues = 0.**', '',
    'Fresh native experiment-audit review; requested gpt-6-astra/max, actual model/effort UNATTESTED. '
    'Review independence: same-family. Acceptance: provisional.', '',
    'This source pass covers the exact hashes in SOURCE_REVIEW.json. No neural network, checkpoint read, '
    'remote query, training pause or FPS measurement was performed by the reviewer. '
    'The real-TRAIN parity result is still pending; the local TRAIN manifest is absent.', '',
]
for name, item in checks.items():
    if name == 'evaluation_type':
        continue
    lines.extend(['## ' + name + ': ' + item['status'], '', item['details'], '',
                  'Evidence: ' + '; '.join(item['evidence']) + '.', ''])
lines.extend([
    '## Classification and claim limits', '',
    'Benchmark scoring remains real_gt. The admission is execution-equivalence only on real TRAIN images; '
    'comparing two causal outputs is not being represented as accuracy.', '',
    *['- ' + x for x in common['claim_limits']], '',
    '## Deterministic checks', '',
    '- Five changed/added Python files compile without importing or executing NN code.',
    '- Forty protected training plans equal Git HEAD byte for byte; all fifty-seven reconstructed training-source hashes match.',
    '- All admission and protocol source hashes match. Only evidence source bindings, evidence_schema and appended scope text changed in the protocols.',
    '- Complete-C/admission/dev-freeze/source/native guard functions remain unchanged. The three live guards remain byte-identical to their pre-change copies.',
    '- No current model-specific evaluation plan was found; reports/joint_candidate_evaluation is absent.', '',
    '## Exact bindings', '',
    '- Admission plan: `' + sha(ROOT / PLAN) + '`.',
    *['- ' + Path(n).name + ': `' + sha(ROOT / n) + '`.' for n in PROTOCOLS], '',
    'Four separate guard payloads are under guard_payloads/. They have not been installed into reports/. '
    'The live guards are intentionally stale until the executor verifies and installs the exact matching payloads.', '',
    '## Remaining execution requirements', '',
    *['- ' + x for x in common['operational_requirements']], '',
    'No source blocker or requested implementation change was found.', '',
])
(OUT / 'SOURCE_REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
(OUT / '001-source-review.response.md').write_text('\n'.join(lines), encoding='utf-8')
save('001-source-review.request.json', dict(call_number=1, purpose='source-review',
    tool='spawn_agent', requested_model='gpt-6-astra', requested_reasoning_effort='max',
    files_referenced=input_names, prompt='Fresh experiment-audit SOURCE review required by project AGENTS.md and C:/Users/gb/.codex/skills/experiment-audit/SKILL.md. Root C:/Users/gb/projects/Dual-RGBT. Review scripts/query_evidence_capture.py, scripts/run_joint_candidate_evaluation.py, scripts/joint_evaluation_contract.py, tools/check_query_evidence_capture.py, tools/register_query_evidence_capture.py; reports/QUERY_EVIDENCE_CAPTURE_ADMISSION_PLAN_20261027.json and three protocols reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json, VRAM18_EVALUATION_PROTOCOL_20261027.json, SERVER25_JOINT_EVALUATION_PROTOCOL_20261027.json. Prior exact evaluation bytes in .aris/traces/experiment-audit/2026-10-27_query_evidence_capture/pre_change/. Relevant originals dual_rgbt_global/candidate_tracker.py,candidate_policy.py, dual_rgbt_evidence/regions.py,readout.py, dual_rgbt/model.py. Verify capture observational only (noGT, no extraencoder, nochangedoutputs/branch/state), genuineprimaryHannbox and originalcoordinateclip, selectedgain/cost proper, STOP bitcheck, CSV contracts evidence complete/sealed; existing completeC/admission/devfreeze/source/nativeguards unchanged. Verify runtime admission plain-vs-observed same frozenB6250 uses realTRAIN and coversbothSTOP/QUERY; noactiveweight/pause/FPS. Check unchanged trainingplans/source and no current model-specificevaluationplans. No NN/remotequeries. Return exact SOURCE_PASS/FAIL blockerinteger plus separate guard payloads: admissionguard plan_sha256/source_sha256 and each three protocolguard samebinding. Save source JSON/MD and payloads under .aris/traces/experiment-audit/2026-10-27_query_evidence_capture/source_review/. RequestedAstra max, actualmodelUNATTESTED,samefamily/provisional. Report issues directly; do not modify implementation or liveprotocols/guards.',
    followup_messages=['One small admission-only metadata update while review runs: check_query_evidence_capture.py result now includes plan_sha256 and checkpoint_sha256; admission plan resealed final SHA377a5f449481e876c737b01a91e197befdbfc7ef655fbb5e0f317399b3257ba2. Protocol/helper/evaluator unchanged. Please bind final bytes. Added deterministic .aris/traces/experiment-audit/2026-10-27_query_evidence_capture/protected_training_sources.json (57 unique training-source hashes, changedpaths disjoint), inspect if useful. No other implementation edits planned pending your findings.']))

save('run.meta.json', dict(skill='experiment-audit', purpose='query-evidence-capture-source-review',
    generated_at=now, project_dir=str(ROOT), requested_model='gpt-6-astra',
    requested_reasoning_effort='max', actual_model='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional'))
save('001-source-review.meta.json', dict(call_number=1, purpose='source-review',
    agent_task_name=common['agent_task_name'], generated_at=now,
    requested_model='gpt-6-astra', requested_reasoning_effort='max', actual_model='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional', status='ok',
    response='SOURCE_REVIEW.md', machine_response='SOURCE_REVIEW.json'))
print(json.dumps(dict(status='SOURCE_PASS', blocking_issues=0,
    admission_plan_sha256=sha(ROOT / PLAN), protocols=review['protocol_bindings'],
    output_directory=str(OUT), guard_payloads=[p.name for p in (OUT/'guard_payloads').glob('*.json')]), indent=2))
