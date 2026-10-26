"""Write the requested aggregate review and exact per-plan source admission files."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
index = read(ROOT / 'reports/CONCURRENT_TRAIN_PLAN_INDEX_20261026.json')
cpu = read(TRACE / 'cpu_verification.json')
assert cpu['status'] == 'CPU_SOURCE_GRID_PASS'
now = datetime.now(timezone.utc).isoformat()
agent = '/root/review_joint_quality_train_20261026'
report_path = 'reports/CONCURRENT_TRAIN_GRID_SOURCE_REVIEW_20261026.json'
md_path = 'reports/CONCURRENT_TRAIN_GRID_SOURCE_REVIEW_20261026.md'
route = dict(reviewer_model_requested='gpt-6-astra', reviewer_reasoning_requested='max',
    reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED', actual_route_attestation='UNATTESTED',
    reviewer_family='openai', review_independence='same-family', acceptance_status='provisional')
checks = {
    'bindings_and_plan_delta': dict(status='PASS', details='Six plans preserve all base recipe fields except K8->4 and/or trajectory cost .05->0, GPU routing, unique identities and registration metadata. Quality inherits49 source/23 inputs; gainpair inherits50 source/26 inputs. Each adds the grid input; only GPU2/3 add the new wrapper source. All51 unique local sources hash/AST verified; all locally present inputs freshly hashed and12 remote-only immutable bindings match the sealed prior CPU receipt.', evidence=['.aris/traces/experiment-audit/2026-10-26_concurrent_train_grid_source_review/cpu_verification.json']),
    'wrapper_and_native_routing': dict(status='PASS', details='New wrapper differs from sealed original only in docstring and argparse GPU choices0/1/2/3. Exact-byte isolated CPU executions pass GPU2/exit0 and GPU3/exit7, native parent/argv/environment/terminal status. All six trainers accept their own parameterized physical GPU while requiring archived B native GPU1. Wrapper does not claim or automatically check hardware availability.', evidence=['scripts/run_task_four_gpu.py:14', 'scripts/run_task_four_gpu.py:21', 'scripts/run_task_four_gpu.py:28', 'scripts/train_candidate_joint_quality_trajectory.py:253', 'scripts/train_candidate_joint_quality_trajectory.py:264', 'scripts/train_candidate_joint_quality_trajectory.py:271', 'scripts/train_candidate_joint_gainpair_trajectory.py:250', 'scripts/train_candidate_joint_gainpair_trajectory.py:261', 'scripts/train_candidate_joint_gainpair_trajectory.py:268']),
    'K_and_cost_existing_module_support': dict(status='PASS', details='K bounds greedy peak count; the existing shared955-feature scorer accepts dynamic query count. K4 gives at most10 query rows plus STOP versus K8 at most18 plus STOP; existing weights/shapes/parameters are unchanged. Cost affects the same gain-minus-cost selection in tracker and selected replay. Both role teachers still sample one request each; no new teacher call site or loss change. Different K/cost can intentionally change selected-extra frequency, trajectory and NN work, so fixed frames/updates are not equal realized compute.', evidence=['dual_rgbt_global/candidate_policy.py:24', 'dual_rgbt_global/candidate_policy.py:48', 'dual_rgbt_global/candidate_policy.py:83', 'dual_rgbt_global/candidate_tracker.py:38', 'scripts/train_candidate_joint_quality_trajectory.py:55', 'scripts/train_candidate_joint_gainpair_trajectory.py:55']),
    'causal_real_GT_and_no_test_use': dict(status='PASS', details='Unchanged trainers use manifest train and the same882 names; archived sealed evidence verifies TRAIN/dev disjointness. Current GT is read after causal track, then dataset IoU supplies quality/gain labels; replay uses copied pre-track state. No public test labels or test-based selection added. real_gt planned training; CPU fixtures are process/assertion checks, not performance evidence.', evidence=['scripts/train_candidate_joint_quality_trajectory.py:179', 'scripts/train_candidate_joint_quality_trajectory.py:186', 'scripts/train_candidate_joint_quality_trajectory.py:297', 'scripts/train_candidate_joint_gainpair_trajectory.py:175', 'scripts/train_candidate_joint_gainpair_trajectory.py:182', 'scripts/train_candidate_joint_gainpair_trajectory.py:294']),
    'matched_counts_parent_optimizer_freeze': dict(status='PASS', details='All six preserve original B6250 SHA b84d00454717cae56048753e0fe54c1d6b92cfcd2820c01fa65296508488ccb3, seed2027, TRAIN882x2/917106 tracks/182182 updates/1764 initializations, stride5, preview1e-5 and heads1e-4. Formal independently reloads original B and constructs fresh AdamW. Existing14 adapted/251 frozen layout and runtime assertions unchanged. No plan input/source depends on discarded old M0 weights.', evidence=['scripts/train_candidate_joint_quality_trajectory.py:308', 'scripts/train_candidate_joint_quality_trajectory.py:319', 'scripts/train_candidate_joint_gainpair_trajectory.py:305', 'scripts/train_candidate_joint_gainpair_trajectory.py:316', 'scripts/train_candidate_joint_quality_trajectory.py:122', 'scripts/train_candidate_joint_gainpair_trajectory.py:118']),
    'distinct_M0_and_formal_admission': dict(status='PASS', details='Each plan has unique run/task/review paths, owns its native wrapper identity and requires actual4 updates/native0, exact sanity plan/source/input binding, strict12-field reload, four-request legacy parity, selected replay/state preservation and251 frozen/14 changed before formal. Existing probe retains each plan K and actual replay uses each plan cost. No new-plan M0 receipt was observed in this source review.', evidence=['scripts/train_candidate_joint_quality_trajectory.py:282', 'scripts/train_candidate_joint_quality_trajectory.py:288', 'scripts/train_candidate_joint_quality_trajectory.py:292', 'scripts/train_candidate_joint_gainpair_trajectory.py:279', 'scripts/train_candidate_joint_gainpair_trajectory.py:285', 'scripts/train_candidate_evidence_policy.py:87']),
    'integrity_scope': dict(status='PASS', details='No new metrics, normalization, accuracy/FPS, completed-training, capacity or GPU ownership claims. Grid is finite two objectives x K4/8 x cost0/.05, including two existing unchanged cases and six new cases. The grid and index explicitly retain planned-not-launched and capacity/M0 requirements; no test-driven selection. Existing conditional-mean argument remains limited to its stated common-state/free-prediction assumptions, with no finite-network calibration guarantee.')
}
report = dict(audit_skills=['run-experiment','experiment-audit'], status='SOURCE_PASS', verdict='PASS',
    blocking_issues=0, blockers=[], generated_at=now, date='2026-10-26', agent_id=agent,
    verdict_id=agent+':concurrent-grid-source:1', **route,
    plan_index_path='reports/CONCURRENT_TRAIN_PLAN_INDEX_20261026.json', plan_index_sha256=cpu['plan_index_sha256'],
    grid_path='reports/TRAIN_CONCURRENCY_CANDIDATE_GRID_20261026.json', grid_sha256=cpu['grid_sha256'],
    wrapper_path='scripts/run_task_four_gpu.py', wrapper_sha256=cpu['wrapper_sha256'],
    plan_sha256={entry['path']:entry['sha256'] for entry in index['plans']},
    per_plan_reviews=[entry['review'] for entry in index['plans']], cases=cpu['cases'], checks=checks,
    deterministic_checks=dict(status='PASS', acceptance_status='accepted-for-listed-checks-only',
        unique_local_source_hashes=51, unique_local_AST_parses=51,
        unique_current_local_input_hashes=len(cpu['current_local_input_hashes']),
        remote_only_input_bindings_from_prior_receipt=len(cpu['remote_only_input_bindings_from_prior_sealed_receipt']),
        source_native_environment_and_B_assertion_cases=6, actual_CPU_wrapper_cases=cpu['wrapper_cpu_native_fixtures'],
        loss_gradient_suite_repeated=False),
    runtime_admission=dict(status='PENDING_CAPACITY_OWNERSHIP_AND_PER_PLAN_ACTUAL_M0',
        gpu0_1='Launch only within independently measured beneficial capacity; original formal recipes remain unchanged.',
        gpu2_3='Reserved cases remain pending until current ReID owners actually terminate and the intended device is confirmed free.',
        each_new_plan='Own actual4-update native0 strictreload/frozen/replay admission, then formal reset from original B6250.',
        capacity_checked_by_this_review=False, active_owners_checked_by_this_review=False, new_plan_M0_executed_by_this_review=False,
        hardware_preflight_owner='external launcher'),
    reused_evidence={
        'reports/JOINT_QUALITY_TRAIN_SOURCE_REVIEW_20261026.json':'9651f561a030a3ce800bdc46a230d95cd564b8bf39a8ee733c04663aad7107ac',
        'reports/JOINT_GAINPAIR_TRAIN_SOURCE_REVIEW_20261026.json':'29f167345f4f66c924a5969bf0dd532f8d4768678cf427338803e8bd8b8cabf3',
        '.aris/traces/experiment-audit/2026-10-26_joint_quality_source_review/remote_cpu_stdout.json':cpu['reused_remote_receipt_sha256']},
    fixture_execution_note=cpu['fixture_execution_note'],
    operations=dict(audited_source_edits=False, plan_edits=False, handoff_edits=False, git_mutation=False,
        remote_queries=False, active_job_queries=False, torch_import=False, neural_execution=False, GPU_allocation=False,
        local_isolated_CPU_child_processes_only=True),
    trace_path=TRACE.relative_to(ROOT).as_posix(), verification_sha256=sha(TRACE/'cpu_verification.json'))
with (ROOT/report_path).open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(report,indent=2)+'\n')
md = f'''# Concurrent training grid source review — 2026-10-26

**SOURCE_PASS; blocking issues: 0.** Source admission only. Actual reviewer route UNATTESTED (requested Astra/max); same-family/provisional.

Six plans preserve their admitted quality/gainpair sources and all inherited input bindings. Training changes are limited to K8→4 and/or trajectory cost .05→0. Unique identities and conditional GPU routes are correct. All51 unique source hashes/ASTs passed; current local inputs were hashed and12 remote-only immutable bindings were matched to the prior sealed receipt.

The four-card wrapper differs only in its docstring and GPU choices. Exact-byte CPU fixtures verified GPU2/exit0 and GPU3/exit7, environment, actual parent/argv and terminal records. An initial Windows venv fixture exposed interpreter process indirection; the direct base executable passed. No production source was changed and these are not Linux GPU receipts.

K changes candidate rows, not parameter shapes; cost is applied identically in causal tracker and selected replay. Existing quality/gain objectives, teachers, current-GT-after-track ordering, probe,14 adapted/251 frozen assertions and fresh optimizer remain unchanged. Every formal run starts original B6250 and retains882×2,917106 tracks,182182 updates,1764 initializations and seed2027. Fixed updates do not imply equal realized conditional query compute. Removed old M0 weights are not inputs.

Each plan still requires its own actual4-update/native0/frozen/replay/strict-reload M0. GPU0/1 require measured beneficial capacity; GPU2/3 remain reserved until their current ReID owners finish and devices are confirmed free. Those are external launcher checks and were not performed here. No NN launch, active job query, performance result or test-label use occurred.

Index SHA256: `{cpu['plan_index_sha256']}`. Wrapper SHA256: `{cpu['wrapper_sha256']}`.

Detailed plan hashes, six per-plan admission paths, source evidence and limitations are in `CONCURRENT_TRAIN_GRID_SOURCE_REVIEW_20261026.json`; CPU evidence is `{TRACE.relative_to(ROOT).as_posix()}/cpu_verification.json`.
'''
with (ROOT/md_path).open('x',encoding='utf-8') as stream:
    stream.write(md)
per_paths=[]
for entry in index['plans']:
    plan=read(ROOT/entry['path'])
    assert sha(ROOT/entry['path']) == entry['sha256']
    gate=dict(status='SOURCE_PASS', blocking_issues=0, blockers=[], plan_path=entry['path'],
        plan_sha256=entry['sha256'], source_sha256=plan['source_sha256'], generated_at=now,
        aggregate_review=report_path, aggregate_review_sha256=sha(ROOT/report_path), **route,
        runtime_admission='PENDING measured capacity/ownership and own actual4-update native0 reload/frozen/replay M0; formal resets original B6250.')
    with (ROOT/entry['review']).open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(gate,indent=2)+'\n')
    actual=read(ROOT/entry['review'])
    assert actual['status']=='SOURCE_PASS' and actual['blocking_issues']==0
    assert actual['plan_sha256']==sha(ROOT/entry['path']) and actual['source_sha256']==plan['source_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in actual['source_sha256'].items())
    per_paths.append(entry['review'])
request=dict(task='Source/plan admission of six finite existing-module config plans; no NN launch or active remote queries.',
    plan_index='reports/CONCURRENT_TRAIN_PLAN_INDEX_20261026.json', grid='reports/TRAIN_CONCURRENCY_CANDIDATE_GRID_20261026.json',
    planned_scope='Verify sealed inherited maps, K/cost-only recipe deltas, GPU wrapper/native routing, original B reset, own M0 gates and no test labels.',
    reviewer_model_requested='gpt-6-astra', reviewer_reasoning_requested='max')
(TRACE/'request.json').write_text(json.dumps(request,indent=2)+'\n',encoding='utf-8')
(TRACE/'response.md').write_text(md,encoding='utf-8')
meta=dict(skill='experiment-audit', agent_id=agent, verdict_id=report['verdict_id'], generated_at=now, trace_mode='full',
    status='SOURCE_PASS', blocking_issues=0, **route, report_path=report_path, report_sha256=sha(ROOT/report_path),
    verification_sha256=sha(TRACE/'cpu_verification.json'), operations=report['operations'])
(TRACE/'meta.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
events=[dict(event='request',at=now,artifact='request.json'),dict(event='deterministic_checks',at=now,artifact='cpu_verification.json',status='PASS'),dict(event='review_complete',at=now,artifact=report_path,status='SOURCE_PASS',blocking_issues=0)]
(TRACE/'events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events),encoding='utf-8')
seals={p:sha(ROOT/p) for p in [report_path,md_path]+per_paths}
for name in ['verify_cpu.py','write_review.py','cpu_verification.json','request.json','response.md','meta.json','events.jsonl']:
    seals[(TRACE/name).relative_to(ROOT).as_posix()]=sha(TRACE/name)
(TRACE/'output_seals.json').write_text(json.dumps(dict(file_sha256=seals),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='SOURCE_PASS',blocking_issues=0,report_sha256=sha(ROOT/report_path),markdown_sha256=sha(ROOT/md_path),per_plan_reviews=per_paths),indent=2))
