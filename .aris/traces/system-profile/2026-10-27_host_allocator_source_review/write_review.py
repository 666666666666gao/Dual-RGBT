import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
PROFILE = 'reports/HOST_ALLOCATOR_PROFILE_PLAN_20261027.json'
JSON_PATH = 'reports/HOST_ALLOCATOR_PROFILE_SOURCE_REVIEW_20261027.json'
MD_PATH = 'reports/HOST_ALLOCATOR_PROFILE_SOURCE_REVIEW_20261027.md'
AGENT = '/root/review_host_allocator_profile_20261027'


def read(name):
    return json.loads((ROOT / name).read_bytes())


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


plan = read(PROFILE)
verification = json.loads((TRACE / 'cpu_verification.json').read_bytes())
assert sha(PROFILE) == verification['profile_sha256']
assert all(sha(name) == expected for name, expected in plan['source_sha256'].items())
now = datetime.now(timezone.utc).isoformat()
inputs = list(plan['source_sha256']) + [PROFILE,
    'reports/host_allocator_default_plan_20261027.json',
    'reports/host_allocator_compact_plan_20261027.json',
    'reports/VRAM18_PRE_EXPANSION_BASELINE_20261027.json',
    'reports/VRAM18_RAM_AND_DISK_CAPACITY_20261027.json',
    'reports/VRAM18_HOST_MEMORY_BREAKDOWN_20261027.json',
    'reports/concurrent_joint_quality_k4_cost0_plan_20261026.json',
    'reports/CONCURRENT_JOINT_QUALITY_K4_COST0_SOURCE_REVIEW_20261026.json',
    'reports/CONCURRENT_TRAIN_GRID_SOURCE_REVIEW_20261026.json']
for name in inputs:
    target = TRACE / 'inputs' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / name).read_bytes())

blocker = dict(
    id='B1', severity='blocking',
    title='The profile has no bounded remote sampler/owned-child lifecycle',
    evidence=['tools/profile_host_allocator.py:46', 'tools/profile_host_allocator.py:49',
              'tools/profile_host_allocator.py:50', 'tools/profile_host_allocator.py:61',
              'tools/profile_host_allocator.py:67', 'scripts/run_task_four_gpu.py:26',
              'scripts/run_task_four_gpu.py:31'],
    details='The generated remote program detaches its wrapper with start_new_session=True, then samples in while True based only on status.json. It never polls the Popen child, has no remote deadline, and has no cleanup/reap path on sampler failure. timeout=600 is applied to the local ssh subprocess only. Consequently the source does not implement a bounded remote profile: local transport timeout does not terminate the detached profile process group, and an observer read/assertion failure can abandon its owned child. A stale RUNNING record cannot terminate the remote loop. This is a source lifecycle defect, not evidence that a live job has failed.',
    required_change='Give the generated remote sampler an explicit deadline and direct child-liveness handling, and end/reap only its own newly created profile process group on deadline or sampler failure. Preserve an honest failed native/result outcome; do not restart, fall back, or signal any of the original eight tasks. Then rehash/review the changed source and plans before launching.')

checks = {
 'source_and_plan_hashes': dict(status='PASS', details='All 51 current local sources match the profile and both case source maps and parse with ast. The original 50-source map, original review plan/source binding and aggregate-review hash match. Both case plan SHA-256 bindings match. The only added source is tools/profile_host_allocator.py.', evidence=['.aris/traces/system-profile/2026-10-27_host_allocator_source_review/cpu_verification.json']),
 'original_input_binding': dict(status='PASS', details='All 24 case input bindings are identical to the original K4/cost0 plan. Twelve locally present non-weight inputs freshly hash-match. Eight nonlocal inputs and four prohibited-to-read weight inputs match the twelve bindings in the sealed historical CPU receipt; this is inherited provenance, not fresh remote verification. Original B6250 b84d0045..., factory ba49b506..., protected SOT ep14 cbcc96ba... remain unchanged.', evidence=['scripts/train_candidate_joint_quality_trajectory.py:257', 'scripts/train_candidate_joint_quality_trajectory.py:259', 'scripts/train_candidate_joint_quality_trajectory.py:260', 'scripts/train_candidate_joint_quality_trajectory.py:269', 'scripts/train_candidate_joint_quality_trajectory.py:307']),
 'environment_only_pair': dict(status='PASS', details='Default/compact differ only in execution_env and separate run/task/review/native paths. Both preserve seed2027, GPU2, K4, cost0, sequence order882, stride5, original architecture/loss/LRs/optimizer setup and all inputs. The remote child environment removes the three allocator variables in both cases, then applies compact arena2 and mmap/trim thresholds131072. OPENCV_FOR_THREADS_NUM=1 is shared. No fallback/retry/restart path was added.', evidence=['tools/profile_host_allocator.py:42', 'tools/profile_host_allocator.py:43', 'reports/host_allocator_default_plan_20261027.json', 'reports/host_allocator_compact_plan_20261027.json']),
 'ownership_and_original_jobs': dict(status='PASS', details='The eight owner identities match the archived pre-expansion baseline. On the selected GPU2, source requires the recorded original native parent identity, exact original child argv and exact GPU compute PID set before launch, plus MemAvailable>12GiB. It writes unique new profile directories and has no command that kills/restarts/edits the original jobs. This audit did not query current ownership or capacity.', evidence=['tools/profile_host_allocator.py:30', 'tools/profile_host_allocator.py:35', 'tools/profile_host_allocator.py:36', 'tools/profile_host_allocator.py:39', 'tools/profile_host_allocator.py:40', 'tools/profile_host_allocator.py:41']),
 'generated_programs_and_argv': dict(status='PASS', details='Exact default/compact remote programs and the CPU equivalence program were generated from the source and AST/compile checked without execution. Wrapper argv selects GPU2 and the unique sanity task; the inner trainer argv ends in --sanity. Native wrapper PID is compared to the Popen PID, while the trainer independently binds its parent/task/GPU/argv. Training stdout/stderr go to the task log; only the result JSON is printed on the successful SSH stdout path. No generated program was run.', evidence=['tools/profile_host_allocator.py:45', 'tools/profile_host_allocator.py:46', 'tools/profile_host_allocator.py:50', 'tools/profile_host_allocator.py:65', 'tools/profile_host_allocator.py:68', 'scripts/run_task_four_gpu.py:20', 'scripts/run_task_four_gpu.py:24', 'scripts/train_candidate_joint_quality_trajectory.py:263']),
 'real_four_update_native_reload_gate': dict(status='PASS', details='The unchanged trainer causally tracks20 frames from the first TRAIN sequence, performs four real backward/AdamW updates and checks all8 sampled role/family combinations. It requires251 frozen tensors/14 changed and actual optimizer step4, saves a checkpoint and strictly reloads a clone for12-field bit equality. The profile requires native COMPLETE/exit0 and actual child exit0 plus COMPLETE4/20 and SANITY PASS/reload. These are prospective runtime gates; this review has no new M0 result.', evidence=['scripts/train_candidate_joint_quality_trajectory.py:118', 'scripts/train_candidate_joint_quality_trajectory.py:169', 'scripts/train_candidate_joint_quality_trajectory.py:179', 'scripts/train_candidate_joint_quality_trajectory.py:197', 'scripts/train_candidate_joint_quality_trajectory.py:225', 'scripts/train_candidate_joint_quality_trajectory.py:335', 'scripts/train_global_evidence_policy.py:40', 'tools/profile_host_allocator.py:61', 'tools/profile_host_allocator.py:63']),
 'sampling_lifecycle': dict(status='FAIL', blocker='B1', details='Two-second /proc memory/environment sampling is purposeful bounded-profile instrumentation, not LLM progress polling. The interval is appropriate to the stated measurement, but the remote lifecycle is not bounded or guaranteed to reap its newly launched child. See B1.'),
 'equivalence_and_formal_scope': dict(status='PASS', details='The planned post-pair comparison loads both new checkpoints with map_location=cpu, requires exact265 state tensors, recursive optimizer equality, and exact trajectory/query-label CSV bytes. It runs only after both native-zero admissions. The output explicitly limits the evidence to4 updates/20 frames and sampled PSS including reload, not fulltraining/accuracy/FPS/steady-state or continuous peak. No formal run is launched by the profile; a future compact formal launch requires successful paired numerical/memory evidence and separate admission. The UNLAUNCHED formal schema names are not authorization.', evidence=['tools/profile_host_allocator.py:71', 'tools/profile_host_allocator.py:75', 'tools/profile_host_allocator.py:76', 'tools/profile_host_allocator.py:83', 'tools/profile_host_allocator.py:84', 'tools/profile_host_allocator.py:85', 'tools/profile_host_allocator.py:88']),
 'A_ground_truth_provenance': dict(status='PASS', details='Planned training is real_gt: sequence boxes come from the hashed TRAIN manifest; current GT is first supplied after causal tracking. Targets are actual box IoUs against dataset GT, not model outputs used as reference labels. No test split is used by this profile.', evidence=['scripts/train_candidate_joint_quality_trajectory.py:186', 'scripts/train_candidate_joint_quality_trajectory.py:297', 'scripts/train_candidate_evidence_policy.py:57', 'scripts/train_candidate_evidence_policy.py:72']),
 'B_score_normalization': dict(status='PASS', details='No accuracy metric or normalized performance score is produced. Internal score/probability features are unchanged model computation. The memory measurements report raw KiB values; model state/optimizer/CSV comparisons use exact equality.', evidence=['tools/profile_host_allocator.py:59', 'tools/profile_host_allocator.py:65', 'tools/profile_host_allocator.py:77', 'dual_rgbt_global/candidate_policy.py:51']),
 'C_result_existence': dict(status='PASS', details='Plans label the profile as prospective and do not claim completed admissions or new accuracy/FPS. Archived resource files exist and their hashes are recorded, but their live values were not rechecked. No profile-result or weight file was opened in this source audit.', evidence=['reports/HOST_ALLOCATOR_PROFILE_PLAN_20261027.json', 'reports/VRAM18_PRE_EXPANSION_BASELINE_20261027.json', 'reports/VRAM18_HOST_MEMORY_BREAKDOWN_20261027.json']),
 'D_dead_code': dict(status='PASS', details='Both case samplers and the equality code are on the main path; actual training/optimizer/reload functions are called from the sanity path. No unused metric is presented as a result.', evidence=['tools/profile_host_allocator.py:27', 'tools/profile_host_allocator.py:67', 'tools/profile_host_allocator.py:87', 'scripts/train_candidate_joint_quality_trajectory.py:332']),
 'E_scope': dict(status='PASS', details='The claimed scope is two sequential short admissions, one seed, one TRAIN sequence,4 updates/20 frames per case. Sampled memory includes a reload clone and is not claimed as formal steady-state, peak allocation, throughput or accuracy. The source lifecycle requirement fails separately under B1.'),
 'F_evaluation_type': dict(status='PASS', classification='real_gt_planned_training_plus_process_memory_instrumentation', details='Not a benchmark accuracy evaluation. No neural execution occurred during this review.')
}

report = dict(audit_skill='experiment-audit', status='SOURCE_FAIL', verdict='FAIL',
    blocking_issues=1, blockers=[blocker], generated_at=now, date='2026-10-27',
    agent_id=AGENT, verdict_id=AGENT+':host-allocator-source:1',
    reviewer_model_requested='gpt-6-astra', reviewer_reasoning_requested='max',
    reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED',
    actual_route_attestation='UNATTESTED', reviewer_family='openai',
    review_independence='same-family', acceptance_status='provisional',
    profile_plan_path=PROFILE, profile_plan_sha256=sha(PROFILE),
    source_sha256=plan['source_sha256'], case_plan_sha256={c['plan']:sha(c['plan']) for c in plan['cases']},
    audited_input_hashes={name:sha(name) for name in inputs},
    checks=checks,
    deterministic_checks=dict(source_hash_count=51, source_AST_count=51,
        exact_generated_program_compile_count=3, local_nonweight_input_hash_count=12,
        inherited_only_input_bindings=12, freshly_read_weight_files=0,
        exact_case_argv_checks=2, owners_match_archived_baseline=True,
        CPU_equality_executed=False, neural_execution=False,
        verification_sha256=sha(str((TRACE/'cpu_verification.json').relative_to(ROOT))),
        generated_program_verification_sha256=sha(str((TRACE/'generated_program_verification.json').relative_to(ROOT)))),
    runtime_admission=dict(status='BLOCKED_SOURCE_LIFECYCLE', paired_M0='NOT_EXECUTED_BY_REVIEW',
        current_ownership='NOT_QUERIED', compact_formal='NOT_ADMITTED',
        per_case_PASS_issued=False),
    operations=dict(source_edits=False, plan_edits=False, remote_queries=False,
        SSH_queries=False, active_job_queries=False, log_queries=False,
        live_PID_queries=False, weight_reads=False, GPU_queries=False,
        torch_import=False, neural_execution=False, subprocess_fixture_execution=False),
    trace_path=str(TRACE.relative_to(ROOT)).replace('\\','/'))
write(ROOT/JSON_PATH, report)

markdown = '''# Host allocator profile source review

Status: **SOURCE_FAIL**. One blocking issue. No per-case SOURCE_PASS reports are issued.

This is a fresh source-only experiment-audit review. Requested route: gpt-6-astra / max; actual model and effort: UNATTESTED. Review independence: same-family; acceptance: provisional. No active task, log, PID, weight, GPU or SSH query and no neural execution occurred.

## Blocking issue B1: remote profile lifecycle is not bounded

`tools/profile_host_allocator.py:46` creates the new wrapper with `start_new_session=True`. Lines49-60 loop indefinitely on its status file, with no remote deadline or direct child-liveness test. Line61 only waits after the file reports a terminal state. The600-second timeout at line67 bounds the local SSH call; it does not terminate/reap the detached remote profile group.

Therefore the source does not implement the requested bounded short profile. A stale RUNNING record cannot end the loop; an observer read/assertion failure or local transport timeout can leave the new profile child unmanaged. This is a source finding, not a claim that any live job has failed. The evidence is the actual generated control flow, not a speculative request for a fallback.

Required change: give the generated sampler an explicit deadline and direct child-liveness handling, and terminate/reap only its own newly created profile process group on deadline or sampling failure. Record an honest failed outcome. Add no restart/fallback, and do not signal or edit the original eight tasks. Rehash and re-review the revised source/plans before launch.

## Passing source checks

- All51 current source hashes match; all51 source files parse. Both case plan hashes match. The original50-source review and aggregate-review hash remain bound.
- Both cases preserve all24 original input bindings and the exact882-sequence order. Twelve local non-weight inputs freshly hash-match. The other12 bindings match a sealed historical CPU receipt; no remote input or weight was freshly read.
- Default and compact differ only in the allocator environment and unique run/task/review paths. Seed2027, GPU2, SOTep14, originalB6250, K4, cost0, losses, learning rates and architecture remain unchanged.
- The archived eight owner records match the pre-expansion baseline. The generated program requires exact GPU2 owner child argv/native PID and compute-PID set, then launches only a unique sanity task. No original task is modified or preempted.
- Exact generated default/compact/equality programs compile without execution. Their argv correctly reaches the wrapper and trainer with `--sanity`; native parent/task/GPU/argv binding is checked. Successful stdout contains a single JSON result while training output goes to its own log.
- The real trainer performs4 optimizer updates/20 causal track frames, checks251 frozen/14 changed tensors and step counts, and strictly reloads12 prediction fields. The sampler requires native COMPLETE/exit0 and actual child exit0. These gates are source checks, not new runtime results.
- Planned CPU comparison requires exact265 model tensors, optimizer equality and both CSV files. It runs only after both successful admissions. No formal run is launched; future compact formal use remains dependent on actual paired evidence and separate admission.
- Two-second sampling is purposeful instrumented measurement, not LLM polling. PSS is correctly described as sampled process memory including reload, without continuous-peak or steady-state claims.

## Integrity checklist A-F

A. PASS: planned training uses dataset TRAIN boxes and IoU-based labels after causal tracking (`train_candidate_joint_quality_trajectory.py:186,297`; `train_candidate_evidence_policy.py:57,72`).

B. PASS: no performance-score normalization or new accuracy metric; memory is rawKiB and equivalence is exact equality (`profile_host_allocator.py:59,77`).

C. PASS: the plans make prospective claims; no new profile result is asserted or opened. Existing archived capacity reports are recorded as historical.

D. PASS: prescribed sampling, training, optimizer, reload and equality paths are called, rather than unused functions reported as results.

E. PASS: scope is two sequential short admissions, one seed and one TRAIN sequence,4 updates/20 frames each. No fulltraining/accuracy/FPS/throughput improvement is claimed. B1 separately blocks the bounded-lifecycle requirement.

F. PASS: classification is planned `real_gt` training plus process-memory instrumentation, not benchmark accuracy evaluation.

## Evidence and limits

The JSON report contains the complete source hash map, exact plan hashes, line references and operations declaration. Trace files retain frozen source/plan inputs, generated programs, deterministic verification and this response. OriginalB and SOT hashes were checked as unchanged bindings against sealed receipts, not reread from weights. Current ownership/capacity, both actual admissions, CPU numerical equality and formal compact use remain unverified/not admitted.
'''
(ROOT/MD_PATH).write_text(markdown,encoding='utf-8')
(TRACE/'001-host-allocator-source.response.md').write_text(markdown,encoding='utf-8')
prompt = '''Fresh source integrity audit per C:/Users/gb/.codex/skills/experiment-audit/SKILL.md. Root C:/Users/gb/projects/Dual-RGBT. Read listed files directly, adversarially; no implementation edits. Files: tools/profile_host_allocator.py; reports/HOST_ALLOCATOR_PROFILE_PLAN_20261027.json; reports/host_allocator_default_plan_20261027.json; reports/host_allocator_compact_plan_20261027.json; reports/VRAM18_PRE_EXPANSION_BASELINE_20261027.json; reports/VRAM18_RAM_AND_DISK_CAPACITY_20261027.json; reports/VRAM18_HOST_MEMORY_BREAKDOWN_20261027.json; original reports/concurrent_joint_quality_k4_cost0_plan_20261026.json and its source_review; scripts/train_candidate_joint_quality_trajectory.py; scripts/run_task_four_gpu.py; scripts/complete_pipeline.py. Follow current profile source-map references and verify all local hashes and original-input binding. Audit no speculative fallback/restart, actual ownership/no preemption, originalB/SOTep14/cost0/K4/seed2027/full real4update+20track/reload/native0 proof, environment-only host allocator comparison, no fake accuracy/fulltraining/FPS, source checks/CPU equivalence before any formal compact use. Inspect generated remote programs and command argv/PID handling, JSON output behavior, sampling lifecycle, original eight jobs untouched. Automated2-second metadata sampling inside one bounded short memory profile is purposeful instrumentation, not LLM polling. NO active-job/log/PID/weight/GPU/SSH query or NN execution. Scope source only. Write reports/HOST_ALLOCATOR_PROFILE_SOURCE_REVIEW_20261027.json and .md with status SOURCE_PASS or SOURCE_FAIL, blocking_issues, profile_plan_sha256, source_sha256 map, source review attribution requested Astra/max actual UNATTESTED if unexposed, same-family/provisional; preserve traces .aris/traces/system-profile/2026-10-27_host_allocator_source_review. If PASS write per-case reports/HOST_ALLOCATOR_DEFAULT_SOURCE_REVIEW_20261027.json and HOST_ALLOCATOR_COMPACT_SOURCE_REVIEW_20261027.json with status SOURCE_PASS/0, plan_sha256 and exact case source_sha256, aggregate review SHA. If FAIL do not issue casePASS. Return findings/final hashes, no NN/result claims; do not edit source.'''
write(TRACE/'run.meta.json',dict(skill='experiment-audit',run_id='2026-10-27_host_allocator_source_review',
    generated_at=now,executor='codex',executor_model='UNATTESTED',executor_family='openai',
    review_independence='same-family',acceptance_status='provisional',project_dir=str(ROOT)))
write(TRACE/'001-host-allocator-source.request.json',dict(call_number=1,purpose='host-allocator-source',
    timestamp=now,tool='spawn_agent',model='gpt-6-astra',reasoning_effort='max',
    actual_route_attestation='UNATTESTED',files_referenced=inputs,prompt=prompt))
write(TRACE/'001-host-allocator-source.meta.json',dict(call_number=1,purpose='host-allocator-source',
    timestamp=now,agent_id=AGENT,model='UNATTESTED',reasoning_effort='UNATTESTED',
    model_requested='gpt-6-astra',reasoning_effort_requested='max',reviewer_family='openai',
    review_independence='same-family',acceptance_status='provisional',status='ok',
    verdict='SOURCE_FAIL',blocking_issues=1,json_report_sha256=sha(JSON_PATH),markdown_report_sha256=sha(MD_PATH)))
event_path=ROOT/'.aris/meta/events.jsonl';event_path.parent.mkdir(parents=True,exist_ok=True)
with event_path.open('a',encoding='utf-8') as stream:
    stream.write(json.dumps(dict(event='review_trace',skill='experiment-audit',purpose='host-allocator-source',
        agent_id=AGENT,trace_path=report['trace_path'],status='ok',verdict='SOURCE_FAIL'))+'\n')
print(json.dumps({'status':report['status'],'blocking_issues':1,'profile_plan_sha256':sha(PROFILE),
    'profile_source_sha256':sha('tools/profile_host_allocator.py'),
    'report_json_sha256':sha(JSON_PATH),'report_md_sha256':sha(MD_PATH),
    'case_plan_sha256':report['case_plan_sha256'],'case_PASS_issued':False},indent=2))
