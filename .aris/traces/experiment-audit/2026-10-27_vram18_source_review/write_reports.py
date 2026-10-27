"""Write the fresh review's conclusions; never alter implementation or plans."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('C:/Users/gb/projects/Dual-RGBT')
TRACE='.aris/traces/experiment-audit/2026-10-27_vram18_source_review'
out=ROOT/TRACE
e=json.loads((out/'deterministic_checks.json').read_bytes())
index=json.loads((ROOT/'reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json').read_bytes())
spec=json.loads((ROOT/'reports/VRAM18_EVALUATION_PROTOCOL_20261027.json').read_bytes())
def sha(name):return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
def write(name,value):
    path=ROOT/name
    assert not path.exists(),name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
assert all(sha(n)==h for n,h in e['source_sha256'].items())
assert all(sha(n)==h for n,h in spec['training_plan_sha256'].items())
assert sha('reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json')==e['index_sha256']
assert sha('reports/VRAM18_EVALUATION_PROTOCOL_20261027.json')==e['protocol_sha256']
assert e['grid_exact'] and e['unique_configuration_count']==32 and not e['source_mismatches']
assert all(v['sha256']==v['registered_sha256'] and v['metadata_matches'] and v['objective_matches'] and not v['unexpected_differences'] and v['original_inputs_exact'] and v['sequence_order_exact'] and not v['source_removed'] and not v['source_changed'] for v in e['plan_checks'])
assert all(v['expected']==v['actual'] and v['retained'] for v in e['original_plan_checks'].values())
assert all(v['expected']==v['actual'] for v in e['profile_source_checks'].values())
assert all(v['expected']==v['actual'] for v in e['profile_case_plan_checks'].values())
assert all(v['status']!='FAIL' for v in e['input_checks'].values())
now=datetime.now(timezone.utc).isoformat()
attribution=dict(audit_skill='experiment-audit',review_round='R1',reviewer_agent='/root/review_vram18_training_grid_20261027',
    requested_reviewer_model='gpt-6-astra',requested_reviewer_reasoning_effort='max',
    actual_reviewer_model='UNATTESTED',actual_reviewer_reasoning_effort='UNATTESTED',
    reviewer_model='UNATTESTED',reviewer_reasoning='UNATTESTED',reviewer_family='openai',
    review_independence='same-family',acceptance_status='provisional',
    attribution_note='Requested Astra/max is not model/effort attestation. This host did not expose authenticated runtime model/effort to this reviewer.',
    generated_at=now,trace_path=TRACE,source_only=True,
    prohibited_operations_executed=False,
    review_scope='Local source/JSON/sha256/AST only. No SSH, neural-network execution, GPU query, active log/PID/weight/environment inspection, training or accuracy certification.')

blockers=[
 dict(id='T1',priority='P1',title='Later waves are admitted without actual throughput, original-task slowdown, or swap evidence',
      evidence=['tools/launch_vram18_training_wave.py:33','tools/launch_vram18_training_wave.py:36','tools/launch_vram18_training_wave.py:81','tools/launch_vram18_training_wave.py:111','tools/launch_vram18_training_wave.py:127','reports/VRAM18_RAM_AND_DISK_CAPACITY_20261027.json','reports/VRAM18_HOST_MEMORY_BREAKDOWN_20261027.json'],
      details='Phases 6/8 accept a previous FORMAL_STARTUP_PASS after one startup CSV observation. next_observation_epoch is written but never consumed or enforced. The launcher neither computes delta-update throughput nor compares the original eight jobs against a measured baseline; it checks only MemAvailable and does not examine system/process swap before expansion. The supplied historical capacity records already contain nonzero swap (43,344, 471,876 and 668,000 KiB for three original children), so this is an evidenced capacity concern. A completed four-update allocator profile is not a sustained wave-health measurement.',
      required_change='Before phase 6 or 8, require a sealed observation of the previous wave at its scheduled diagnostic node, with actual aggregate update rate, original-task slowdown, RAM and swap measurements and an explicit admission decision. Preserve the original eight tasks.'),
 dict(id='T2',priority='P1',title='The ownership guard conflates a running native child with CUDA readiness',
      evidence=['tools/launch_vram18_training_wave.py:68','tools/launch_vram18_training_wave.py:74','tools/launch_vram18_training_wave.py:79','tools/launch_vram18_training_wave.py:88','tools/launch_vram18_training_wave.py:90','scripts/run_task_four_gpu.py:24','scripts/run_task_four_gpu.py:28','scripts/train_candidate_joint_quality_trajectory.py:258','scripts/train_candidate_joint_quality_trajectory.py:307'],
      details='The wrapper writes RUNNING before spawning its child. A child then imports code and hashes source/input files before its first .cuda(). The next same-card launch includes every RUNNING child in expected GPU PIDs and requires actual==expected after only a fixed two-second sleep. Phase 4 starts three consecutive sanities on GPU 2 and three on GPU 3, so legitimate CPU initialization can make the next launch abort. The same transition applies to sequential same-card formal launches. Static string expansion reproduces an admissible RUNNING/no-CUDA state rejected by the guard; this is a source state-machine counterexample, not an observed runtime failure.',
      required_change='Represent registered-but-initializing children separately from GPU-ready owners. Require all known ready children to be present and every actual CUDA process to belong to registered children. Limit slots by registered live children, including initializing children. Do not weaken foreign-process rejection or restart any job.'),
 dict(id='T3',priority='P2',title='Partial sanity launches are not durably represented in the wave state',
      evidence=['tools/launch_vram18_training_wave.py:40','tools/launch_vram18_training_wave.py:55','tools/launch_vram18_training_wave.py:89','tools/launch_vram18_training_wave.py:91','tools/launch_vram18_training_wave.py:106','tools/launch_vram18_training_wave.py:109'],
      details='The initial report is saved before launching, but each sanity PID/command remains only in memory until all eight launch calls succeed. A T2 guard failure during the second same-card sanity leaves a launched child absent from the durable cases and the report still SOURCE_ADMITTED_HOST_PROFILE_PASSED. The report then prevents re-entry with assert not report_path.exists(). Formal receipts are written per case, but the state label remains SANITIES_LAUNCHED until the entire loop completes. This does not stop the old eight, but makes the new wave lifecycle incomplete and unreviewable after a partial launch.',
      required_change='Persist each returned sanity/formal launch receipt immediately, with explicit partial-launch/admission/formal transition states. Preserve existing identities and do not add automatic retries, fallbacks, or duplicate launches.')
]
artifact_paths=['reports/VRAM18_TRAINING_GRID_20261027.json','reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json','reports/VRAM18_EVALUATION_PROTOCOL_20261027.json','reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json','reports/HOST_ALLOCATOR_PROFILE_PLAN_20261027.json','reports/host_allocator_default_plan_20261027.json','reports/host_allocator_compact_plan_20261027.json','reports/VRAM18_PRE_EXPANSION_BASELINE_20261027.json','reports/VRAM18_RAM_AND_DISK_CAPACITY_20261027.json','reports/VRAM18_HOST_MEMORY_BREAKDOWN_20261027.json','reports/FOUR_GPU_EIGHT_TRAIN_SIX_HOUR_NODE_PLAN_20261026.json']
artifacts={name:sha(name) for name in artifact_paths}
runtime_required=[
 'The actual paired allocator result was not inspected and is not certified. Before any new formal training, it must demonstrate native exit 0 for default and compact, bit-equal 265 model-state tensors and complete optimizer state, byte-equal trajectory/query-label CSVs, and strictly lower compact sampled peak PSS.',
 'The compact allocator environment is identical across all new plans and the registered profile case; four-step equivalence does not establish full-training equivalence or formal steady-state PSS.',
 'All local-unavailable inputs and all weight hashes must be checked by the original trainers at runtime. This audit intentionally did not read weights or remote inputs.',
 'Each new configuration must pass its own real four-update/20-frame/one-initialize sanity, native exit 0, 251 frozen plus 14 changed tensors, strict reload, selected execution replay and preserved teacher/tracker state before formal reset to original B6250.',
 'All native ownership, RAM/GPU capacity, throughput and slowdown claims remain runtime-gated. No formal job or evaluation is launched by this review.'
]
checks=dict(
 registration=dict(status='PASS',details='Exactly 32 unique configurations = 2 objectives x K4/K8 x eight costs. 24 new plans, original eight exact hashes retained, 64 distinct run names and task names; phase counts yield 4/6/8 per card.'),
 binding=dict(status='PASS',source_files=57,source_mismatches=0,original_plan_hashes_retained=8,new_plan_hashes_verified=24,input_hash_status_counts=dict(Counter(v['status'] for v in e['input_checks'].values()))),
 training_recipe=dict(status='PASS',details='All 24 plans match objective-specific original K4/cost0 controls except declared configuration, identities, allocator environment, source additions and descriptive bindings. Seed2027, 882 sequence order, two rounds, stride5, LR1e-4/preview1e-5, weight decay1e-4, original B6250 and protected SOT ep14 bindings unchanged; full 182182 updates/917106 tracked frames/1764 initializes. Original trainers and all original source hashes unchanged.'),
 source_gates=dict(status='PASS',details='Native/plan/source/input checks precede NN in each trainer; sanity admission and frozen/teacher/replay/reload checks precede formal. Formal loads original candidate_b6250, never sanity weights. No source kill/preemption/retry/fallback path in the new launcher.'),
 generated_programs=dict(status='PASS',details='Expanded and AST-parsed 96 generated remote programs; checked all 48 sanity/formal argv vectors without executing project imports or remote code.'),
 host_allocator_recipe=dict(status='PASS_RUNTIME_REQUIRED',details='Both profile plans/source maps match their registered hashes and differ only in allocator environment and identities. Profile source compares 265 state tensors, recursive optimizer state and both CSV bytes after native0. Launcher requires equality flags and reduced sampled peak PSS before any launch; actual result remains unverified.'),
 wave_lifecycle=dict(status='FAIL',blocker_ids=['T1','T2','T3']),
 no_new_results=dict(status='PASS',details='Registration is PLANNED/REGISTERED_SOURCE_REVIEW_PENDING. No new training, accuracy, speed, memory success, or model completion is asserted by this audit.'))
training=dict(**attribution,status='SOURCE_FAIL',blocking_issues=len(blockers),blockers=blockers,
    index_sha256=e['index_sha256'],launcher_sha256=e['launcher_sha256'],registration_script_sha256=e['register_sha256'],
    source_sha256=e['source_sha256'],source_count=57,training_plan_sha256=spec['training_plan_sha256'],
    source_map_scope='Exact union of the 32 training-plan maps and extended evaluation source map (57). The separately audited allocator helper is supplementary.',
    supplementary_source_sha256={'tools/profile_host_allocator.py':sha('tools/profile_host_allocator.py')},
    audited_artifact_sha256=artifacts,input_sha256=e['input_bindings'],input_verification=e['input_checks'],
    plan_checks=e['plan_checks'],checks=checks,runtime_requirements=runtime_required,
    case_source_pass_files_issued=0,next_action='Fix T1-T3 in the new orchestration only, reseal affected registrations, and obtain a fresh source review before any new formal launch.')
write('reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json',training)

eval_checks=dict(
 protocol_extension=dict(status='PASS',details='Only source_review, source_sha256, training_plan_sha256 and scope differ. Every original protocol field and original eight plan binding is retained; 32 exact plans and 57 exact source hashes verified.'),
 gt_provenance=dict(status='PASS',evaluation_type='real_gt',evidence=['scripts/benchmark_inputs.py:102','scripts/run_joint_candidate_evaluation.py:104','scripts/run_joint_candidate_evaluation.py:131','tools/joint_candidate_results.py:91'],details='GT comes from sealed dataset manifests, with official toolkit consistency checks. Only the initial box reaches initialization; per-frame GT is not used to select online actions.'),
 score_normalization=dict(status='PASS',evidence=['tools/joint_candidate_results.py:45','tools/joint_candidate_results.py:65','scripts/joint_evaluation_contract.py:254'],details='Supplied toolkit computes precision/success curves. Reports use conventional threshold/curve means times100; no own-prediction-statistic normalization.'),
 training_completion=dict(status='PASS_RUNTIME_REQUIRED',evidence=['scripts/joint_evaluation_contract.py:32','scripts/joint_evaluation_contract.py:68','tools/joint_candidate_results.py:16'],details='Preparation checks terminal native0 and full882x2/182182/917106/1764 metadata before final checkpoint, then CPU witness265 finite tensors/exact251 frozen+14 changed against originalB6250.'),
 dev_freeze=dict(status='PASS_RUNTIME_REQUIRED',evidence=['scripts/joint_evaluation_contract.py:124','scripts/joint_evaluation_contract.py:240','tools/joint_candidate_results.py:124','scripts/joint_evaluation_contract.py:266'],details='Requires actual11-frame train smoke+reload, all97dev/54646frames at costs0/.05/.15, fresh DEV_RESULT_AUDIT_PASS/0 binding exact plan/checkpoint/three report SHA maps, and strict SR>56.43314371195134. Failing candidates receive no test.'),
 test_uniformity=dict(status='PASS_RUNTIME_REQUIRED',evidence=['scripts/joint_evaluation_contract.py:147','scripts/joint_evaluation_contract.py:294','scripts/run_joint_candidate_evaluation.py:62'],details='One frozen checkpoint/config across LasHeR245/220703, RGBT210210/104706, RGBT234234/116650; exact shard/frame coverage and native0 required.'),
 cost_reporting=dict(status='PASS_RUNTIME_REQUIRED',evidence=['scripts/run_joint_candidate_evaluation.py:77','scripts/run_joint_candidate_evaluation.py:103','scripts/run_joint_candidate_evaluation.py:115','dual_rgbt_global/candidate_tracker.py:21','dual_rgbt_global/candidate_tracker.py:65','tools/joint_candidate_results.py:110'],details='Actual total model parameters103218887, GPU/input dimensions, decode+initialize-inclusive summed frame-service FPS, mean/P95 latency; excludes load/writes/verification, and explicitly rejects parallel aggregate throughput or unpaired baseline speedup claims.'),
 result_existence_and_scope=dict(status='PASS',details='This is a protocol/source review, not a result audit. No new result is claimed. Actual weights, remote manifests, runtime receipts and metric files were not read; their runtime gates remain mandatory.'),
 dead_code=dict(status='PASS',details='The prepare/score/freeze/commands dispatcher and evaluation runner invoke the reviewed metadata, tensor witness, full-coverage, official-metric and frozen-selection paths.'))
evaluation=dict(**attribution,status='SOURCE_PASS',blocking_issues=0,blockers=[],protocol_sha256=e['protocol_sha256'],
    source_sha256=spec['source_sha256'],source_count=57,original_protocol_sha256=e['original_protocol_sha256'],
    training_plan_sha256=spec['training_plan_sha256'],audited_artifact_sha256=artifacts,input_verification=e['input_checks'],checks=eval_checks,
    training_admission='NOT_GRANTED: VRAM18_TRAINING_SOURCE_REVIEW_20261027 is SOURCE_FAIL/T1-T3. This independent evaluation recipe verdict does not override it.',
    runtime_requirements=runtime_required+['Evaluation still requires actual complete training, smoke, full dev results, fresh result audit and frozen selection. No runtime, model-quality or FPS certification.'])
write('reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.json',evaluation)

training_md='''# VRAM18 training source review — R1

Status: **SOURCE_FAIL — 3 blockers**. Fresh same-family review; acceptance provisional. Requested Astra/max; actual model and reasoning effort **UNATTESTED**.

Only local source, JSON, hashes and AST were inspected. No SSH, NN/GPU execution, active log/PID/weight/environment query or implementation edit occurred.

'''
for b in blockers:
    training_md+='## '+b['id']+' — '+b['title']+'\n\n'+b['details']+'\n\nEvidence: '+', '.join('`'+v+'`' for v in b['evidence'])+'.\n\nRequired change: '+b['required_change']+'\n\n'
training_md+='''## Verified source properties

- 57 source hashes match; all 32 configuration identities are unique, with 24 new cases and eight original plan hashes unchanged. All original input maps and sequence order are retained. Declared phase placement yields 4→6→8 tasks per card.
- The objective-specific training code, architecture/layers, seed2027, learning rates, batch behavior, original B6250/SOTep14 initialization and full882×2/182182 updates/917106 tracked frames/1764 initializes remain unchanged.
- Native0, actual4-update sanity, selected replay, teacher/tracker-state equality, 251 frozen+14 changed tensors and strict reload precede formal reset to original B. No kill, preemption, fallback or retry path is added.
- All 96 generated remote programs parse; all 48 training argv vectors match their plans. The source-only STARTING-state counterexample is saved in the trace and is not a runtime experiment.
- The allocator recipe compares all265 state tensors, recursive optimizer state and both CSV byte streams; the launcher requires native0, equality flags and lower compact sampled peak PSS. Actual allocator results remain unverified and mandatory before any new formal run; four steps do not establish formal steady-state memory.

No per-case SOURCE_PASS files were issued. The original eight tasks were not queried, changed, abandoned or restarted.

## Sealed inputs

'''
for name,key in [('index','index_sha256'),('launcher','launcher_sha256'),('registration','register_sha256'),('extended evaluation protocol','protocol_sha256')]:training_md+='- '+name+': `'+e[key]+'`\n'
training_md+='\nFull57-source union, all32 plan hashes, all available non-weight input hashes and limitations are in the JSON report. Trace: `'+TRACE+'`.\n'
(ROOT/'reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.md').write_text(training_md,encoding='utf-8')
eval_md='''# VRAM18 evaluation source review — R1

Status: **SOURCE_PASS — 0 evaluation-source blockers**. Fresh same-family review; acceptance provisional. Requested Astra/max; actual model and reasoning effort **UNATTESTED**.

The extension retains the original protocol's full evaluation contract. Only the source map, registered plan map, review path and scope text differ. All57 source hashes and all32 plan hashes match; the original eight plan bindings are unchanged.

Dataset GT and the supplied toolkit remain authoritative, with no prediction-statistic normalization. The source requires completed full882×2 training/native0 and an exact265-tensor witness; actual11-frame smoke and strict reload; full97dev/54646 frames at costs0/.05/.15; a fresh result audit binding the plan/checkpoint/exact three reports; and strict dev SR>56.43314371195134. Eligible models freeze one checkpoint/config across all three full tests: LasHeR245/220703, RGBT210210/104706, RGBT234234/116650.

The result schema preserves actual total parameters103218887, GPU/input dimensions, decode+initialization-inclusive frame-service FPS and mean/P95 latency. Parallel aggregate throughput and unpaired baseline speedup are excluded. Relevant source: `scripts/joint_evaluation_contract.py:21,32,68,124,147,240,266`; `scripts/run_joint_candidate_evaluation.py:36,77,103,115,143`; `tools/joint_candidate_results.py:16,45,79,124`; `scripts/benchmark_inputs.py:102`; `dual_rgbt_global/candidate_tracker.py:21,65`.

This verdict does not grant training admission: the training source review is SOURCE_FAIL/T1–T3. No runtime/model/accuracy/FPS result is certified, and no per-case training PASS is issued. The actual allocator proof and all runtime ownership/capacity, completion, smoke, dev-audit and test-freeze gates remain mandatory.

Protocol SHA256: `'''+e['protocol_sha256']+'`. Original protocol SHA256: `'+e['original_protocol_sha256']+'`. Exact source and input maps are in the JSON report. Trace: `'+TRACE+'`.\n'
(ROOT/'reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.md').write_text(eval_md,encoding='utf-8')

prompt='''Fresh source integrity audit per C:/Users/gb/.codex/skills/experiment-audit/SKILL.md, project C:/Users/gb/projects/Dual-RGBT. Source only, NO NN/GPU/SSH/active logs/PID/weights/environment query. Read tools/register_vram18_training_plans.py, tools/launch_vram18_training_wave.py, reports/VRAM18_TRAINING_GRID_20261027.json, reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json and ALL24 referenced training plans; reports/VRAM18_EVALUATION_PROTOCOL_20261027.json vs original reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json; original scripts/joint_evaluation_contract.py, scripts/run_joint_candidate_evaluation.py, tools/joint_candidate_results.py, two original joint trainers + fourGPUwrapper; reports/HOST_ALLOCATOR_PROFILE_PLAN_20261027.json (actual result not yet available, must remain runtime-gated); reports/VRAM18_PRE_EXPANSION_BASELINE_20261027.json; VRAM18_RAM_AND_DISK_CAPACITY/VRAM18_HOST_MEMORY_BREAKDOWN corresponding20261027. Verify source maps57 and all original input bindings/local available hashes, generated remote programs/argv, finite exactly32 unique (2objectives×K4/8×8costs), original8 unchanged, no duplicate/seed/LR/layer/batch changes. New24 from originalB6250/SOTep14, full882x2/182182/917106/1764, native0/real4step/reload/frozen/teacher-state proof before formal originalBreset. Staged4→6→8 percard, ownership/native PID/gpu/memory/RAM actual guards, no preemption/no fallback/retry. Hostallocator empirical exactstate/optimizer/CSV proof and reducedPSS is mandatory before any newformal. Check state tracking including concurrent sanities percard and transition to formal; ensure no abandon/restart existing8. Extended eval schema retains full97dev3cost/freshaudit/frozen uniform3tests, totalP/serviceFPS, no new results claimed. Keep simple minimal review; implementation edits forbidden. Write reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json/.md status SOURCE_PASS or SOURCE_FAIL, blockers, index_sha256, launcher_sha256, source_sha256 union. If PASS issue per24 plan source-review files at each plan.source_review with exact plan_sha/source maps and aggregate review SHA. Also write reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.json/.md with status/0 ifpass/protocol_sha256/source_sha256 exact. If fail no casePASS. Attribution requestedAstra/max actualUNATTESTED unlessexposed, same-family/provisional. Trace .aris/traces/experiment-audit/2026-10-27_vram18_source_review. No runtime/model/accuracy certification. Return exactfindings/source hashes.'''
write(TRACE+'/run.meta.json',dict(skill='experiment-audit',run_id='2026-10-27_vram18_source_review',generated_at=now,project_dir=str(ROOT),**{k:attribution[k] for k in ['requested_reviewer_model','requested_reviewer_reasoning_effort','actual_reviewer_model','actual_reviewer_reasoning_effort','review_independence','acceptance_status']}))
write(TRACE+'/001-source-review.request.json',dict(call_number=1,purpose='fresh-source-integrity-review',tool='spawn_agent',requested_model='gpt-6-astra',requested_reasoning_effort='max',files_referenced=artifact_paths+list(e['source_sha256'])+list(spec['training_plan_sha256']),prompt=prompt))
response=training_md+'\n'+eval_md
(out/'001-source-review.response.md').write_text(response,encoding='utf-8')
write(TRACE+'/001-source-review.meta.json',dict(call_number=1,purpose='fresh-source-integrity-review',timestamp=now,agent_id=attribution['reviewer_agent'],status='completed',training_verdict='SOURCE_FAIL',evaluation_verdict='SOURCE_PASS',**{k:attribution[k] for k in ['requested_reviewer_model','requested_reviewer_reasoning_effort','actual_reviewer_model','actual_reviewer_reasoning_effort','review_independence','acceptance_status']}))
event_path=ROOT/'.aris/meta/events.jsonl';event_path.parent.mkdir(parents=True,exist_ok=True)
with event_path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(dict(event='review_trace',skill='experiment-audit',purpose='fresh-source-integrity-review',agent_id=attribution['reviewer_agent'],trace_path=TRACE,status='completed',training_verdict='SOURCE_FAIL',evaluation_verdict='SOURCE_PASS'))+'\n')
report_paths=['reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json','reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.md','reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.json','reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.md']
receipt=dict(report_sha256={p:sha(p) for p in report_paths},index_sha256=e['index_sha256'],launcher_sha256=e['launcher_sha256'],protocol_sha256=e['protocol_sha256'],source_sha256=e['source_sha256'],training_plan_sha256=spec['training_plan_sha256'],training_status='SOURCE_FAIL',training_blockers=3,evaluation_status='SOURCE_PASS',evaluation_blockers=0,case_pass_files_issued=0)
write(TRACE+'/review_receipt.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_sha256','training_plan_sha256']},indent=2))
