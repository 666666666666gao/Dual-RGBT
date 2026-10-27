"""Seal only the independent serial controller; preserve all training source guards."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('C:/Users/gb/projects/Dual-RGBT')
TRACE='.aris/traces/experiment-audit/2026-10-27_vram18_source_review'
SUB=TRACE+'/serial_controller_round1'
read=lambda n:json.loads((ROOT/n).read_bytes())
sha=lambda n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
v=read(SUB+'/verification.json');r3=read(TRACE+'/round3/review_receipt.json')
assert v['status']=='PASS'
assert sha('tools/launch_vram18_serial_training_wave.py')==v['launcher_sha256']
assert sha('reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json')==v['index_sha256']
assert all(sha(n)==h for n,h in r3['source_sha256'].items())
assert all(sha(n)==h for n,h in r3['training_plan_sha256'].items())
assert all(sha(n)==h for n,h in r3['case_review_sha256'].items())
assert all(sha(n)==h for n,h in r3['report_sha256'].items())
now=datetime.now(timezone.utc).isoformat()
checks=dict(
 isolated_external_scheduler=dict(status='PASS',evidence=['tools/launch_vram18_serial_training_wave.py:70','tools/launch_vram18_serial_training_wave.py:73','tools/launch_vram18_serial_training_wave.py:121'],details='Only phase6 is accepted. The controller requires its own SOURCE_PASS guard with exact index/controller hashes and records its own source/hash,90-second formal interval and maximum4 new sanities. It is a distinct local resource scheduler, not a model variant or silently substituted V1 trainer source.'),
 preserved_registration=dict(status='PASS',details='Current indexe4, V1 controller1335, registrar3a02,58 training/evaluation sources, all32 plans,24 per-case guards and all R3 aggregate/evaluation reports match their sealed bytes. No model code, plan, weight or environment was edited or queried.'),
 unchanged_preflight=dict(status='PASS',evidence=['tools/launch_vram18_serial_training_wave.py:77','tools/launch_vram18_serial_training_wave.py:94','tools/launch_vram18_serial_training_wave.py:110'],details='AST comparison proves pre-state preflight differs only in phase choice and independent controller review path. Exact TRIM result/review/plan/source/case binding, native0, model/CSV equality, declared optimizer tolerance, sampled PSS reduction, bound prior phase4 health and all per-case source checks remain mandatory. No actual runtime profile/phase4-health artifact was read by this review.'),
 grouping_and_native_admission=dict(status='PASS',evidence=['tools/launch_vram18_serial_training_wave.py:160','tools/launch_vram18_serial_training_wave.py:164','tools/launch_vram18_serial_training_wave.py:166','tools/launch_vram18_serial_training_wave.py:173'],details='Exactly two phase6 cases per GPU are asserted. Two batches launch one sanity per card; after180 seconds each batch must pass all four native0/full4-update/frozen/replay/teacher/reload/config admissions before the next batch. The generated admission AST is exactly V1. No fallback, retry, preemption or task restart is added.'),
 serial_formal_startup=dict(status='PASS_SOURCE_ONLY',evidence=['tools/launch_vram18_serial_training_wave.py:175','tools/launch_vram18_serial_training_wave.py:177','tools/launch_vram18_serial_training_wave.py:180','tools/launch_vram18_serial_training_wave.py:181'],details='One formal launches at a time in GPU0/1/2/3 order, repeated twice. A90-second delay precedes the unchanged V1 startup proof; next launch follows only successful startup and cuda_ready=True. The proof validates native/wrapper/child/argv/environment/config and at least one contiguous finite-loss update. It does not require four formal updates, demonstrate continuing rate over90s, or establish steady-state RAM/throughput.'),
 memory_and_gpu_capacity=dict(status='PASS_SOURCE_ONLY',evidence=['tools/launch_vram18_serial_training_wave.py:139','tools/launch_vram18_serial_training_wave.py:150','tools/launch_vram18_serial_training_wave.py:153','tools/launch_vram18_serial_training_wave.py:158'],details='Launch AST is identical to V1 except cases→active_batch for remaining new slots. Sanity reservations are4,3,2,1 twice; formal reservations are1 each. Current MemAvailable includes resident jobs; unallocated sampled-peak headroom for initializing owners remains reserved across all GPUs. Registered per-card slot limits, warmed-owner/foreign-CUDA exclusion, GPU capacity and >8GiB remaining RAM are unchanged. Four exact RAM-condition fixtures pass. This is not proof that32 jobs fit128GiB or that trimming solved steady-state memory.'),
 durable_lifecycle=dict(status='PASS',evidence=['tools/launch_vram18_serial_training_wave.py:169','tools/launch_vram18_serial_training_wave.py:170','tools/launch_vram18_serial_training_wave.py:179','tools/launch_vram18_serial_training_wave.py:182','tools/launch_vram18_serial_training_wave.py:183'],details='Every successful launch receipt is persisted before another launch, delay or admission. Exact schedule AST with inert I/O passed full lifecycle and three partial-failure fixtures: fourth sanity prelaunch failure retains first3 identities; second-group admission failure retains all8 sanity identities and starts no formal; second-formal startup failure retains2 formal identities/one startup and starts no third formal. No fixture ran subprocesses, waited in wall time, queried runtime or certified model behavior.'),
 templates_and_health=dict(status='PASS',details='32 generated remote programs parse and16 launch argv vectors match the eight phase6 plans. Sanity/startup program ASTs match V1 exactly. measurement and score_health ASTs match V1 exactly, so previously accepted health-policy validation is reused. Same full-training recipe and originalB reset remain in unchanged trainers.'),
 synchronization_scope=dict(status='PASS',evidence=['tools/launch_vram18_serial_training_wave.py:116','tools/launch_vram18_serial_training_wave.py:118','tools/launch_vram18_serial_training_wave.py:121'],details='SCP list remains the sealed V1 source/registrar/index/aggregate and per-case plans/guards. The independent local controller embeds its actual source/hash in the execution receipt; it is not falsely included in existing NN source maps. Publishing the controller source separately is outside this source review execution.')
)
report=dict(status='SOURCE_PASS',blocking_issues=0,blockers=[],audit_skill='experiment-audit',review_round='SERIAL_R1',generated_at=now,
    reviewer_agent='/root/review_vram18_training_grid_20261027',requested_reviewer_model='gpt-6-astra',requested_reviewer_reasoning_effort='max',actual_reviewer_model='UNATTESTED',actual_reviewer_reasoning_effort='UNATTESTED',review_independence='same-family',acceptance_status='provisional',
    index_sha256=v['index_sha256'],launcher_sha256=v['launcher_sha256'],controller_source='tools/launch_vram18_serial_training_wave.py',
    source_sha256={'tools/launch_vram18_serial_training_wave.py':v['launcher_sha256']},
    preserved_v1_launcher_sha256=v['v1_launcher_sha256'],preserved_training_source_sha256=r3['source_sha256'],preserved_training_plan_sha256=r3['training_plan_sha256'],preserved_case_guard_sha256=r3['case_review_sha256'],preserved_r3_report_sha256=r3['report_sha256'],
    allowed_phase=6,maximum_simultaneous_new_sanities=4,maximum_initializing_new_formals=1,formal_startup_delay_seconds=90,
    checks=checks,trace_path=SUB,verification_sha256=sha(SUB+'/verification.json'),source_only=True,runtime_certified=False,
    scope='Independent external phase6 resource scheduler source admission only. No runtime/NN/GPU/active-log/PID/weight/environment query or implementation edit. No32-job RAM solution, steady-state memory, model accuracy, or online-FPS claim.',
    runtime_requirements=['Actual bound TRIM proof and actual bound phase4 ADMIT_NEXT_WAVE health receipt must pass before phase6.',
        'Every new sanity must independently complete native0 and unchanged model/trajectory/reload checks.',
        'Before every formal launch, actual ownership/GPU/RAM capacity must pass; the preceding formal must pass its90-second startup observation.',
        'Original and already launched tasks are preserved on a capacity or startup failure; no automatic restart/fallback is authorized by this guard.',
        'Measured health of the completed24-task wave and complete training/evaluation evidence remain necessary for further capacity or performance claims.'])
json_path='reports/VRAM18_SERIAL_CONTROLLER_SOURCE_REVIEW_20261027.json';md_path='reports/VRAM18_SERIAL_CONTROLLER_SOURCE_REVIEW_20261027.md'
assert not (ROOT/json_path).exists() and not (ROOT/md_path).exists()
(ROOT/json_path).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
md='''# VRAM18 independent serial controller source review

**SOURCE_PASS — 0 blockers.** Same-family/provisional; requested Astra/max, actual model/effort UNATTESTED. No runtime query or implementation edit.

Only phase6 is accepted. The separate controller guard binds indexe4 and controllerc1ef. Sealed V1,58 source maps,32 plans,24 case guards and R3 reports remain unchanged. This is an external resource scheduler, not a new model variant.

The source launches two sanity groups of four, one per GPU, and requires all four native0 admissions before the next group. Formal launches are serial; each PID is persisted immediately, then90 seconds elapse before the unchanged native/child/argv/environment/config/finite-update startup proof. Only then is the next formal launched. Partial failures retain launched identities and do not retry or stop original jobs.

RAM reservations use the current group's remaining new slots (4→3→2→1 for each sanity group;1 per formal), plus existing initializing-job headroom to sampled peak, with >8GiB remaining. Ownership, GPU, source, TRIM proof and bound phase4-health gates are unchanged. No32-job RAM solution or steady-state memory saving is claimed.

The90-second proof permits one finite logged update; it is not a four-update formal proof, sustained throughput measurement or steady-state memory certification. Actual profile/health/training results were not read by this reviewer.

Verification: exact AST comparison against V1;32 generated programs and16 argv; full lifecycle plus three partial-failure fixtures; four RAM-condition fixtures. All fixture states are synthetic source tests. Existing unchanged R2/R3 checks were reused.

Controller SHA256: `'''+v['launcher_sha256']+'`.\nIndex SHA256: `'+v['index_sha256']+'`.\nTrace: `'+SUB+'`.\n'
(ROOT/md_path).write_text(md,encoding='utf-8')
prompt='''Runtime firstwave4 COMPLETE0 via68341:16formal4/card; actual measurement_start in VRAM18_WAVE4_EXECUTION...:48.52GiBRAMfree,newPSS4.365GiB,eachGPU~9.3GiB. Bothpostloadtrim and independentwarm4steptrim releaseonly82/116KiB (HOST_ALLOCATOR_TRIM_STARTUP_MEMORY,HOST_ALLOCATOR_WARM_MEMORY_RELEASE), so no32-RAMsolution claimed; no adopting warmwrapper. Existing16 activefiles/V1controller1335/V1registrar3a02/indexe4/32plans unchanged. To safely reach phase6≈24jobs (32wouldexceed128GB), NEW EXTERNAL orchestration only tools/launch_vram18_serial_training_wave.py SHAc1ef685e700b57d3a7754c4730edb889702abc235ab8b1b20e563eab217fb4b1. Clone V1, same model/plan/argv/admit/startup templates andhealthrules; ONLY phase6, M0 two groups4 (one/card, eachnative0 checked before next), formal startsone atatime90s warmcheck before next; RAM new_slots tiedcurrentgroup notall8 while accountinginflightheadroom. Own controllerSourceReview path separate (VRAM18_SERIAL_CONTROLLER_SOURCE_REVIEW_20261027.json), explicitcontrollerSHA recorded; permodel/sourceguard unchangedR3, no NN code modifications. Profile stillrequiresactualTRIMstatus+boundrawproof alreadycompleted. Review targeted ASTdiff V1 andgeneratedprograms/inflightRAM/lifecycle/group4/1/partiallaunchpersistence/90secchecktruthfulness/currentindexe4 andall24 planhashesunchanged. NO runtimequeries/NN/edits. Write SOURCE_PASS orFAIL JSON/MD at reports/VRAM18_SERIAL_CONTROLLER_SOURCE_REVIEW_20261027.* keysindex_sha256(e4),launcher_sha256(current),blockingissues etc. Sourceonly; oldsource/guardsremain. This is distinct local resource scheduler, not new modelvariant. SourcecopyprocedureunchangedSCPknownfiles, localcontrollercode itself canpublishseparately (not misleadingtrainingSourceMap ofV1). All16active kept; rootonlycallsphase6 ifrealphase4healthpasses. Preservetrace/R1-3.'''
(ROOT/TRACE/'004-source-review.request.json').write_text(json.dumps(dict(call_number=4,purpose='independent-serial-controller',tool='followup_task',timestamp=now,prompt=prompt,files_referenced=['tools/launch_vram18_serial_training_wave.py','tools/launch_vram18_training_wave.py','reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json']+list(r3['training_plan_sha256'])),indent=2)+'\n')
(ROOT/TRACE/'004-source-review.response.md').write_text(md,encoding='utf-8')
(ROOT/TRACE/'004-source-review.meta.json').write_text(json.dumps(dict(call_number=4,purpose='independent-serial-controller',timestamp=now,status='completed',verdict='SOURCE_PASS',requested_model='gpt-6-astra',requested_reasoning_effort='max',actual_model='UNATTESTED',actual_reasoning_effort='UNATTESTED',review_independence='same-family',acceptance_status='provisional'),indent=2)+'\n')
receipt=dict(status='SOURCE_PASS',blocking_issues=0,report_sha256={json_path:sha(json_path),md_path:sha(md_path)},index_sha256=v['index_sha256'],launcher_sha256=v['launcher_sha256'],runtime_certified=False)
(ROOT/SUB/'review_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
with (ROOT/'.aris/meta/events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='review_trace',skill='experiment-audit',purpose='independent-serial-controller',trace_path=SUB,status='completed',verdict='SOURCE_PASS'))+'\n')
assert all(sha(n)==h for n,h in r3['report_sha256'].items())
assert all(sha(n)==h for n,h in r3['case_review_sha256'].items())
print(json.dumps(receipt,indent=2))
