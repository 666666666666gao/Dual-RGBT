"""Publish R2 source guards, retaining the exact R1 review bytes."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('C:/Users/gb/projects/Dual-RGBT')
TRACE='.aris/traces/experiment-audit/2026-10-27_vram18_source_review'
OUT=ROOT/TRACE/'round2'
read=lambda n:json.loads((ROOT/n).read_bytes())
sha=lambda n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
def write(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
e=read(TRACE+'/round2/deterministic_checks.json')
binding=read(TRACE+'/round2/bindings_summary.json')
fixtures=read(TRACE+'/round2/focused_regressions.json')
index=read('reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json')
spec=read('reports/VRAM18_EVALUATION_PROTOCOL_20261027.json')
profile=read(index['host_profile_plan'])
assert all(sha(n)==h for n,h in e['source_sha256'].items())
assert all(sha(n)==h for n,h in spec['training_plan_sha256'].items())
assert sha('reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json')==binding['index_sha256']
assert sha('reports/VRAM18_EVALUATION_PROTOCOL_20261027.json')==binding['protocol_sha256']
assert sha(index['observer_source'])==index['observer_sha256']
assert binding['status']==fixtures['status']=='PASS'

r1=read(TRACE+'/review_receipt.json')
archive=ROOT/TRACE/'round1';archive.mkdir(exist_ok=True)
lineage={}
for name,digest in r1['report_sha256'].items():
    assert sha(name)==digest
    dest=archive/Path(name).name
    if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
    else:dest.write_bytes((ROOT/name).read_bytes())
    lineage[dest.relative_to(ROOT).as_posix()]=digest
training=read('reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json')
evaluation=read('reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.json')
now=datetime.now(timezone.utc).isoformat()
resolutions=[
 dict(id='T1',status='RESOLVED',evidence=['tools/launch_vram18_training_wave.py:15','tools/launch_vram18_training_wave.py:35','tools/launch_vram18_training_wave.py:98','tools/launch_vram18_training_wave.py:194','tools/observe_vram18_training.py:27'],
      details='Startup now captures all same-wave task identities/progress/memory. Health scoring requires >=600 seconds, exact task set, running native wrapper/child identities and unchanged command, positive progress and finite losses. It compares actual aggregate update rate against the sealed original-eight baseline for phase4 and preceding admitted wave thereafter, requires each original task retain >=0.4 of its baseline rate, summed task Swap growth <=65536KiB and current MemAvailable >=8GiB. Phases6/8 require a bound ADMIT_NEXT_WAVE health receipt and the numeric conditions. Scheduling metadata retains the nominal +1200s observation node. Operational state-dependent rates are not claimed as paired speedup or online FPS.'),
 dict(id='T2',status='RESOLVED',evidence=['tools/launch_vram18_training_wave.py:92','tools/launch_vram18_training_wave.py:101','tools/launch_vram18_training_wave.py:132','tools/launch_vram18_training_wave.py:145','tools/launch_vram18_training_wave.py:156','tools/launch_vram18_training_wave.py:174'],
      details='Original/prior-wave warmed tasks are cuda_ready=True; current-wave registrations are False. The exact guard requires warmed PIDs <= actual CUDA PIDs <= registered child PIDs, and caps registered live same-card children. CPU initialization no longer requires premature GPU visibility, while unknown processes, missing warm owners and occupied registered slots fail. No preemption/retry/fallback is introduced.'),
 dict(id='T3',status='RESOLVED',evidence=['tools/launch_vram18_training_wave.py:155','tools/launch_vram18_training_wave.py:157','tools/launch_vram18_training_wave.py:173','tools/launch_vram18_training_wave.py:175','tools/launch_vram18_training_wave.py:194'],
      details='Each sanity and formal launch receipt is written to the wave cases before the next task or delay. The cases retain partial-launch identities even while the stage label is at the preceding coarse boundary. Sanity admissions remain mandatory before formal; all startup evidence and measurement_start precede FORMAL_STARTUP_PASS. No resume/retry or restart of the original eight is added.')]
training.update(status='SOURCE_PASS',blocking_issues=0,blockers=[],review_round='R2',generated_at=now,
    index_sha256=binding['index_sha256'],launcher_sha256=binding['launcher_sha256'],registration_script_sha256=binding['register_sha256'],
    source_sha256=e['source_sha256'],source_count=57,training_plan_sha256=spec['training_plan_sha256'],
    observer_source=index['observer_source'],observer_sha256=index['observer_sha256'],
    supplementary_source_sha256={index['observer_source']:index['observer_sha256'],**binding['profile_helper_sha256']},
    source_map_scope='Exact 57-file union of the 32 training-plan and extended evaluation maps; the separately index-bound observer and reviewed R4 allocator helper are recorded as supplementary sources.',
    input_sha256=e['input_bindings'],input_verification=e['input_checks'],plan_checks=e['plan_checks'],
    previous_review_status='SOURCE_FAIL',previous_review_sha256=r1['report_sha256']['reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json'],
    preserved_r1_report_sha256=lineage,resolved_blockers=resolutions,case_source_pass_files_issued=24,
    focused_regression_sha256=sha(TRACE+'/round2/focused_regressions.json'),
    next_action='Source admission only. Actual R4 paired native0/numeric equality/reduced-PSS proof and per-case real sanity remain mandatory; phase6/8 additionally require measured admitted health receipts. No runtime/model/accuracy certification.')
training['checks']['binding'].update(details='R2 hashes verified. Original55 source files and eight plans unchanged; all24 resealed plans retain original data/order/seed/LR/architecture/batch recipe. Index additionally seals the observer and both historical baseline files.')
training['checks']['generated_programs']=dict(status='PASS',details='98 generated remote programs parse; all48 argv vectors match. Five exact ownership-condition and nine exact health-scorer fixtures pass using only inert in-memory I/O. Fixtures are synthetic source regressions, not empirical training or allocator evidence.')
training['checks']['wave_lifecycle']=dict(status='PASS',resolved_blocker_ids=['T1','T2','T3'])
training['checks']['host_allocator_recipe']=dict(status='PASS_RUNTIME_REQUIRED',profile_plan_sha256=binding['profile_plan_sha256'],profile_source_review_sha256=binding['profile_review_sha256'],details='Launcher now requires exact R4 profile-plan/review/source/case-list bindings, native0 for both cases,265-state/optimizer/CSV equality and strictly reduced compact sampled peak PSS. The R4 source guard and all51 source hashes are checked locally. Actual R4 outputs were not read or certified; earlier failed profile observations are not adopted as success.')
training['checks']['health_policy']=dict(status='PASS_SOURCE_ONLY',minimum_interval_seconds=600,nominal_observation_offset_seconds=1200,aggregate_ratio_strict_minimum=1.0,minimum_original_rate_retention=0.4,maximum_task_swap_growth_kib=65536,minimum_current_mem_available_kib=8388608,scope='Operational staged admission; these limits are not optimality, full-training equivalence or performance claims.')
training['runtime_requirements'][0]='The actual R4 paired allocator result was not inspected and is not certified. Before any new launch, the source requires exact reviewed R4 plan/source/case binding, native0 in default+compact, bit-equal265 model tensors and optimizer state, byte-equal trajectory/query-label CSVs, and strictly lower compact sampled peak PSS.'
training['runtime_requirements'].append('Phase6/8 require bound ADMIT_NEXT_WAVE observations with actual throughput, original-task retention, task Swap growth and current MemAvailable. HOLD does not stop/restart tasks; this audit does not establish any observed admission.')
artifacts={n:h for n,h in training['audited_artifact_sha256'].items() if n not in ['reports/HOST_ALLOCATOR_PROFILE_PLAN_20261027.json','reports/host_allocator_default_plan_20261027.json','reports/host_allocator_compact_plan_20261027.json']}
for name in list(artifacts)+[index['host_profile_plan'],profile['source_review'],index['baseline_previous']]+[c['plan'] for c in profile['cases']]:artifacts[name]=sha(name)
training['audited_artifact_sha256']=artifacts
write('reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json',training)
aggregate_sha=sha('reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json')

guard_hashes={}
for row in index['plans']:
    plan=read(row['path']);assert sha(row['path'])==row['sha256']
    path=plan['source_review'];assert not (ROOT/path).exists(),path
    guard=dict(status='SOURCE_PASS',blocking_issues=0,blockers=[],review_round='R2',generated_at=now,
        plan=row['path'],plan_sha256=row['sha256'],source_sha256=plan['source_sha256'],
        aggregate_review='reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json',aggregate_review_sha256=aggregate_sha,
        index_sha256=binding['index_sha256'],launcher_sha256=binding['launcher_sha256'],
        audit_skill='experiment-audit',reviewer_agent=training['reviewer_agent'],
        requested_reviewer_model='gpt-6-astra',requested_reviewer_reasoning_effort='max',actual_reviewer_model='UNATTESTED',actual_reviewer_reasoning_effort='UNATTESTED',
        review_independence='same-family',acceptance_status='provisional',trace_path=TRACE,
        scope='Source-only guard for the exact registered plan and sources. Actual R4 allocator proof, real native0 sanity/frozen/replay/teacher/reload, originalB6250 formal reset and wave capacity/health gates remain mandatory. No execution, runtime/model/accuracy certification.')
    write(path,guard);guard_hashes[path]=sha(path)

evaluation.update(review_round='R2',generated_at=now,status='SOURCE_PASS',blocking_issues=0,blockers=[],
    protocol_sha256=binding['protocol_sha256'],source_sha256=spec['source_sha256'],training_plan_sha256=spec['training_plan_sha256'],
    audited_artifact_sha256=artifacts,input_verification=e['input_checks'],
    previous_review_sha256=r1['report_sha256']['reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.json'],preserved_r1_report_sha256=lineage,
    training_source_review_sha256=aggregate_sha,
    training_admission='Aggregate source review R2 is SOURCE_PASS/0; this does not certify actual R4 allocator proof, per-case sanity, full training or wave health. All runtime gates remain mandatory.',
    runtime_requirements=training['runtime_requirements']+['Evaluation still requires completed native0 full training, smoke, full dev, fresh result audit and frozen selection. No runtime, quality or FPS certification.'])
evaluation['checks']['protocol_extension']['details']='R2 rebinds the two new orchestration source hashes and24 plan hashes. All original55 sources, eight original plan hashes and all original evaluation rules are unchanged; extended map remains57. Only source_review/source_sha256/training_plan_sha256/scope differ from the original protocol.'
write('reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.json',evaluation)

text='''# VRAM18 training source review — R2

**SOURCE_PASS — 0 blockers.** Same-family/provisional. Requested Astra/max; actual model and reasoning effort UNATTESTED. This is source admission only, with no remote/runtime/NN/weight inspection or implementation edit.

R1 is preserved byte-for-byte under `'''+TRACE+'''/round1/`. Its three blockers are resolved:

'''
for r in resolutions:text+='- **'+r['id']+'**: '+r['details']+' Evidence: '+', '.join('`'+v+'`' for v in r['evidence'])+'.\n'
text+='''
All57 training/evaluation source hashes,24 new plan hashes and the index-bound observer are coherent. The original55 source files, eight original plans, exact32-configuration grid, seed, training recipe and input/order bindings remain unchanged. All98 generated programs parse;48 argv vectors match. Five ownership and nine health fixtures pass; their data are explicitly synthetic and do not constitute measured results.

The actual R4 paired allocator result remains unread and uncertified. Native0 for both cases, exact265-state/optimizer/CSV equality, lower compact sampled peak PSS and each candidate's real sanity remain mandatory. Stage6/8 additionally require actual bound health admission. No original job is stopped, altered or restarted.

Issued24 exact per-plan source guards, each binding this aggregate report SHA. Evaluation retains full97dev at three costs, fresh result audit and uniform frozen three-test configuration, total parameters and frame-service FPS.

'''
for name,key in [('Index','index_sha256'),('Launcher','launcher_sha256'),('Extended protocol','protocol_sha256')]:text+=name+' SHA256: `'+binding[key]+'`.\n\n'
(ROOT/'reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.md').write_text(text,encoding='utf-8')
eval_text='''# VRAM18 evaluation source review — R2

**SOURCE_PASS — 0 blockers.** Same-family/provisional; requested Astra/max, actual model/effort UNATTESTED.

The resealed57-source/32-plan protocol retains all original evaluation rules and eight original plan hashes. It still requires native0/full882×2 completion and265-tensor witness; actual11-frame smoke+reload; full97dev/54646frames at costs0/.05/.15; fresh audit binding the exact plan, checkpoint and three reports; strict dev SR>56.43314371195134; and one frozen checkpoint/config across LasHeR245/220703, RGBT210210/104706 and RGBT234234/116650. Dataset GT, supplied toolkit metrics, actual total parameters, decode+initialize-inclusive service FPS and mean/P95 timing remain unchanged.

Training R2 now passes source review, but actual R4 allocator equivalence/reducedPSS, candidate sanity, wave health and full-training/evaluation results remain runtime-gated. No live result, model quality, accuracy or FPS is certified. R1 report bytes are preserved in the review trace.

Protocol SHA256: `'''+binding['protocol_sha256']+'`.\nTraining source review SHA256: `'+aggregate_sha+'`.\n'
(ROOT/'reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.md').write_text(eval_text,encoding='utf-8')

prompt='''R2 revised and ready, all originals8/source55 unchanged. Launcher4ce30f1fdec91bc199a410861bedfd5d4c058753114de01b4c7703a69c680d42; registrar0ecc88342a5e3a4aa02b5d3852afc798c6f3c0d6703e07c2642ee99c66d157ec; index7ee21e8e15a407c8dd20e9792cb84e44a07ef303a538f94174afc4fd0084892b; extendedprotocol6f0a4c021cdab555f686bd7a3b00262df35487653cd3fd2ed1235adaffad618f. B1 added actual poststartup measurement+--health-node scorer, requires >=600sec same-NN progress/raw task identity/finite losses, aggregate speed>previousstage, original8minretention>=.4 vs baseline, Swapgrowth<=64MiB, MemAvailable>=8GiB; nextphase requires boundADMIT_NEXT_WAVE receipt. New read-only tools/observe_vram18_training.py binds observerSHA inindex and includesactualSwap/PSS; audit this newfile too. B2 explicit cuda_ready=True only original/priorphase warmed, False current initialized; required_warm<=actualCUDA<=all_registered, slotlimit based registered counts. B3 immediatelypersist each launch PID before nexttask, including partial startup identities. Also exact R4 host-profile plan/review/source/case binding. Original R3 host profile runtime FAILED from actual /proc disappearance race/killed ONLYM0group; preserved records. New R4 paired profile currently running with source1b77749ePASS after specific process-exit samplerfix, outputR4 deferred/no claims untilactual0. Do NOT queryremote/runtime/NN. Please focused R1regressions + generated programs/healthscore fixture and hash/coherence; no expanded unrelatedsuites. IfPASS produce caseguards+aggregate and updatedextended evaluation guard; keep R1lineage. No implementation edits.'''
write(TRACE+'/002-source-review.request.json',dict(call_number=2,purpose='focused-R1-regressions',tool='followup_task',timestamp=now,prompt=prompt,files_referenced=list(artifacts)+[index['observer_source'],'tools/launch_vram18_training_wave.py','tools/register_vram18_training_plans.py']))
(ROOT/TRACE/'002-source-review.response.md').write_text(text+'\n'+eval_text,encoding='utf-8')
write(TRACE+'/002-source-review.meta.json',dict(call_number=2,purpose='focused-R1-regressions',timestamp=now,agent_id=training['reviewer_agent'],requested_model='gpt-6-astra',requested_reasoning_effort='max',actual_model='UNATTESTED',actual_reasoning_effort='UNATTESTED',review_independence='same-family',acceptance_status='provisional',status='completed',training_verdict='SOURCE_PASS',evaluation_verdict='SOURCE_PASS'))
report_paths=['reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json','reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.md','reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.json','reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.md']
receipt=dict(status='SOURCE_PASS',blocking_issues=0,round='R2',report_sha256={n:sha(n) for n in report_paths},case_review_sha256=guard_hashes,
    index_sha256=binding['index_sha256'],launcher_sha256=binding['launcher_sha256'],protocol_sha256=binding['protocol_sha256'],observer_sha256=index['observer_sha256'],source_sha256=e['source_sha256'],training_plan_sha256=spec['training_plan_sha256'],preserved_r1_report_sha256=lineage)
write(TRACE+'/round2/review_receipt.json',receipt)
with (ROOT/'.aris/meta/events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='review_trace',skill='experiment-audit',purpose='focused-R1-regressions',agent_id=training['reviewer_agent'],trace_path=TRACE,status='completed',round='R2',training_verdict='SOURCE_PASS',evaluation_verdict='SOURCE_PASS'))+'\n')
assert all(sha(n)==h for n,h in e['source_sha256'].items())
assert all(sha(n)==h for n,h in spec['training_plan_sha256'].items())
for row in index['plans']:
    p=read(row['path']);v=read(p['source_review'])
    assert v['status']=='SOURCE_PASS' and v['blocking_issues']==0 and v['plan_sha256']==row['sha256'] and v['source_sha256']==p['source_sha256'] and v['aggregate_review_sha256']==aggregate_sha
print(json.dumps(dict(status='SOURCE_PASS',blocking_issues=0,case_guards=24,report_sha256=receipt['report_sha256'],index_sha256=binding['index_sha256'],launcher_sha256=binding['launcher_sha256'],protocol_sha256=binding['protocol_sha256'],runtime_certified=False),indent=2))
