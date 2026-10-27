"""Publish R3 source guards; preserve all prior review bytes and no runtime claims."""
import copy
from datetime import datetime,timezone
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path('C:/Users/gb/projects/Dual-RGBT')
TRACE='.aris/traces/experiment-audit/2026-10-27_vram18_source_review'
OUT=ROOT/TRACE/'round3'
read=lambda n:json.loads((ROOT/n).read_bytes())
sha=lambda n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
def write(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
e=read(TRACE+'/round3/deterministic_checks.json');b=read(TRACE+'/round3/bindings_summary.json');f=read(TRACE+'/round3/focused_regressions.json')
index=read('reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json');spec=read('reports/VRAM18_EVALUATION_PROTOCOL_20261027.json')
original=read('reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json');profile=read(index['host_profile_plan']);host_review=read(profile['source_review'])
assert b['status']==f['status']=='PASS'
assert all(sha(n)==h for n,h in e['source_sha256'].items())
assert all(sha(n)==h for n,h in spec['training_plan_sha256'].items())
assert sha('reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json')==b['index_sha256']
assert host_review['status']=='SOURCE_PASS' and host_review['blocking_issues']==0
assert host_review['profile_plan_sha256']==b['profile_plan_sha256'] and host_review['source_sha256']==profile['source_sha256']
assert all(sha(n)==h for n,h in profile['source_sha256'].items())

r2=read(TRACE+'/round2/review_receipt.json')
lineage={}
prior=OUT/'prior_reviews';prior.mkdir(exist_ok=True)
old_paths=dict(r2['report_sha256']);old_paths.update(r2['case_review_sha256'])
old_original='reports/JOINT_CANDIDATE_EVALUATION_SOURCE_REVIEW_20261026.json'
old_paths[old_original]=sha(old_original)
original_md=ROOT/'reports/JOINT_CANDIDATE_EVALUATION_SOURCE_REVIEW_20261026.md'
if original_md.exists():old_paths[original_md.relative_to(ROOT).as_posix()]=sha(original_md.relative_to(ROOT).as_posix())
for name,digest in old_paths.items():
    assert sha(name)==digest
    dest=prior/Path(name).name
    if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
    else:dest.write_bytes((ROOT/name).read_bytes())
    lineage[dest.relative_to(ROOT).as_posix()]=digest
for name in ['scripts/train_candidate_host_compact.py','tools/launch_vram18_training_wave.py','tools/register_vram18_training_plans.py','scripts/joint_evaluation_contract.py','tools/profile_host_allocator_trim.py']:
    dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
old_contract=(ROOT/TRACE/'pre_host_trim_eval/joint_evaluation_contract.py').read_text().splitlines(True)
new_contract=(ROOT/'scripts/joint_evaluation_contract.py').read_text().splitlines(True)
(OUT/'evaluation_contract.diff').write_text(''.join(difflib.unified_diff(old_contract,new_contract,fromfile='pre_host_trim_eval/joint_evaluation_contract.py',tofile='scripts/joint_evaluation_contract.py')))
training=read('reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json');evaluation=read('reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.json')
now=datetime.now(timezone.utc).isoformat()
runtime=[
 'No TRIM runtime result or completed reference sample was read in this review. The launcher must obtain exact reviewed TRIM profile-plan/source/case-list binding and native0 for both the explicitly reused default reference and new compact admission.',
 'The new admission requires all265 model tensors and both trajectory/query-label CSVs bit/byte exact. Optimizer state is a declared tolerance gate, not bit-exact equivalence: shapes/dtypes/scalars match, all tensor values finite, atol1e-8 and rtol1e-6. Actual optimizer_bit_equal and maximum absolute error must be recorded. Earlier strict-bit failure is not relabeled as a pass.',
 'Actual compact sampled peak PSS must be below the recorded default reference. Four-update sampled whole-process memory, including load/reload, is not continuous peak or full-training steady-state proof. The wrapper records its own pre/post-trim memory separately.',
 'Each of the24 cases still requires its own real4-update/20-frame native0 sanity, selected replay/teacher preservation,251 frozen+14 changed tensors and strict reload before originalB6250 formal reset; no sanity checkpoint initializes formal.',
 'Each launch requires current foreign-process exclusion, registered per-card slot capacity, GPU capacity and current-wave RAM reservation plus initializing-child headroom with >8GiB remaining. Future waves independently require bound actual health admission.',
 'Full882x2 training, actual native completion,11-frame evaluation smoke, full97dev at three costs, fresh result audit and uniform frozen full three-test evaluations remain mandatory before model/accuracy/FPS claims.'
]
artifacts={n:h for n,h in training['audited_artifact_sha256'].items() if 'HOST_ALLOCATOR' not in n and 'host_allocator' not in n}
for n in list(artifacts)+[index['host_profile_plan'],profile['source_review']]+[c['plan'] for c in profile['cases']]:artifacts[n]=sha(n)
training.update(status='SOURCE_PASS',blocking_issues=0,blockers=[],review_round='R3',generated_at=now,
    index_sha256=b['index_sha256'],launcher_sha256=b['launcher_sha256'],registration_script_sha256=b['register_sha256'],
    source_sha256=e['source_sha256'],source_count=58,training_plan_sha256=spec['training_plan_sha256'],
    source_map_scope='Exact58-file union of the32 training-plan/extended-evaluation maps, including compact wrapper. Observer and separate profile helpers are supplementary.',
    supplementary_source_sha256={index['observer_source']:index['observer_sha256'],**{n:h for n,h in profile['source_sha256'].items() if 'allocator' in n}},
    input_sha256=e['input_bindings'],input_verification=e['input_checks'],plan_checks=e['plan_checks'],audited_artifact_sha256=artifacts,
    previous_review_status='SOURCE_PASS',previous_review_sha256=r2['report_sha256']['reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json'],
    preserved_r2_guard_sha256=lineage,case_source_pass_files_issued=24,runtime_requirements=runtime,
    focused_regression_sha256=sha(TRACE+'/round3/focused_regressions.json'),
    next_action='Source admission only. No new formal job until actual TRIM profile and each case sanity pass; later waves require actual health admission. No runtime/model/accuracy certification.')
training['checks']['binding']=dict(status='PASS',source_files=58,source_mismatches=0,original_active_training_sources_unchanged=51,original_training_plan_hashes_retained=8,new_plan_hashes_verified=24,
    details='All32 configurations remain unique. All24 plans add only the explicit compact trainer/source binding to the already reviewed declared recipe changes. Original51 active training sources and eight plan bytes remain exact. Original55 evaluation source map changes only the metadata contract helper; its historical bytes/protocol/review are retained.')
training['checks']['training_recipe']=dict(status='PASS',details='Architecture, three module duties, losses, seed2027, sequence order, batch behavior, LR1e-4/preview1e-5, weight decay1e-4, SOTep14 and originalB6250 remain unchanged. Full882x2/182182 updates/917106 tracked frames/1764 initializes still enforced. The new engineering wrapper performs one post-load host heap release before calling the exact original trajectory function; no new model module is claimed.')
training['checks']['compact_wrapper']=dict(status='PASS_SOURCE_ONLY',wrapper_sha256=b['wrapper_sha256'],evidence=['scripts/train_candidate_host_compact.py:10','scripts/train_candidate_host_compact.py:16','scripts/train_candidate_host_compact.py:19','scripts/train_candidate_host_compact.py:24'],details='Dispatches objective-specific original module.main without changing argv. The injected stage invokes gc.collect and libc.malloc_trim(0) once after original model/optimizer/before-state/config setup, records pre/post memory, and forwards the same eight objects/arguments to original trajectory_stage. Four inert fixtures cover both objectives and sanity/formal; no real libc/model action was executed by this reviewer.')
training['checks']['generated_programs']=dict(status='PASS',details='48 exact wrapper argv vectors and98 newly expanded remote strings parse. All32 native metadata fixtures accept the correct old/new trainer path and reject the alternate path. R2 unchanged ownership/health fixtures are reused; no unrelated suites run.')
training['checks']['host_allocator_recipe']=dict(status='PASS_RUNTIME_REQUIRED',profile_plan_sha256=b['profile_plan_sha256'],profile_source_review_sha256=sha(profile['source_review']),evidence=['tools/profile_host_allocator_trim.py:29','tools/profile_host_allocator_trim.py:74','tools/profile_host_allocator_trim.py:101','tools/profile_host_allocator_trim.py:104','tools/launch_vram18_training_wave.py:79'],details='One new compact admission with explicit hash-bound completed default reference; reference is not represented as a current rerun. Native0 and exact model/CSV remain required; finite optimizer tolerance1e-8/1e-6 and actual bit flag/max absolute error are separately reported. Five synthetic proof-gate fixtures pass. The source guard is bound and all profile-source hashes match; runtime result remains unread and uncertified.')
training['checks']['wave_ram']=dict(status='PASS_RUNTIME_REQUIRED',evidence=['tools/launch_vram18_training_wave.py:139','tools/launch_vram18_training_wave.py:142','tools/launch_vram18_training_wave.py:153','tools/launch_vram18_training_wave.py:158'],details='Current MemAvailable already includes resident owners. Reserve remaining current-stage cases times sampled compact peak, plus max(0,peak-currentPSS) for each registered initializing child across all GPUs; require >8GiB remainder. Completed sanities are skipped. For each of three phases and both sanity/formal, new_slots is exactly8..1. Five synthetic arithmetic fixtures pass. Later waves remain separately health-gated; no simultaneous future32-job reload-peak assertion remains.')

write('reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json',training)
aggregate_sha=sha('reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json');case_hashes={}
for row in index['plans']:
    p=read(row['path']);assert sha(row['path'])==row['sha256']
    old=read(p['source_review'])
    guard=dict(old,status='SOURCE_PASS',blocking_issues=0,blockers=[],review_round='R3',generated_at=now,
        plan_sha256=row['sha256'],source_sha256=p['source_sha256'],aggregate_review_sha256=aggregate_sha,
        index_sha256=b['index_sha256'],launcher_sha256=b['launcher_sha256'],previous_case_review_sha256=r2['case_review_sha256'][p['source_review']],
        launch_trainer=p['launch_trainer'],scope='Exact-source admission only. Actual reviewed TRIM proof (exact model/CSV, finite optimizer atol1e-8/rtol1e-6 with honest bit flag), reduced sampled PSS, real case sanity, originalB formal reset and staged capacity/health gates remain mandatory. No runtime/model/accuracy certification.')
    write(p['source_review'],guard);case_hashes[p['source_review']]=sha(p['source_review'])

evaluation.update(status='SOURCE_PASS',blocking_issues=0,blockers=[],review_round='R3',generated_at=now,
    protocol_sha256=b['protocol_sha256'],source_sha256=spec['source_sha256'],source_count=58,original_protocol_sha256=b['original_protocol_sha256'],
    training_plan_sha256=spec['training_plan_sha256'],audited_artifact_sha256=artifacts,input_verification=e['input_checks'],
    previous_review_sha256=r2['report_sha256']['reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.json'],preserved_r2_guard_sha256=lineage,
    training_source_review_sha256=aggregate_sha,training_admission='R3 training source passes; actual TRIM profile, case sanity, phase health and full-training/evaluation evidence remain mandatory.',runtime_requirements=runtime)
evaluation['checks']['protocol_extension']=dict(status='PASS',details='Extended58-source/32-plan protocol is coherent. Original eight training-plan hashes remain exact. Relative to preserved original8 evaluation protocol, only the contract helper source hash changes; the helper diff is exactly optional explicit launch_trainer selection and assertion of its presence in the training source map. Old plans continue to select original TRAINERS, new24 select the registered compact wrapper. All32 correct/wrong-native-path fixtures pass. Metric, dataset, smoke,97dev/fresh-audit, uniform-test-freeze and service-FPS rules are unchanged.',evidence=['scripts/joint_evaluation_contract.py:60','scripts/joint_evaluation_contract.py:61',TRACE+'/round3/evaluation_contract.diff'])
write('reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.json',evaluation)

original_review=copy.deepcopy(evaluation)
original_review.update(protocol_sha256=b['original_protocol_sha256'],source_sha256=original['source_sha256'],source_count=55,
    training_plan_sha256=original['training_plan_sha256'],previous_review_sha256=old_paths[old_original],
    protocol='reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json',
    review_scope='Targeted reseal of original8 evaluation after metadata trainer-dispatch extension only. The original8 training plans and51 active training source hashes are unchanged; their native commands still use original objective trainers.',
    runtime_requirements=['Original8 evaluation requires actual complete native0/full882x2 training, exact final/B tensor witness, actual11-frame smoke+reload, complete97dev at three costs, fresh exact-bound result audit and uniform frozen three-test configuration. No runtime/model/accuracy/FPS result is certified.'],
    training_admission='This guard covers evaluation of the original eight unchanged training plans. Compact/TRIM requirements apply to the separately registered new24 and do not alter these original8 recipes.')
write(old_original,original_review)

md='''# VRAM18 training source review — R3

**SOURCE_PASS — 0 blockers.** Same-family/provisional; requested Astra/max, actual model/effort UNATTESTED. No remote/NN/GPU/active-log/PID/weight/environment query, real libc action or implementation edit occurred.

The58-source/32-plan bindings match. Original51 active training sources and eight training plans remain unchanged. The compact wrapper calls the objective's original trainer main without altering argv, then performs one post-load GC/malloc_trim before forwarding the same arguments to the original trajectory function. Model architecture, losses, seed, data order, LR/batch and B6250 reset remain unchanged.

The host proof now honestly distinguishes exact265-model/CSV equality from finite optimizer tolerance (atol1e-8/rtol1e-6), recording actual bit equality and maximum absolute error. It explicitly reuses the sealed completed default reference and requires a new native0 compact admission plus reduced sampled peak PSS. The actual TRIM result and reference sample were not read or certified.

The RAM guard reserves the current wave's remaining new cases and the unallocated peak headroom of initializing registered children, leaving >8GiB available. Later waves retain R2 actual health admission. Current-stage new_slots is8..1 in every sanity/formal phase. No original job is stopped, altered or restarted.

Focused verification: four wrapper fixtures; all32 native-path fixtures (correct accepted, alternate rejected);48 argv vectors;98 generated program parses; five RAM and five proof-gate fixtures. R2 unchanged ownership/health checks are reused. All fixture objects are synthetic and are not experimental evidence.

Issued refreshed24 source guards. Evaluation contract changes only trainer-path selection and a source-map assertion; both original8 and expanded32 evaluation guards are resealed. Full97dev, fresh audit, uniform frozen tests, total parameters and service FPS rules remain unchanged. All prior review/guard bytes and pre-trim evaluation snapshots are preserved in the trace.

'''
for name,key in [('Index','index_sha256'),('Launcher','launcher_sha256'),('Compact wrapper','wrapper_sha256'),('Extended protocol','protocol_sha256'),('Original8 protocol','original_protocol_sha256')]:md+=name+' SHA256: `'+b[key]+'`.\n\n'
(ROOT/'reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.md').write_text(md,encoding='utf-8')
emd='''# VRAM18 evaluation source review — R3

**SOURCE_PASS — 0 blockers.** Same-family/provisional; requested Astra/max, actual model/effort UNATTESTED.

The extended58-source/32-plan and original55-source/8-plan protocols are coherent. The only evaluation-code change selects an explicit registered compact launch_trainer when present, otherwise the existing objective-specific trainer, and asserts that source is bound. Correct and alternate native command paths were exercised using32 synthetic metadata fixtures. Original training plans and51 active sources remain unchanged; prior evaluation source/protocol/review bytes remain preserved.

All real-GT/toolkit, completed-training/tensor witness,11-frame smoke,97dev/three-cost/fresh audit, strict eligibility and uniform full three-test freeze rules remain unchanged. Total parameters, decode+initialize-inclusive service FPS and latency reporting remain unchanged. This is source readiness; no actual runtime, accuracy, memory saving or FPS is certified.

Extended protocol SHA256: `'''+b['protocol_sha256']+'`. Original8 protocol SHA256: `'+b['original_protocol_sha256']+'`.\n'
(ROOT/'reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.md').write_text(emd,encoding='utf-8')
(ROOT/'reports/JOINT_CANDIDATE_EVALUATION_SOURCE_REVIEW_20261026.md').write_text(emd.replace('# VRAM18 evaluation source review — R3','# Original8 joint evaluation source reseal — R3'),encoding='utf-8')

prompt='''R3 necessary engineering change BEFORE ANY24formal launched; old active51 trainer sources/plans untouched. R4 realpairedM0 bothnative0, net265/CSV bitexact, optimizerbuffers max4.6566e-10 so STRICT optimizer-bit admission failed (report HOST_ALLOCATOR_R4_COMPARISON_DIAGNOSIS...); compact env alone PSS~3.76GiB stillRAM-limited. New small scripts/train_candidate_host_compact.py SHAd62266395ca43967f959787850892db0c0ab08c3666b11fc67eeb50328c162d3 dispatches exact original trainer.main; one post-load gc.collect+libc.malloc_trim(0) before original trajectory_stage, writes pre/postPSS, no argv/RNG/model/optimizer changes. New singleprobe profile recipe HOST_ALLOCATOR_TRIM_PROFILE_PLAN_20261027.json uses existing completed defaultR4reference explicitly, tight finite optimizer atol1e-8/rtol1e-6 with actualbitflag/errors recorded, model/CSV still exact; source review by hostallocator agent underway, actual remainsruntimegate. New launcher1335fa7e5015b09a47a908afc3a3a9a1f641fdf5f709541aa7a0694b30839832 uses plan.launch_trainer, exactTRIM profile binding; RAM guard now current wave new-slots×sampledpeak plus headroom for initializing ownedjobs to peak, then8GiBfree; laterwaves independently actualhealth gated (avoids fictitious all32simultaneous reload-clonepeak). Registrar3a02e8cea76055995019069dc8db7efbdad18e82fc7b69e5e0dddbfaf098ea16; indexe4ff359c6abc177d66d53da3416194f471ce42c5471d7984d062b78f94b61f44. Evaluation helper scripts/joint_evaluation_contract.py changed ONLY trainer selection to explicit optional launch_trainer else existingTRAINERS + assert inplan source map. This helper is OUTSIDE active51 NN sources; no evalmodel/plan/run exists yet. Snapshot prior original8evalcode/protocol/review under trace/pre_host_trim_eval. Original8evalprotocol now7a1aa4182278cfc93ff3061e3ba26ad60c447dcd14f173daccf24ffaeabb4fe2; extended32protocol93469fbaee31574e435b825fb4a02b1fafb8d40c5290ef9c7621b6359e4744ff/source58. Please targeted wrapper/hostproof/RAMguard/old8+new24 native path fixtures/hash updates/generated compile; reuseunchangedR2 ownership/healthscores. Do not broaden. NO remoteNNqueries or edits. IfPASS issue refreshed24caseguards+aggregate+extended evaluation and ORIGINAL8 evalsource review at reports/JOINT_CANDIDATE_EVALUATION_SOURCE_REVIEW_20261026.json with exact55map/protocolnewSHA. Preservealloldreviews/snapshots. Runtimeprofile result not claimed; no newformal untilactualadmission.'''
write(TRACE+'/003-source-review.request.json',dict(call_number=3,purpose='host-trim-targeted-review',tool='followup_task',timestamp=now,prompt=prompt,files_referenced=list(artifacts)+list(spec['training_plan_sha256'])+['scripts/train_candidate_host_compact.py','scripts/joint_evaluation_contract.py','tools/profile_host_allocator_trim.py']))
(ROOT/TRACE/'003-source-review.response.md').write_text(md+'\n'+emd,encoding='utf-8')
write(TRACE+'/003-source-review.meta.json',dict(call_number=3,purpose='host-trim-targeted-review',timestamp=now,agent_id=training['reviewer_agent'],requested_model='gpt-6-astra',requested_reasoning_effort='max',actual_model='UNATTESTED',actual_reasoning_effort='UNATTESTED',review_independence='same-family',acceptance_status='provisional',status='completed',training_verdict='SOURCE_PASS',extended_evaluation_verdict='SOURCE_PASS',original_evaluation_verdict='SOURCE_PASS'))
paths=['reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json','reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.md','reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.json','reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.md',old_original,'reports/JOINT_CANDIDATE_EVALUATION_SOURCE_REVIEW_20261026.md']
receipt=dict(status='SOURCE_PASS',blocking_issues=0,round='R3',report_sha256={n:sha(n) for n in paths},case_review_sha256=case_hashes,
    index_sha256=b['index_sha256'],launcher_sha256=b['launcher_sha256'],protocol_sha256=b['protocol_sha256'],original_protocol_sha256=b['original_protocol_sha256'],source_sha256=e['source_sha256'],training_plan_sha256=spec['training_plan_sha256'],preserved_prior_review_sha256=lineage,runtime_certified=False)
write(TRACE+'/round3/review_receipt.json',receipt)
with (ROOT/'.aris/meta/events.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(dict(event='review_trace',skill='experiment-audit',purpose='host-trim-targeted-review',agent_id=training['reviewer_agent'],trace_path=TRACE,round='R3',status='completed',training_verdict='SOURCE_PASS',extended_evaluation_verdict='SOURCE_PASS',original_evaluation_verdict='SOURCE_PASS'))+'\n')
assert all(sha(n)==h for n,h in e['source_sha256'].items())
for row in index['plans']:
    p=read(row['path']);v=read(p['source_review'])
    assert sha(row['path'])==row['sha256'] and v['plan_sha256']==row['sha256'] and v['source_sha256']==p['source_sha256'] and v['aggregate_review_sha256']==aggregate_sha and v['status']=='SOURCE_PASS' and v['blocking_issues']==0
for path,protocol in [('reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.json',spec),(old_original,original)]:
    v=read(path);assert v['source_sha256']==protocol['source_sha256'] and v['status']=='SOURCE_PASS' and v['blocking_issues']==0
print(json.dumps(dict(status='SOURCE_PASS',blocking_issues=0,case_guards=24,report_sha256=receipt['report_sha256'],index_sha256=b['index_sha256'],launcher_sha256=b['launcher_sha256'],protocol_sha256=b['protocol_sha256'],original_protocol_sha256=b['original_protocol_sha256'],runtime_certified=False),indent=2))
