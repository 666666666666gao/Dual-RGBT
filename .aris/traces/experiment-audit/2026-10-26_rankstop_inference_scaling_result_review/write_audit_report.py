"""Write this reviewer's assessment and forensic trace without changing inputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

TRACE = Path(__file__).resolve().parent
ROOT = TRACE.parents[3]
TRACE_REL = str(TRACE.relative_to(ROOT)).replace('\\', '/')
BASE = 'RANKSTOP_INFERENCE_SCALING_RESULT_AUDIT_20261026'
REPORT_JSON = ROOT / 'reports' / (BASE + '.json')
REPORT_MD = ROOT / 'reports' / (BASE + '.md')
assert not REPORT_JSON.exists() and not REPORT_MD.exists()
data = json.loads((TRACE / 'DETERMINISTIC_RECOMPUTATION.json').read_bytes())
remote = json.loads((TRACE / 'REMOTE_READONLY_PROVENANCE.json').read_bytes())
assert not data['failed'] and data['deterministic_checks'] == 978
now = datetime.now(timezone.utc).isoformat()
agent = '/root/audit_rankstop_inference_scaling_20261026'
out = 'profile_output/rankstop_inference_scaling_server26_20261026'
native = 'runs/pipeline_tasks/rankstop_inference_scaling_server26_20261026'
source = 'scripts/profile_rankstop_scaling.py'

def line_of(relative, text):
    return next(i for i,line in enumerate((ROOT / relative).read_text(encoding='utf-8-sig').splitlines(),1) if text in line)

recompute = TRACE_REL + '/DETERMINISTIC_RECOMPUTATION.json'
provenance = TRACE_REL + '/REMOTE_READONLY_PROVENANCE.json'

checks = {
    'A_ground_truth_provenance': dict(status='PASS',
        evidence=[source+':34',source+':190','dual_rgbt/data.py:15',provenance+':'+str(line_of(provenance,'"selected_train"'))],
        details='Initialization uses the installed LasHeR TRAIN init.txt first box through the hash-bound TRAIN manifest. All8 installed initial boxes equal the manifest and stored prediction[0]; 882 TRAIN and97 dev names are disjoint for the8 selected streams. Only boxes[0] is used on the profiling path. Model-output equality is an explicitly scoped cross-setting consistency check, not a substitute for dataset GT accuracy. No PR/NPR/SR or accuracy claim is generated.'),
    'B_score_normalization': dict(status='PASS',
        evidence=[source+':146',source+':153',out+'/arm8_n8_cv1/COMPLETE.json:56',recompute+':'+str(line_of(recompute,'"arms"'))],
        details='All12 total-frame counts, earliest-start/latest-finish timestamps, durations, aggregate FPS, sampled GPU0 peak memory and timed-sample mean utilization recompute exactly. Rate numerator is actual stored frame count; denominator is measured elapsed time, never model-output statistics. End-to-end FPS arithmetic recomputes from the stored spawn-to-join duration, but the raw spawn/join endpoints were not saved separately. Any speedup is a ratio of runtime rates and must retain its baseline.'),
    'C_result_file_existence': dict(status='PASS',
        evidence=[native+'/status.json:16','scripts/run_task.py:23',source+':131',source+':213',out+'/COMPLETE.json:15761',recompute+':'+str(line_of(recompute,'"remote_archived_copies"'))],
        details='Top-level COMPLETE,10 unprofiled arm COMPLETE files,2 cProfile COMPLETE files,34 worker JSON files and2 .prof files exist. Parent command/task/GPU0/PID identity binds startup,config,node and native COMPLETE exit0; native wrapper702029 owns profiler parent702037 in the archived startup witness. Every worker PID matches admissions and report ownership and has native join exit0. All12 log summaries and the final summary agree. All54 archived remote artifact hashes match local copies. This profile exit0 does not repair the original training run missing native-exit receipt.'),
    'D_dead_code_and_execution': dict(status='PASS',
        evidence=[source+':87',source+':93',source+':210',source+':217',source+':228',native+'/run.log:365',recompute+':'+str(line_of(recompute,'"profiles"'))],
        details='The executed worker path calls trajectory and final-state construction, and the parent calculates/checks the stored summaries. Both saved cProfile files independently reproduce their top40 entries; each records1 initialize,127 track,128 read_pair,256 imread and257 CUDA synchronize calls. Separate profile arms are excluded from best-arm argmax. All12 saved trajectory sets equal their serial reference, including the profiled stream which the runner itself did not compare. No unused scoring helper is presented as an executed accuracy result.'),
    'E_scope_assessment': dict(status='WARN',
        evidence=['reports/RANKSTOP_INFERENCE_SCALING_PLAN_20261026.json:90','reports/RANKSTOP_INFERENCE_SCALING_PLAN_20261026.json:103',source+':197',source+':233',recompute+':'+str(line_of(recompute,'"workload_action_counts"'))],
        details='The evidence covers one final checkpoint,seed2027,cost0.05,N1/2/4/8,OpenCV80/1,on GPU0 RTX3090,8 selected TRAIN prefixes of128 frames. Each unprofiled arm has8 initializations plus1016 track calls, not1024 track calls. Only19/1016 calls request auxiliary encoding (17 local-main,2 local-alternate); global-local/global-wide branches are not exercised. N2/4/8 have one run per setting; N1 repeats drift -4.324991% atCV80 and+5.418880% atCV1. Fixed ordering, warm filesystem pages and a continuing GPU1 workload limit causal attribution/generalization. Supports a qualified observed-throughput selection, not full97dev/test FPS, single-stream49.33FPS, universal optimality, all-branch equivalence or accuracy improvement.'),
    'F_evaluation_type': dict(status='PASS',
        evidence=[source+':34',source+':213',source+':233'],
        classification='self_supervised_proxy',
        details='The checklist nearest class for the no-accuracy runtime measurement is self_supervised_proxy: this is an empirical system profile with no task score. The serial-output parity subtest is synthetic_proxy consistency; it compares model outputs to model outputs and makes no accuracy inference. Initialization provenance alone is real_gt from installed TRAIN annotations. These labels do not classify the throughput values as synthetic timing.')
}

limitations = [
    dict(id='L1', title='Aggregate throughput is not single-stream service FPS',
        details='Best observed N8/CV1 is49.33089223000486 aggregate FPS over1024 frame events/20.757783889770508s. Its8 individual128-frame worker blocks are6.166403811712504–7.4588949689563115FPS; these are block averages including initialization/state summaries, not per-frame service latency. N1/CV1 serial block results are10.069393362398067 and10.615041723066662FPS. No per-frame mean/P95 service-time trace was saved by this runner.'),
    dict(id='L2', title='Timing scopes and unrecorded raw endpoints',
        details='aggregate_fps excludes imports,model construction/checkpoint load,5-frame worker warmup and post-timing model-state comparison/JSON output, but includes image decoding,8 initializations,1016 track steps,list construction and per-stream final-state hashing. Spawn-to-join end_to_end_fps includes worker load/warmup/verification/output and is23.136924537815403FPS for bestN8, but excludes parent hash checks,filesystem prewarm,monitor lifecycle and final report writes. Parent began/released/joined timestamps are not persisted separately, so only this field arithmetic/source definition is independently verified. Native whole-command elapsed time is967.272358417511s across all12 arms and setup.'),
    dict(id='L3', title='Fixed short scope and low auxiliary-query activity',
        details='8 TRAIN sequence prefixes,128 frames each,seed2027,cost0.05,one final checkpoint. The1016 track calls consist of997STOP,17local-main,2local-alternate and0global-local/global-wide; every route has8 candidate proposals. This sparse-query workload does not establish scaling for other sequences,costs,full trajectories or auxiliary-heavy workloads. No accuracy metrics were computed.'),
    dict(id='L4', title='Ordered arms, repeats and shared host background',
        details='N1 repeats change -4.324991006796742% forCV80 and+5.418880175107654% forCV1. N2/N4/N8 have one replicate each; arm order is fixed and pages were prewarmed. GPU1 remains at2328MiB in every saved sample and has varying sampled utilization. BestN8 timed GPU1 mean utilization7.888888888888889%,range1–17%; per-arm background summaries are preserved. GPU1 host-resource interference is uncontrolled; no confidence interval or universal optimum is supported.'),
    dict(id='L5', title='Saved-state evidence is not independent neural replay',
        details='Audit compares all saved prediction numbers,8 non-timing route fields and final-state SHA256 strings exactly. The digest covers concatenated bytes of box state,templates,history; raw templates/history were not saved and cannot be rehashed by this audit. Frame=127 and since_query can be reconstructed from the fixed workload/routes; deterministic window and other attributes are source-inspected,not independently tensor-replayed. model_state_bit_equal is a witnessed runtime assertion based on state_dict snapshots before warmup and after measurement. This audit performs0 NN calls,0 checkpoint deserializations and0 profiling reruns.'),
    dict(id='L6', title='Sampling and CPU-accounting definitions',
        details='854 GPU samples have mean intervals about1.10–1.13s rather than exactly1.0s because each query takes time. Device peak memory is a sampled peak over the whole arm,utilization an unweighted arithmetic mean of samples inside the worker measurement interval. BestN8 uses18 timed samples,meanGPU0util73.5%,sampled peak18149MiB. Each worker reports peak allocated546334208B and reserved614465536B,distinct from whole-device memory. cpu_children_seconds includes full reaped worker lifetimes and nvidia-smi subprocess accounting,not steady-inference-only CPU time.'),
    dict(id='L7', title='Binding and inherited provenance boundary',
        details='49 listed source files hash-match locally; original48 bindings equal the frozen dev plan. All26 plan input files hash-match in the archived remote project,and all54 copied run artifacts match remote bytes. Original model/tracker builders,strict final state load,eval/requires_grad(false),no_grad,seed2027,torchthreads1 andTF32false are source verified; no optimizer/backward/AMP/half path is introduced. Runtime precision/tensor dtypes are not separately dumped. The49-file plan does not independently seal every imported lib file,experiment YAML,installed package or every image byte; selected image paths exist and first-frame image hashes were captured. Original RankSTOP training native exit remains unknown; this inference run has its own genuine exit0.'),
    dict(id='L8', title='Profile overhead and hotspot interpretation',
        details='Both cProfile arms use only the first stream2up and are excluded from the10-arm speed ranking. Each has383302 total/352879 primitive calls. read_pair cumulative time is4.157345s(CV80) versus2.461215s(CV1),while track is9.601712s versus10.085093s. These overlapping Python cumulative timings can contain CUDA waiting and do not constitute kernel attribution or proof of a single causal bottleneck.'),
    dict(id='L9', title='One saved memory-sum discrepancy retained',
        details='arm7_n4_cv1 sample index6 at1793023553.9618304 reports GPU0 used59MiB/free24069MiB,sum24128MiB versus24127MiB in the other samples. It is before that arm timed start1793023563.5502985,far below the peak,and does not alter any recomputed FPS,peak or timed utilization. Cause is undetermined. The initial audit harness treated exact used+free as invariant; that unsupported check was removed while preserving initial script/output and this observation. No experimental data changed.')
]
claims = [
    dict(id='C1',claim='N8/CV1 had the largest aggregate FPS among these10 measured arms.',impact='supported',value=49.33089223000486),
    dict(id='C2',claim='BestN8 is4.65–4.90x the measuredCV1 N1 baselines.',impact='needs_qualifier',details='4.6472631494992624x versus repeat and4.899092770992562x versus first; aggregate short-workload rates only.'),
    dict(id='C3',claim='All stored predictions/routes/final-state digests are exact across settings.',impact='supported',details='Scope is saved records for these8 prefixes; no independent NN replay or raw-state rehash.'),
    dict(id='C4',claim='This tracker runs49.33FPS on one live stream or full97dev/test.',impact='unsupported'),
    dict(id='C5',claim='Changing OpenCV/concurrency improves accuracy or universally preserves all trajectories.',impact='unsupported'),
    dict(id='C6',claim='No training or precision-reduction change was introduced by this standalone profile.',impact='supported_with_evidence_boundary',details='Source path and runtime unchanged-state assertions; precision not independently replayed and full environment not cryptographically sealed.'),
    dict(id='C7',claim='Reported total model parameter count is103218887.',impact='runtime_witness_supported',details='Identical in all34 worker receipts and all arm summaries; count expression inspected,checkpoint/model not loaded again by auditor.'),
    dict(id='C8',claim='The original training process exited0 because this profile exited0.',impact='unsupported')
]
input_hashes = dict(data['audited_artifact_hashes'])
input_hashes.update({k:v['sha256'] for k,v in remote['inputs'].items()})
report = dict(audit_skill='experiment-audit', verdict='WARN', overall_verdict='WARN', integrity_status='warn',
    status='RESULT_PASS_WITH_LIMITATIONS', reason_code='fixed_short_profile_scope_and_evidence_boundaries',
    summary='978 deterministic checks pass;0 blocking findings. Saved throughput,workload,parity,ownership,input binding and native exit0 are internally and independently arithmetically consistent. Performance claims require the stated scope and timing qualifiers.',
    blocking_issues=0, blockers=[], generated_at=now, date='2026-10-26', project='Dual-RGBT',
    agent_id=agent, verdict_id=agent, auditor='fresh-context delegated Codex auditor',
    requested_model='gpt-6-astra', requested_reasoning_effort='max', actual_model='UNATTESTED',
    actual_reasoning_effort='UNATTESTED', actual_route='UNATTESTED', executor_model='UNATTESTED',
    executor_family='openai', reviewer_model='UNATTESTED', reviewer_family='openai', reviewer_reasoning='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional',
    deterministic_evidence=dict(status='PASS', checks=data['deterministic_checks'],failed=0,review_independence='deterministic',acceptance_status='accepted_for_record_arithmetic_and_hash_checks_only'),
    checks=checks, limitations=limitations, claims=claims,
    action_items=['When reporting speed,retain aggregate vs worker-block vs spawn-to-join scopes,all10 arms and repeated N1 values.',
                  'Keep the8-prefix/cost0.05/low-query/shared-host limitations and do not use this profile as accuracy or a full97 result.',
                  'Do not promote stored hash/state assertions into independent neural replay or use inference exit0 to fill the historical training-exit gap.'],
    scope=dict(checkpoint_sha256='527a3bc0f56f6a26e31adcc5e656f971b6bd34bbcd220251151d7db1cc228753',
               total_parameters=103218887,total_parameters_evidence='runtime sums corroborated across34 worker records;no independent model instantiation',
               hardware='NVIDIA GeForce RTX3090 physicalGPU0',background_gpu='physicalGPU1 continuing workload disclosed in archived samples',
               seeds=[2027],cost_weight=0.05,sequences=[x['sequence'] for x in data['per_sequence']],frames_per_sequence=128,
               unprofiled_arms=10,profiled_arms=2,workers=34,timed_frames_all_arms=10496,timed_initializations=82,
               timed_track_calls=10414,excluded_warmup_frames=170,excluded_warmup_initializations=34,excluded_warmup_track_calls=136),
    native_exit=data['native'],command_wall_seconds=data['command_wall_seconds'],
    numeric_evidence=dict(arms=data['arms'],repeated_baseline=data['repeated_baseline'],best_arm=data['best'],
        per_sequence=data['per_sequence'],workload_action_counts=data['workload_action_counts'],
        workload_candidate_count_distribution=data['workload_candidate_count_distribution'],profiles=data['profiles'],
        sample_sum_anomalies=data['sample_sum_anomalies']),
    binding=dict(plan_sha256=data['plan_sha256'],source_files=49,inherited_source_files=48,local_inputs=14,remote_verified_inputs=26,
        remote_archived_copies=54,installed_TRAIN_first_boxes=8,selected_dev_overlap=[],
        source_and_input_runtime_admission='pre/post SHA assertions reached under native exit0; independently hashed listed local sources and archived remote inputs'),
    audited_input_hashes={k:'sha256:'+v for k,v in sorted(input_hashes.items())},
    trace_path=TRACE_REL+'/',recomputation_path=recompute,remote_provenance_path=provenance,
    execution_scope=dict(checkpoint_loads=0,torch_imports=0,neural_calls=0,new_training=0,new_profiling=0,
        active_dev_or_resource_or_PID_queries=0,remote_mutations=0,source_plan_results_handoff_or_git_edits=0,
        writes='Unique result-audit JSON/Markdown and this trace only'),
    harness_history=[dict(initial_check_failure='Exact integer sampled used+free invariant unsupported by one saved1MiB discrepancy',resolution='Kept raw evidence;split sample field validity from informational memory-sum observation;initial output/script retained'),
                     dict(initial_python_command='Bare python selected E:/Scripts/python.exe and failed with No pyvenv.cfg file',resolution='Used already-installed localPython3.12 executable;stdlib-only;no package/environment changes')])

lines=[
    '# RankSTOP inference scaling result audit — 2026-10-26','',
    '**Verdict: WARN — qualified result PASS; 0 blocking findings.** All978 deterministic checks pass. The observed best configuration is8 processes/OpenCV1 thread at49.330892 aggregateFPS for this fixed TRAIN workload.','',
    'Fresh-context Codex review; same-family/provisional. Requested model gpt-6-astra,effort max; actual model/effort/routing are UNATTESTED. Deterministic arithmetic/hash checks are distinguished from semantic review and from neural replay.','',
    '## Verified measurements','',
    'Each unprofiled arm performs the same8 TRAIN streams ×128 frame events:8 initializations +1016 tracking calls. The frame numerator includes initialization; no frames are dropped or repeated within an arm. Model total parameter count103,218,887 is corroborated in all34 worker receipts.','',
    '| Arm | Processes | OpenCV threads | Measured seconds | Aggregate FPS | Spawn-to-join FPS | GPU0 sampled peak MiB | GPU0 timed mean utilization % |',
    '|---|---:|---:|---:|---:|---:|---:|---:|'
]
for a in data['arms'][:10]:
    lines.append('| '+a['arm']+' | '+str(a['processes'])+' | '+str(a['opencv_threads'])+' | '+f"{a['inference_wall_seconds']:.6f} | {a['aggregate_fps']:.6f} | {a['end_to_end_fps']:.6f} | {a['gpu0_peak_used_mib']} | {a['gpu0_mean_utilization']:.6f} |")
lines += ['',f'Raw best-arm evidence: `{out}/arm8_n8_cv1/COMPLETE.json:56`; independent per-worker/timing/GPU recomputation: `{recompute}`. All12 arm summaries match the native log; profile arms are excluded from this table/ranking.','',
    'BestN8/CV1 worker block rates are6.1664–7.4589FPS. SerialN1/CV1 blocks are10.0694 and10.6150FPS. Therefore49.3309 is total throughput across8 workers; it is not single-stream49.3309FPS. No per-frame service mean/P95 is available. Spawn-to-join best-arm rate is23.1369FPS; this is not whole-command latency.','',
    '## Integrity checks A–F','']
for key,value in checks.items():
    lines += ['### '+key.replace('_',' ')+' — '+value['status'],'',value['details'],'','Evidence: '+', '.join('`'+s+'`' for s in value['evidence'])+'.','']
lines += ['## Provenance, exit status and equality','',
    'Native wrapper702029 recorded COMPLETE/exit0 for the exact profiler command onGPU0, from2026-10-26T13:54:18.107318Z to14:10:25.379676Z (967.272358s). Archived startup identifies profiler parent702037; all34 spawned worker IDs align with their readiness and report records and all34 joins exit0. No current process query was needed. Source ownership logic is at `scripts/run_task.py:23` and `scripts/profile_rankstop_scaling.py:184`.','',
    'All49 listed source hashes match, including the unchanged48 files inherited from the dev plan. All26 bound remote input hashes match; all54 remote archived artifact hashes match local copies. The8 installed TRAIN initial boxes match init.txt,manifest and stored prediction[0], and selected names do not overlap97dev. Checkpoint SHA256: `527a3bc0f56f6a26e31adcc5e656f971b6bd34bbcd220251151d7db1cc228753`.','',
    'Every one of10,240 stored main-arm prediction rows and10,160 route rows is exactly equal to the corresponding serial reference, including all8 route fields. All80 main-arm final box/template/history digest records agree per sequence. Both separate128-frame cProfile outputs also agree, adding256 prediction rows,254 route rows and2 digests. This independently checks saved records; it does not regenerate network outputs or recover raw template/history tensors.','',
    'The source loads the same final model strictly,sets eval/requires_grad(false),uses no_grad,seed2027,torchthreads1 andTF32false,and asserts the full state_dict unchanged from before warmup through measured execution. No optimizer/backward or precision-reduction path is introduced by this runner. The audit imported no Torch and loaded no checkpoint. These source/runtime witnesses are not independent tensor replay.','',
    '## Limitations and reporting requirements','']
for item in limitations:
    lines += ['### '+item['id']+' — '+item['title'],'',item['details'],'']
lines += ['## Claim impact','', '| Claim | Assessment |', '|---|---|']
for item in claims:
    lines.append('| '+item['claim']+' | '+item['impact']+(' — '+item['details'] if 'details' in item else '')+' |')
lines += ['', '## Reproduction and trace','',
    'This audit used only stdlib parsing,hashing,arithmetic and saved pstats files. The single read-only SSH operation was restricted to archived profile/input hashes and installed TRAIN initialization provenance. No active dev status,resource,temperature,power or PID query was made. No source,input,result,plan,HANDOFF or Git state was changed.','',
    '- Report JSON: `reports/'+BASE+'.json`',
    '- Deterministic verifier/output: `'+TRACE_REL+'/verify_saved_artifacts.py` and `'+recompute+'`',
    '- Read-only remote script/output: `'+TRACE_REL+'/remote_readonly_provenance.py` and `'+provenance+'`',
    '- Full request,response,metadata and hash manifest: `'+TRACE_REL+'/`','',
    'No additional run is required to accept the qualified measurements in this report. Stronger claims remain unsupported by this artifact set.'
]
markdown='\n'.join(lines)+'\n'
REPORT_JSON.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
REPORT_MD.write_text(markdown,encoding='utf-8')

prompt='''Apply experiment-audit to completed standalone inference system-profile. Read C:/Users/gb/.codex/skills/experiment-audit/SKILL.md and shared-references/local-codex-policy.md. Fresh independent context, same-family/provisional, requested Astra/max actual routing UNATTESTED. Local root C:/Users/gb/projects/Dual-RGBT. Paths only, no author interpretation: scripts/profile_rankstop_scaling.py; reports/RANKSTOP_INFERENCE_SCALING_PLAN_20261026.json; reports/RANKSTOP_INFERENCE_SCALING_SOURCE_REVIEW_20261026.json; reports/RANKSTOP_INFERENCE_SCALING_LAUNCH_20261026.json; reports/RANKSTOP_INFERENCE_SCALING_STARTUP_20261026.json; reports/RANKSTOP_INFERENCE_SCALING_NODE_20261026.json; runs/pipeline_tasks/rankstop_inference_scaling_server26_20261026/{status.json,run.log}; profile_output/rankstop_inference_scaling_server26_20261026/config.json and COMPLETE.json, all arm*/{COMPLETE.json,worker*.json}, cprofile_cv*/{COMPLETE.json,worker0.json,worker0.prof}; reports/rankstop_dev_plan_server26_20261026.json; source/model/tracker files from plan source map. CPU-only, do not load Torch/checkpoint or run NN/profiling again, change input/results or query active dev/resources/PIDs. May use readonly SSH alias2026 only for archived profile artifacts/checkpoint/manifest hashes and installed TRAIN first-frame provenance if needed (project /data2/gb/Dual-RGBT); no active dev progress. Independently recompute per-arm exact fixed work, timestamps/FPS, frame counts, GPU memory/utilization from saved samples, worker/native exit ownership, all predictions/routes/final-state summaries across arms, source/input binding and no hidden training/precision change. Distinguish state-hash comparisons/assertions vs independent neural replay; steady aggregate vs single-stream service vs end-to-end, profiled-arm exclusion, background GPU1 disclosure, repeat baseline drift, scope/selection limits. Check A-F and numeric evidence, native exit0. Write unique reports/RANKSTOP_INFERENCE_SCALING_RESULT_AUDIT_20261026.json/.md with blockers/limitations and trace. Do not edit HANDOFF/source/plan/Git. Return verdict and artifact paths.'''
def write_trace(name,value):
    (TRACE/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
write_trace('run.meta.json',dict(skill='experiment-audit',run_id=TRACE.name,generated_at=now,executor='codex',executor_model='UNATTESTED',executor_family='openai',review_independence='same-family',acceptance_status='provisional',project_dir=str(ROOT)))
write_trace('001-result-review.request.json',dict(call_number=1,purpose='result-review',timestamp=now,tool='spawn_agent',model='gpt-6-astra',reasoning_effort='max',actual_route='UNATTESTED',files_referenced=sorted(input_hashes),prompt=prompt))
(TRACE/'001-result-review.response.md').write_text(markdown,encoding='utf-8')
write_trace('001-result-review.meta.json',dict(call_number=1,purpose='result-review',timestamp=now,agent_id=agent,requested_model='gpt-6-astra',requested_reasoning_effort='max',actual_model='UNATTESTED',actual_reasoning_effort='UNATTESTED',actual_route='UNATTESTED',reviewer_family='openai',review_independence='same-family',acceptance_status='provisional',status='ok',verdict='WARN',blocking_issues=0))
event_path=ROOT/'.aris/meta/events.jsonl'
event_path.parent.mkdir(parents=True,exist_ok=True)
with event_path.open('a',encoding='utf-8') as handle:
    handle.write(json.dumps(dict(event='review_trace',skill='experiment-audit',purpose='result-review',agent_id=agent,trace_path=TRACE_REL+'/',status='ok',verdict='WARN'))+'\n')
manifest={}
for path in sorted(TRACE.iterdir()):
    if path.is_file() and path.name != 'TRACE_SHA256.json':
        manifest[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
for path in [REPORT_JSON,REPORT_MD]:
    manifest[str(path.relative_to(ROOT)).replace('\\','/')]=hashlib.sha256(path.read_bytes()).hexdigest()
write_trace('TRACE_SHA256.json',manifest)
print(json.dumps(dict(verdict=report['verdict'],status=report['status'],blocking_issues=0,checks=data['deterministic_checks'],reports=[str(REPORT_JSON),str(REPORT_MD)],trace=str(TRACE),report_hashes={k:v for k,v in manifest.items() if k.startswith('reports/')}),indent=2))
