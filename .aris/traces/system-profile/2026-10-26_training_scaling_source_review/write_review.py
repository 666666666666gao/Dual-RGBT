import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
PLAN = 'reports/TRAIN_CONCURRENCY_PLAN_20261026.json'
SOURCE = 'scripts/profile_candidate_training_scaling.py'
REPORT = 'reports/TRAIN_CONCURRENCY_SOURCE_REVIEW_20261026'
AGENT = '/root/review_joint_quality_train_20261026'
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = json.loads((ROOT / PLAN).read_text())
cpu = json.loads((TRACE / 'cpu_verification.json').read_text())
assert cpu['status'] == 'CPU_SOURCE_FIXTURES_PASS'
assert sha(ROOT / PLAN) == cpu['plan_sha256'] and sha(ROOT / SOURCE) == cpu['new_source_sha256']
assert all(sha(ROOT / n) == h for n, h in plan['source_sha256'].items())
checks = {
    'binding_and_isolation': dict(status='PASS', evidence=[SOURCE + ':199', PLAN],
        details='Exact requested plan/source hashes; all51 source hashes/ASTs pass. Original active50 unchanged;28 inputs covered by16 current local hashes and12 unchanged remote-only bindings from sealed prior verification. Source rehashes every dependency before/after profiling. ProfileNN is fixed toGPU1; GPU0 is observed only at arm boundaries; noGPU2/3,temperature/power or checkpoint-writing path.'),
    'import_and_spawn': dict(status='PASS_CPU', evidence=[SOURCE + ':13', SOURCE + ':39', SOURCE + ':144', 'cpu_verification.json'],
        details='The direct script --help invocation succeeds without importing Torch. CPython direct-script search path includes scripts; prepending ROOT retains that path and spawn inherits it. Real spawned CPU workers resolve all three bare trainer helper modules to the intended scripts directory. Existing formal admission/startup receipts confirm the unchanged helper dependency imports in the target environment. No concrete import-path blocker found for the intended direct-file invocation.'),
    'native_identity_and_existing_jobs': dict(status='PASS_SOURCE', evidence=[SOURCE + ':207', SOURCE + ':210', 'reports/JOINT_GAINPAIR_TRAIN_EXECUTION_20261026.json', 'reports/JOINT_GAINPAIR_TRAIN_STARTUP_20261026.json'],
        details='Profile binds its wrapper parent PID,GPU1 and exact command; it verifies the expected existingGPU1 wrapper1054384,child1054385 and /proc argv before creating output/workers. Archived local receipts show actual4-update native0/reload/freeze/replay admission and150 finite formal startup updates. This review does not freshly assert current process activity; runtime must pass the source identity checks.'),
    'native_training_parity_and_reset': dict(status='PASS_SOURCE', evidence=[SOURCE + ':57', SOURCE + ':79', SOURCE + ':90', SOURCE + ':105'],
        details='Every stream strictly resets B6250 model weights,constructs fresh AdamW at1e-5/1e-4,tracker and NumPy sampler,then uses the native causal tracker,batch,teacher,selected replay,loss and optimizer helper. Current GT enters after tracking; teacher state-preservation assertions remain. Requires-grad prefixes/eval mode match formal. No new teacher/model call,parameter or test label is introduced.'),
    'actual_updates_and_work_coverage': dict(status='PASS_SOURCE_AND_CPU', evidence=[SOURCE + ':77', SOURCE + ':106', SOURCE + ':112', SOURCE + ':178'],
        details='Eight fixedTRAIN streams each require32 real step_optimizer returns and finite losses; even strided assignment across1/2/4/8 workers is checked for exact unique coverage. Each arm totals256 updates. Before work, source checks each sequence has enough valid stride5 frames. End checks require251 frozen/14 changed states,no frozen gradients,optimizer step32 and worker native exit0. CPU allocation/IPC fixtures reproduce8 streams/256 mock updates at1/2/4/8/1; they do not execute neural training.'),
    'barriers_ipc_and_failure_cleanup': dict(status='PASS_CPU', evidence=[SOURCE + ':71', SOURCE + ':137', SOURCE + ':148', SOURCE + ':166'],
        details='Workers sendREADY before release andRESULT before finish. finally releases both events,joins owned workers,terminates only remaining owned workers and closes/joins the queue. Five normal spawn arms pass. BadREADY,exit7 during work and deliberately held worker each fail as intended,release both barriers and leave no owned worker alive. Fixture-only wait caps5s/.25s exercise production180/900s queue and30s join branches. The cleanup addresses the recorded priorR1 stranded-worker failure; no retry/fallback is added.'),
    'timing_and_reporting': dict(status='PASS_WITH_SCOPE_LIMITS', evidence=[SOURCE + ':70', SOURCE + ':120', SOURCE + ':180'],
        details='Model construction/initial checkpoint load occur beforeREADY. Arm training wall starts immediately before release and ends at maximum synchronized worker completion,including per-stream reset,tracking/teacher/backward/AdamW and step JSON serialization. CPU frozen-state validation is after worker completion timestamps and excluded from training_wall; training_and_validation_wall is separately reported. spawn_to_join includes startup/cleanup. CPU fixtures verify clock ordering. Main summary processes means additional profile workers; plan/arm names explicitly identify totalGPU1 NN counts2/3/5/9/2.'),
    'hash_rng_and_slowdown_limits': dict(status='QUALIFIED_PASS', evidence=[SOURCE + ':53', SOURCE + ':86', SOURCE + ':113', SOURCE + ':27'],
        details='Full step hashes include runtime seconds; prediction/route hashes explicitly remove seconds. No bitwise-equality assumption gates results. Torch seed is per worker and NumPy sampler reset per stream; nondeterministic training/route differences remain possible. Existing training is observed only before/after arms via complete CSV rows and logged updates. CSV buffering/flushes coarsen those deltas; they are not exact per-step timings. Repeated additional1 arm checks cache/load drift; no full882 training or inference-FPS acceleration claim follows.')
}
report = dict(audit_skills=['system-profile', 'experiment-audit'], status='SOURCE_PASS', verdict='PASS',
    blocking_issues=0, blockers=[], generated_at=NOW, date='2026-10-26', agent_id=AGENT,
    reviewer_model_requested='gpt-6-astra', reviewer_reasoning_requested='max',
    reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED', actual_route_attestation='UNATTESTED',
    reviewer_family='openai', review_independence='same-family', acceptance_status='provisional',
    plan_path=PLAN, plan_sha256=cpu['plan_sha256'], source_sha256=plan['source_sha256'],
    input_sha256=plan['input_sha256'], new_source_path=SOURCE, new_source_sha256=cpu['new_source_sha256'],
    checks=checks, runtime_readiness=dict(source_ready=True, concrete_worker_import_blocker=False,
        concrete_ipc_deadlock_blocker=False, actual_nn_profile_executed=False,
        first_additional1_arm_required_before_larger_arms=True,
        note='Ready for the planned first actual NN/process arm after live identity checks. CPU fixtures do not attest GPU memory capacity or neural runtime success at larger concurrency.'),
    workload=dict(fixed_train_streams=8, updates_per_stream=32, updates_per_arm=256,
        additional_workers=[1,2,4,8,1], total_gpu1_nn_processes=[2,3,5,9,2],
        initialized_from='B6250 each stream', saved_checkpoints=0, total_parameters=103218887),
    cpu_validation=dict(source_hashes=51, inherited_source_hashes=50, input_bindings=28,
        current_local_inputs=16, reused_remote_input_bindings=12, normal_spawn_arms=5,
        injected_failure_cases=3, no_surviving_owned_workers=True,
        fixture_queue_cap_seconds=5, fixture_join_cap_seconds=.25,
        torch_import=False, neural_execution=False, remote_queries=False),
    instrumentation_changelog=[
        dict(file=SOURCE, change='standalone profiler created by executor', lines='21-231',
            details='Resource/progress observations, native training workers, barriers/timers, JSON outputs and owned-worker cleanup; no inline edits to active50.'),
        dict(file=PLAN, change='profile configuration created by executor', details='Fixed streams,concurrency arms,dependency hashes,existing-job identities and scope.'),
        dict(file=REPORT + '.json/.md', change='source-review reports created by reviewer'),
        dict(file=str(TRACE.relative_to(ROOT)).replace('\\','/'), change='CPU-only review fixtures and trace created by reviewer')],
    operations=dict(audited_source_edits=False, audited_plan_edits=False, active_remote_queries=False,
        gpu_or_neural_execution=False, active_job_changes=False, only_cpu_fixture_workers_created=True),
    trace_path=str(TRACE.relative_to(ROOT)).replace('\\','/'))
markdown = f'''# Training concurrency source review

**SOURCE_PASS; 0 blocking issues.** No concrete worker-import or IPC deadlock blocker found for the intended direct-script+spawn invocation. Requested reviewer:gpt-6-astra/max; actual route:UNATTESTED. Same-family/provisional.

Plan SHA:`{cpu['plan_sha256']}`. New source SHA:`{cpu['new_source_sha256']}`. All51 source hashes/ASTs pass; the active50 are unchanged. All28 input bindings pass using16 current local hashes and12 unchanged remote-only bindings from sealed prior verification. No active remote query or neural execution was performed.

The script directory remains on sys.path when the profiler prepends ROOT; multiprocessing spawn inherits it. Direct-script --help succeeds, and real spawned CPU workers resolve the three native trainer helpers correctly. The target environment's archived actual sanity/formal-startup receipts establish those unchanged helper imports and native4-update admission. First additional1 remains the profiler's actual NN/process admission before larger arms.

Each of8 fixedTRAIN streams reloads originalB6250 and creates a fresh optimizer/tracker/NumPy sampler. The native causal track,teacher,replay,loss and step_optimizer paths are reused. Current GT follows tracking, teacher state-preservation assertions remain, and32 finite actual optimizer updates are required per stream. Stream assignment and final unique coverage preserve256 updates per arm. Freeze/changed/optimizer checks and child exit0 are required. GPU1 totals are2/3/5/9/2 NN processes including its existing formal task; GPU0 is observed only, andGPU2/3 are not used.

Five normal CPU spawn arms at1/2/4/8/1 workers passed, each covering8 streams/256 mock updates. Three injected failures—badREADY,worker exit7 during work and a deliberately held worker—each propagated failure, released both barriers and left no owned worker alive. The held worker exercised terminate/join. CPU fixture waits were capped at5s/.25s; production uses180/900s message waits and30s joins. This tests control flow, not GPU execution. Source cleanup is limited to children created by that arm and addresses the recorded priorR1 stranded-worker failure; no retries or fallback appear.

Timing is separated correctly. Model construction/initial load are beforeREADY. Training wall spans release through the latest synchronized worker completion, including stream resets,tracking,teacher/backward/AdamW and step JSON writing. CPU state validation occurs after those completion timestamps and is excluded from training wall; validation-inclusive and spawn-to-join timings are separate. Full step hashes include timing; the prediction/route hash removes route.seconds. Torch seeding is per worker, sampler reset per stream, and the plan makes no training bitwise-equality guarantee.

Existing-run observations occur only at arm boundaries and record logged CSV updates. Buffering/flushes limit their time resolution; they are not exact per-step timers. Repeated additional1 checks cache/load drift. Prefix throughput is not full882 training throughput, inferenceFPS or a proven causal acceleration; altered learned routes can change conditional work.

The profile validates its current native wrapper and the expected existingGPU1 wrapper/child/argv before work. Recorded local M0/native0/reload/freeze/replay and150-finite-update startup evidence was inspected without querying active jobs. Live identity checks and actual first-arm NN success remain required at execution; larger-arm GPU memory capacity is not established by CPU fixtures.

| File | Change | Instrumentation |
|---|---|---|
| `{SOURCE}` | Standalone file created by executor; reviewed only | Resource/progress observations,training workers,timers,JSON outputs,owned-child cleanup; lines21-231 |
| `{PLAN}` | Configuration created by executor; reviewed only | Fixed streams,arms,hashes,job identities and scope |
| Active50 sources | Unchanged | No inline instrumentation |
| `{REPORT}.json/.md` and `{report['trace_path']}/` | Created by reviewer | Source findings and CPU-only fixtures |

Detailed tests:`{report['trace_path']}/cpu_verification.json`. Source review is complete; the planned first actual arm is ready to attempt under its runtime identity checks.
'''
with (ROOT / (REPORT + '.json')).open('x', encoding='utf-8', newline='\n') as f:
    json.dump(report, f, indent=2); f.write('\n')
with (ROOT / (REPORT + '.md')).open('x', encoding='utf-8', newline='\n') as f:
    f.write(markdown)
prompt = '''New explicit user instruction: lowVRAM -> may run more independent tasks/card; useGPU2/3 only after their current training complete. Actual read-only resource report reports/TRAIN_CONCURRENCY_RESOURCE_SNAPSHOT_20261026.json showsGPU0/1~2.4GB/11%util,80CPU~8%busy,RAM99GB available; GPU2/3 stillReIDNN (untouched). Need fresh source review under system-profile/experiment-audit before measured training scaling. Read scripts/profile_candidate_training_scaling.py SHAc76b74b3efd4828f909a63842ec94c33e9da6eb52c65a9ce674fdc80baaf34fa; reports/TRAIN_CONCURRENCY_PLAN_20261026.json SHA0f9a31ac23c4a3c1450ef683e3c386e2f138756985ae4a71864a116543c03ce5; all referenced51source/28input bindings and existing gainpair review29f167. This new219+line standalone script reuses native tracker/teacher/causal_batch/selected_target/controller_loss/step_optimizer, no edits to sealedactive50. 8fixedTRAIN streams x32 actual updates each, same256 per arm, reset B6250+optimizer eachstream. Additional1/2/4/8/1 profileworkers beside activeGPU1 formal =2/3/5/9 TOTAL card1 processes, clearly labeled. ExistingGPU0/1 formal unchanged; before/after eacharm CSVdelta only torecordslowdown. Existing knownGPU1 wrapper1054384 child1054385; formalM0 actualnative0/reload/frozen/replayPASS, startup150finite (reports/JOINT_GAINPAIR_TRAIN_EXECUTION_20261026.json and STARTUP). No GPU2/3 use, no temperatures/power, no checkpoint saved/test data. Scope shortprefix throughput, repeatedbaseline/cache/load and RNG/hash non-equality disclosed, no fulltraining/FPS acceleration claim. Process/barrier finallycleanup is justified by observed earlier profileR1 stranded workers, no exception/fallback/retry. Audit importability from script+spawn, causal parity/freeze/optimizer reset, finite/actualupdates, same work coverage, timing/excludedvalidation, JSONroute seconds exclusion, nativeID, IPC normal/failure cleanup with boundedCPU fixtures. Write reports/TRAIN_CONCURRENCY_SOURCE_REVIEW_20261026.json/.md statusSOURCE_PASS/0 or exact blockers; DO NOT edit source/plan, no active remote queries/NN launch. Prioritize concrete correctness and necessary checks; no optional big reports. Please report whether runtime ready worker/import/deadlock risk exists.'''
def save(name, data):
    (TRACE / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
metadata = dict(model='UNATTESTED', requested_model='gpt-6-astra', reasoning_effort='UNATTESTED',
    requested_reasoning_effort='max', review_independence='same-family', acceptance_status='provisional')
save('run.meta.json', dict(skill='system-profile', companion_skill='experiment-audit',
    timestamp=NOW, run_id='2026-10-26_training_scaling_source_review', project_dir=str(ROOT), **metadata))
save('001-source.request.json', dict(call_number=1, purpose='training-concurrency-source-review',
    timestamp=NOW, tool='followup_task', prompt=prompt, **metadata))
(TRACE / '001-source.response.md').write_text(markdown, encoding='utf-8')
save('001-source.meta.json', dict(call_number=1, purpose='training-concurrency-source-review',
    timestamp=NOW, agent_id=AGENT, status='ok', verdict='SOURCE_PASS', blocking_issues=0, **metadata))
outputs = {name: sha(ROOT / name) for name in (REPORT + '.json', REPORT + '.md')}
save('output_seals.json', dict(timestamp=NOW, report_sha256=outputs,
    verifier_sha256=sha(TRACE / 'verify_cpu.py'), cpu_result_sha256=sha(TRACE / 'cpu_verification.json')))
with (ROOT / '.aris/meta/events.jsonl').open('a', encoding='utf-8') as f:
    f.write(json.dumps(dict(event='review_trace', skill='system-profile', purpose='training-concurrency-source-review',
        agent_id=AGENT, trace_path=report['trace_path'], status='ok')) + '\n')
print(json.dumps(dict(status='SOURCE_PASS', blocking_issues=0, outputs=outputs), indent=2))
