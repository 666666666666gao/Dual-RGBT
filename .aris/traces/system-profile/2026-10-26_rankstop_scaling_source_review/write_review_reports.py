"""Persist the independent source review and preserved R1 finding."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r'C:\Users\gb\projects\Dual-RGBT')
TRACE = Path(__file__).resolve().parent
TRACE_REL = TRACE.relative_to(ROOT).as_posix()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


plan_path = ROOT / 'reports/RANKSTOP_INFERENCE_SCALING_PLAN_20261026.json'
plan = read(plan_path)
r1_plan = read(TRACE / 'R1_plan.json')
lifecycle = read(TRACE / 'R2_CPU_LIFECYCLE_FIXTURES.json')
parity = read(TRACE / 'R2_CPU_SOURCE_AND_PARITY_FIXTURES.json')
assert sha(plan_path) == lifecycle['plan_sha256'] == parity['plan_sha256'] == 'cb64752e50fea18da27eb9be6451cbcfd01c7561fdfcab6707409f6960651f6b'
assert sha(ROOT / 'scripts/profile_rankstop_scaling.py') == lifecycle['source_sha256'] == parity['source_sha256'] == '3ffa29862ec8e881bdd98b2dd0d8ee8f159160c39d4a9fcc6b90c08a99b95a56'
assert all(sha(ROOT / name) == digest for name, digest in plan['source_sha256'].items())
identity = dict(requested_model='gpt-6-astra', requested_reasoning_effort='max',
                actual_model='UNATTESTED', actual_reasoning_effort='UNATTESTED', actual_route='UNATTESTED',
                review_independence='same-family', acceptance_status='provisional',
                reviewer='fresh-context delegated Codex source reviewer',
                created_utc=datetime.now(timezone.utc).isoformat())
r1_issue = dict(
    id='R1-LIFECYCLE-001', severity='BLOCKING', status='CONFIRMED_ON_R1_RESOLVED_ON_R2',
    file='scripts/profile_rankstop_scaling.py', original_lines='111-136', revised_lines='121-142',
    finding='The non-daemon sampling thread and remaining workers were not cleaned up when readiness timed out, the barrier failed, or a joined worker had nonzero exit status. The unhandled failure could leave the launcher waiting instead of obtaining a native FAILED receipt.',
    evidence=TRACE_REL + '/R1_CPU_LIFECYCLE_FIXTURES.json',
    evidence_sha256=sha(TRACE / 'R1_CPU_LIFECYCLE_FIXTURES.json'),
    correction='The author added a narrow try/finally around spawn/readiness/barrier/join. It stops and joins the sampler, then terminates only still-live children created by this run_arm and joins them. Exceptions still propagate; no retry, fallback, or exception suppression.',
    recheck=TRACE_REL + '/R2_CPU_LIFECYCLE_FIXTURES.json',
    recheck_sha256=sha(TRACE / 'R2_CPU_LIFECYCLE_FIXTURES.json'))
r1 = dict(identity, status='SOURCE_FAIL', blocking_issues=1, round='R1',
          plan_sha256=sha(TRACE / 'R1_plan.json'), source_sha256=r1_plan['source_sha256'],
          script_sha256=sha(TRACE / 'R1_profile_rankstop_scaling.py'), issues=[r1_issue],
          scope='Preserved pre-fix source-only failure, superseded for launch admission by the separately bound final review. No NN/GPU/SSH activity during review.')

checked_files = [
    'scripts/profile_rankstop_scaling.py', 'reports/RANKSTOP_INFERENCE_SCALING_PLAN_20261026.json',
    'reports/RANKSTOP_SCALING_TRAIN_STREAMS_20261026.json', 'reports/rankstop_dev_plan_server26_20261026.json',
    'scripts/run_task.py', 'scripts/run_benchmark_tracking.py', 'scripts/run_rankstop_dev.py',
    'dual_rgbt_global/candidate_tracker.py', 'dual_rgbt_global/tracker.py',
    'dual_rgbt_global/candidate_policy.py', 'dual_rgbt/model.py', 'dual_rgbt/data.py',
    'scripts/train_global_evidence_policy.py', 'reports/RANKSTOP_COST0_TERMINAL_OBSERVATION_20261026.json',
    'runs/pipeline_tasks/rankstop_full_dev_cost0p0_server26_20261026/status.json',
    'reports/RANKSTOP_C882_COMPLETION_AUDIT_20261026.json',
    'reports/SERVER26_TRAIN_DEV_MIGRATION_20261026.json', 'reports/SERVER26_WEIGHT_MIGRATION_20261026.json',
    '.aris/compute/ssh.md', 'AGENTS.md',
]
notes = [
    dict(id='SCOPE-01', kind='CLARIFICATION', text='The task owner clarified GPU0-only applies to neural workers. GPU0 AND GPU1 memory/free/utilization sampling is intentional and disclosed so the continuing GPU1 dev workload is visible. The sampled fields contain no power or temperature.'),
    dict(id='MEASUREMENT-01', kind='LIMITATION', text='aggregate_fps is 1024/(latest worker finish minus earliest worker start). It includes eight initializations, 1016 tracking steps, image decoding, result-list construction and per-stream final-state hashing. Model construction, imports, warmups and post-timing model-state verification/report serialization are excluded.'),
    dict(id='MEASUREMENT-02', kind='LIMITATION', text='end_to_end_seconds/fps cover parent process-spawn to child-join, including per-worker load/warmup and post-timing verification/output. They exclude parent source/input hashing, filesystem prewarm, monitor setup/teardown and final report writing; they are not whole-command latency. load_seconds begins after imports and includes CPU snapshot plus warmup.'),
    dict(id='MEASUREMENT-03', kind='LIMITATION', text='cpu_children_seconds uses RUSAGE_CHILDREN. Reaped workers contribute their complete lifetime CPU use, including loading/warmup and final verification, while nvidia-smi subprocesses reaped during the interval also contribute. Do not describe this as inference-only CPU time or divide it by inference wall time to claim inference CPU utilization.'),
    dict(id='MEASUREMENT-04', kind='LIMITATION', text='One-second GPU samples observe aggregate device utilization and sampled memory peaks, not exact transient allocation peaks. The memory projection is an empirical admission estimate, not a proof against OOM. The fixed 1024-frame warm-filesystem workload, ordered arms and changing background host load do not establish full-dev/test FPS or a universal optimal worker/thread count.'),
    dict(id='MEASUREMENT-05', kind='LIMITATION', text='The two cProfile arms each use the first 128-frame TRAIN stream. They are excluded from throughput selection. Python cumulative timing can include CUDA waits and is not a CUDA kernel profile. Retain repeated N1 results and GPU1 samples when interpreting drift.'),
    dict(id='GATE-01', kind='RUNTIME_REQUIRED', text='Per-arm COMPLETE files are written before cross-arm parity is asserted. Accept the benchmark only with the top-level COMPLETE, all required parity flags and the native run_task exit0 receipt. Source PASS and CPU fake parity do not establish actual GPU parity or performance.'),
    dict(id='INPUT-01', kind='LIMITATION', text='14 of 26 bound inputs exist locally and all match. 12 remote-only inputs, including model binaries and the full train manifest, were not locally rehashed. Existing migration/training/terminal receipts bind them; main enforces every planned input/source SHA before and after actual execution. This reviewer made no remote query.'),
    dict(id='PROVENANCE-01', kind='INHERITED_WARNING', text='The original RankSTOP training launch still has no native terminal exit receipt; its actual training exit code remains unknown. The separately observed cost0 full97 dev native exit0 is valid and must not be substituted for the missing training receipt.'),
]
review = dict(identity,
    status='SOURCE_PASS', blocking_issues=0, round='R2_FINAL',
    plan='reports/RANKSTOP_INFERENCE_SCALING_PLAN_20261026.json', plan_sha256=sha(plan_path),
    source_sha256=plan['source_sha256'], script_sha256=sha(ROOT / 'scripts/profile_rankstop_scaling.py'),
    original48_unchanged=True, previous_plan_sha256=sha(ROOT / 'reports/rankstop_dev_plan_server26_20261026.json'),
    issues=[], resolved_issues=[r1_issue], notes=notes,
    reviewed_files_sha256={name: sha(ROOT / name) for name in checked_files},
    fixture_results=dict(lifecycle_cases=5, lifecycle_status=lifecycle['status'],
                         source_parity_check_groups=parity['check_count'], source_parity_status=parity['status'],
                         python37_source_ast_parses=49, details=parity['checks']),
    fixture_artifacts={name: sha(TRACE / name) for name in [
        'R1_cpu_lifecycle_fixtures.py', 'R1_CPU_LIFECYCLE_FIXTURES.json', 'R1_profile_rankstop_scaling.py', 'R1_plan.json',
        'R2_cpu_lifecycle_fixtures.py', 'R2_CPU_LIFECYCLE_FIXTURES.json', 'R2_profile_rankstop_scaling.py', 'R2_plan.json',
        'R2_cpu_source_and_parity_fixtures.py', 'R2_CPU_SOURCE_AND_PARITY_FIXTURES.json']},
    fixture_directory=TRACE_REL,
    input_verification=parity['input_facts'],
    conclusions=dict(
        fixed_workload='All10 arms use the same8 TRAIN streams x128 frames, one initial GT box per stream, sequential causal frames. Strided distribution gives every stream exactly once at N1/2/4/8; both OpenCV80/1 arms compare to the same first-N1 reference. Selected streams belong to frozen882 TRAIN and are disjoint from97 dev.',
        timing='Every worker loads and warms before a shared count+1 barrier. Timing starts after release and ends after CUDA synchronization. The slowest worker determines aggregate throughput. Native process exit codes are checked before accepting reports.',
        model_binding='Same original model/tracker builders and final checkpoint as existing dev; identical factory and protected pretrained checkpoint bindings. Strict state reload, eval/requires_grad false, no_grad, seed2027, torch threads1 and TF32 false. Model state is checked equal after workload. No training/optimizer/backward call or noninitial GT read is present on the executed path.',
        exact_parity='Predictions, all8 non-timing route fields, and SHA256 of final box/template/history bytes are compared across all10 arms. Actual tracker initialize resets frame/since_query/history/templates/window. frame is determined by128-frame workload; since_query is determined by saved routes; window is deterministic and never mutated on candidate track path.',
        lifecycle='R1 non-daemon sampler leak is fixed and CPU-reproduced failures now unwind with owned worker cleanup. Successful workers are not terminated and success timing remains unchanged.',
        memory_admission='Each arm checks physicalGPU0 used<500MiB before spawning. From the second arm, maximum prior sampled GPU0 peak per worker times proposed worker count must be below GPU0 total minus1024MiB. CPU fixtures verify N8 projection and reject equality/excess at the headroom boundary. GPU1 is sampled only; never selected for NN by this runner.',
        observer_effect='Standalone runner only; inherited48 source hashes unchanged. One-second external sampling, per-stream state hashing and output accumulation have disclosed observer cost. Separate cProfile runs are excluded from all10-arm throughput selection.'),
    execution_scope=dict(source_only=True, cpu_fixtures_only=True, ssh_calls=0, runtime_resource_queries=0,
                         torch_imports=0, gpu_or_neural_execution=0, job_launches=0, job_stops=0,
                         original_source_plan_handoff_or_git_changes=False),
    instrumentation_changelog=[
        dict(file='scripts/profile_rankstop_scaling.py', change='Standalone instrumentation created by the author, reviewed only by this agent; R2 author change limited to owned-child cleanup.'),
        dict(file=TRACE_REL, change='Reviewer-created CPU fake fixtures, exact source/plan snapshots and receipts; original source not edited.'),
        dict(file='reports/RANKSTOP_INFERENCE_SCALING_SOURCE_REVIEW_R1_20261026.json/.md', change='Reviewer-created preserved failed source review.'),
        dict(file='reports/RANKSTOP_INFERENCE_SCALING_SOURCE_REVIEW_20261026.json/.md', change='Reviewer-created final source-only review.')],
    review_harness_note='One initial R2 fixture attempt had a harness-only IndexError in the simulated second-process spawn selector. The selector was corrected, then all5 cases passed. This was not a source/benchmark failure and caused no runtime or GPU activity.')

r1_md = f'''R1 SOURCE_FAIL — 1 blocking issue, preserved after correction.

Plan SHA256: `{r1['plan_sha256']}`.
Runner SHA256: `{r1['script_sha256']}`.

The non-daemon sampler was left running when readiness timed out, the barrier failed, or the first joined worker exited1. Remaining workers were not reaped. AST-isolated CPU fixtures confirmed all three paths while the success fixture passed. This could prevent run_task from recording a native failure.

Evidence: `{TRACE_REL}/R1_CPU_LIFECYCLE_FIXTURES.json`; exact source and plan snapshots are retained alongside it. The final separately bound review verifies the narrow author fix. No GPU/NN/SSH action was performed.

Requested reviewer: gpt-6-astra / max. Actual route: UNATTESTED. Independent context, same-family, provisional acceptance.
'''
final_md = f'''SOURCE_PASS — 0 blocking issues for the exact reviewed bytes.

Plan SHA256: `{review['plan_sha256']}`.
Runner SHA256: `{review['script_sha256']}`.

All49 bound source files match; the inherited48 are unchanged. All49 parse as Python3.7. The fixed workload is eight frozen TRAIN streams x128 frames per arm, giving eight initializations and1016 causal tracking steps. The streams are disjoint from97 dev. Ten arms compare OpenCV80/1 at N1/2/4/8/repeat1 against one shared prediction/route/state reference. Only the initial GT box enters each tracker.

The final checkpoint, factory and pretrained weight bindings match the current dev plan. The existing model/tracker builders, strict reload, eval/freeze, seed2027 and no-grad inference are retained. Model-state equality is checked after execution.14 locally present input bindings match;12 remote-only inputs were not rehashed in this local source review and remain subject to the executable SHA gates.

The author fixed the confirmed R1 failure-cleanup defect.5 CPU lifecycle cases now pass: success, readiness timeout, broken barrier, worker exit1 and spawn failure. Every started owned worker is reaped; the sampler is stopped/joined; original failures propagate. Success work and timing are unchanged.15 additional grouped source/parity/admission checks pass. R1 SOURCE_FAIL, the actual counterexample receipt and exact snapshots remain preserved in `{TRACE_REL}`.

Neural workers bind physicalGPU0 only. GPU0/1 memory/free/utilization sampling is intentionally disclosed to record background GPU1 dev load; no power/temperature is queried. Before each arm GPU0 must be below500MiB. Prior observed peak-per-worker memory projects the next count, includingN8, with1024MiB headroom. This sampled estimate does not prove that transient peaks cannot exceed the projection.

Aggregate FPS is1024 divided by the interval from earliest worker start to latest finish. It includes decoding, initialization, tracking, result accumulation and final-state hashing, while excluding model loading/warmup and post-timing weight verification/report serialization. The field end_to_end covers process-spawn to child-join, not all parent preparation. cpu_children_seconds includes reaped workers' entire lifetime CPU use and some sampler subprocess CPU; it is not inference-only CPU time. Separate cProfile runs use one128-frame stream each and do not enter throughput selection.

Only a top-level COMPLETE plus native exit0 and all actual parity gates supports accepting runtime results; per-arm COMPLETE is written before cross-arm parity. RepeatedN1 and background samples must be considered when interpreting drift. This short warmed workload is not full-dev/test FPS or accuracy. Original training native exit status remains unknown; completed cost0 dev exit0 does not repair that provenance gap.

No SSH, runtime/resource query, Torch import, GPU/NN run, job launch/stop, original source/plan/HANDOFF edit or Git mutation was performed by this reviewer. Review artifacts and CPU fixtures were the only files created.

Requested reviewer: gpt-6-astra / max. Actual model, effort and route: UNATTESTED. Independent context, same-family; acceptance is provisional.
'''

for name, value in [('RANKSTOP_INFERENCE_SCALING_SOURCE_REVIEW_R1_20261026.json', r1),
                    ('RANKSTOP_INFERENCE_SCALING_SOURCE_REVIEW_20261026.json', review)]:
    path = ROOT / 'reports' / name
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
for name, text in [('RANKSTOP_INFERENCE_SCALING_SOURCE_REVIEW_R1_20261026.md', r1_md),
                   ('RANKSTOP_INFERENCE_SCALING_SOURCE_REVIEW_20261026.md', final_md)]:
    path = ROOT / 'reports' / name
    assert not path.exists(), path
    path.write_text(text, encoding='utf-8')
print(json.dumps(dict(status=review['status'], blocking_issues=review['blocking_issues'],
    report_sha256=sha(ROOT / 'reports/RANKSTOP_INFERENCE_SCALING_SOURCE_REVIEW_20261026.json'),
    markdown_sha256=sha(ROOT / 'reports/RANKSTOP_INFERENCE_SCALING_SOURCE_REVIEW_20261026.md'),
    r1_report_sha256=sha(ROOT / 'reports/RANKSTOP_INFERENCE_SCALING_SOURCE_REVIEW_R1_20261026.json'),
    plan_sha256=review['plan_sha256'], script_sha256=review['script_sha256']), indent=2))
