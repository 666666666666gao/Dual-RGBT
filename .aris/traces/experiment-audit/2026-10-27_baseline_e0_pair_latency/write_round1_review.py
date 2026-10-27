"""Persist reviewer findings only; all project/source/result inputs remain read-only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

TRACE = Path(__file__).resolve().parent
ROOT = TRACE.parents[3]
PLAN_NAME = 'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json'
SOURCE_NAME = 'tools/profile_baseline_e0_pair.py'
PLAN_SHA = 'b484194f296dfd5a9fdebe98fe5a4b0eda6ae92250cfafe0cb1f00fb9c32c4d6'
SOURCE_SHA = '0d737c4a83a7f841657b5282921b3fbb2fd82bd61dade2f032c590839fde228b'


def sha(data):
    return hashlib.sha256(data).hexdigest()


plan_bytes = (ROOT / PLAN_NAME).read_bytes()
source_bytes = (ROOT / SOURCE_NAME).read_bytes()
assert sha(plan_bytes) == PLAN_SHA and sha(source_bytes) == SOURCE_SHA
plan = json.loads(plan_bytes)
deterministic = json.loads((TRACE / 'deterministic_source_checks.json').read_bytes())
assert deterministic['plan_sha256'] == PLAN_SHA
assert not deterministic['hash_mismatches'] and not deterministic['syntax_failures']
old_model = subprocess.check_output(['git', 'show', 'dc79171^:dual_rgbt/model.py'], cwd=ROOT)
current_model = (ROOT / 'dual_rgbt/model.py').read_bytes()
expected_old = b"def __init__(self, checkpoint='/home/gaob/CADTrack_ep0014.pth.tar'):"
expected_new = b"def __init__(self, checkpoint=Path(__file__).resolve().parents[1] / 'checkpoints/CADTrack_ep0014.pth.tar'):"
assert sha(old_model) == '434da9f818158c16bd7563e93ef44c65b2b726d881f35f94284fb6e8f9966e50'
assert old_model.replace(expected_old, expected_new) == current_model
(TRACE / 'round1.plan.json').write_bytes(plan_bytes)
(TRACE / 'round1.profile_baseline_e0_pair.py').write_bytes(source_bytes)
(TRACE / 'historical_model_path_only.diff').write_bytes(subprocess.check_output(
    ['git', 'diff', 'dc79171^', 'dc79171', '--', 'dual_rgbt/model.py'], cwd=ROOT))

issues = [
    dict(id='F1', severity='P1', status='FAIL', title='Profiler has no native-wrapper admission binding',
         evidence=['tools/profile_baseline_e0_pair.py:24-35', 'tools/profile_baseline_e0_pair.py:42-66',
                   'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json:128-160',
                   'scripts/run_benchmark_tracking.py:160-165', 'scripts/run_task_four_gpu.py:18-31'],
         details='The profiler verifies the two training wrappers, but never reads or binds its own native task status, expected executable/command, RUNNING state or parent PID. Its plan also has no native task identity. Therefore direct or unregistered invocation reaches SIGSTOP with the same environment values. The reviewed four-GPU wrapper supports persistent native exit receipts, but nothing in these artifacts requires the profiler to run under it. This fails the requested native-profile admission requirement; it is not a claim that an unregistered launch has already occurred.',
         required_change='Before any SIGSTOP, bind this profiler to a named native task, exact child command, physical GPU and wrapper parent PID/RUNNING state. Use the detached registered wrapper for execution and preserve its actual exit receipt. No neural source change is needed.'),
    dict(id='F2', severity='P2', status='FAIL', title='Six imported package initializers are not hash-bound',
         evidence=['tools/profile_baseline_e0_pair.py:19-21', 'tools/profile_baseline_e0_pair.py:33',
                   'tools/profile_baseline_e0_pair.py:69-81', 'lib/models/__init__.py:1',
                   'lib/utils/__init__.py:1', 'lib/test/utils/params.py:1',
                   'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json:13-127',
                   '.aris/traces/experiment-audit/2026-10-27_baseline_e0_pair_latency/deterministic_source_checks.json'],
         details='The 113 registered hashes match, but the actual package-import closure includes six omitted __init__.py files. lib/models/__init__.py executes the model-factory import and lib/utils/__init__.py exports TensorDict/TensorList used by TrackerParams; these are real imported paths. The other omitted initializers are empty or docstring-only today. Thus the existing runtime source guard does not cover the full imported local source surface.',
         paths=[row['path'] for row in deterministic['unbound_imported_sources']],
         required_change='Add the six exact current initializer hashes to the plan and review payload; do not edit their source or add unrelated compatibility logic.')]

checks = {
    'original_baseline_identity': dict(status='PASS', evidence=[
        'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json:5-11', 'tools/profile_baseline_e0_pair.py:36-37',
        'tools/profile_baseline_e0_pair.py:69-89', 'lib/test/tracker/cadtrack.py:23-38',
        'lib/models/cadtrack/cadtrack.py:101-156', 'lib/models/cadtrack/vit_care.py:45-63',
        'lib/models/cadtrack/vit_care.py:98-146'],
        details='Protected SOTep14 checkpoint cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731 is hashed before pause. Original CADTrack builds with training=False, strictly loads net, evaluates both RGB/TIR streams with the unchanged stripped shared backbone, decode fusion and original head. Baseline source/config hashes match the prior source/result audit. Total parameter enumeration is asserted as103138437; this reviewer did not instantiate the model or open a checkpoint.'),
    'historical_e0_identity': dict(status='PASS', evidence=[
        'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json:6-9', 'reports/PAIRED_LATENCY_E0_RESTORE_20261027.json:2-7',
        'reports/ALL882_FULL_TEST_RESULTS.json:2-9', 'tools/profile_baseline_e0_pair.py:81-88',
        'dual_rgbt/model.py:34-48', 'scripts/run_benchmark_tracking.py:21-42'],
        details='The selected inactive all882 E0 checkpoint ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c matches the historical full-test identity. Restore metadata names the retired server25 archive, not an active32 trainer output. DualRGBT first strictly loads the protected baseline, then the profiler strictly loads historical E0 net. The only source difference from the historical model hash434da9f... is the default checkpoint path relocation at line34; byte replacement reproduces current c7fae6f... exactly. The source adds Preview14336 + quality4865 + gain216577 =235778 parameters to the baseline103138437, giving expected103374215; token_cost is a buffer. Both runtime counts are asserted. No checkpoint bytes were read by this audit.'),
    'inference_only_no_accuracy': dict(status='PASS', evidence=[
        'tools/profile_baseline_e0_pair.py:80-86', 'tools/profile_baseline_e0_pair.py:95-127',
        'scripts/run_benchmark_tracking.py:40-42', 'dual_rgbt/tracker.py:16-31',
        'dual_rgbt/model.py:181-195', 'lib/test/tracker/cadtrack.py:67-68',
        'lib/test/tracker/cadtrack.py:87-99'],
        details='Both networks are eval/frozen; E0 initialize/track/predict and baseline forward use no_grad. No optimizer/backward, accuracy scorer, later GT box, checkpoint write or trainable-parameter output occurs. Imports expose training functions, but those functions/main entry points are not called. Only boxes[0] initializes each sequence. Measured records contain timings, phase identity and an E0 auxiliary-call count.'),
    'e0_crop_binding': dict(status='PASS', evidence=[
        'tools/profile_baseline_e0_pair.py:19-21', 'tools/profile_baseline_e0_pair.py:69-92',
        'scripts/run_benchmark_tracking.py:16-27', 'scripts/run_benchmark_tracking.py:32-35',
        'scripts/train_global_evidence_policy.py:16-22', 'dual_rgbt_global/policy.py:6-7',
        'dual_rgbt_global/tracker.py:7-13', 'dual_rgbt_global/training.py:10-15',
        'dual_rgbt/tracker.py:6', 'scripts/profile_crop_api.py:19-25',
        'scripts/profile_interior_crops.py:24-47'],
        details='The pre-build imports load global/evidence policy and tracker modules but not dual_rgbt.tracker. build_model(e0,None) sets dual_rgbt.data.crop_pair=six_channel_crop before build_tracker first imports dual_rgbt.tracker. Its by-value crop_pair binding is therefore the patched function. The same build_model/build_tracker factories are used by the official benchmark. Baseline uses its original sample_target, unaffected by that alias. This conclusion is source-traced without project imports or NN execution.'),
    'causal_resets_and_sample': dict(status='PASS', evidence=[
        'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json:129-159', 'tools/profile_baseline_e0_pair.py:38-41',
        'tools/profile_baseline_e0_pair.py:90-119', 'tools/profile_baseline_e0_pair.py:136',
        'lib/test/tracker/cadtrack.py:60-74', 'lib/test/tracker/cadtrack.py:110-122',
        'dual_rgbt/tracker.py:17-28', 'dual_rgbt/tracker.py:34-61'],
        details='The exact order is baseline_before, E0cost0, E0cost0.05, baseline_after. Each phase reinitializes each of bluegirl/bike2/comingboy\'shead at TRAIN frame0, runs causal warm frames1..20, then measured21..220:600 rows per phase,2400 overall. Baseline resets state, templates, track_query_before and frame_id; E0 resets state/templates/history/window/frame/since_query. Full template updates and history changes remain active. Manifest is unavailable locally, so actual images/lengths are runtime requirements; the code verifies the bound manifest before pausing.'),
    'timing_and_precision': dict(status='PASS', evidence=[
        'tools/profile_baseline_e0_pair.py:56-59', 'tools/profile_baseline_e0_pair.py:95-127',
        'dual_rgbt/data.py:15-19', 'lib/test/tracker/data_utils.py:23-27',
        'dual_rgbt/tracker.py:32-63'],
        details='CUDA synchronization precedes frame start and follows tracker completion. End-to-end includes both RGB/TIR file decode, concat and service. Service includes crop/preprocess/transfer/forward/head/template update/state work and its synchronization; it excludes decode. Initialization including frame0 decode is separately timed and excluded from FPS; model construction/load and warmup are excluded. FPS=count/sum seconds, mean and P95 use raw durations. FP32 is used without autocast/half and TF32 is disabled. Actual device/latency remain unmeasured by this source review.'),
    'owned_training_groups': dict(status='PASS', evidence=[
        'tools/profile_baseline_e0_pair.py:36-52', 'tools/launch_server25_training.py:62-86',
        'tools/launch_server25_training.py:103-109', 'scripts/run_task_four_gpu.py:20-28',
        'reports/SERVER25_REMAINING_GRID_EXECUTION_20261027.json:1238-1242',
        'reports/SERVER25_REMAINING_GRID_EXECUTION_20261027.json:2389-2427',
        'reports/SERVER25_REMAINING_GRID_EXECUTION_20261027.json:2471-2475',
        'reports/SERVER25_REMAINING_GRID_EXECUTION_20261027.json:3622-3660'],
        details='Hash-bound execution metadata selects exactly two GPU0 cases, native wrappers223759/223760 and historically recorded sole children223766/223768. Runtime startup checks RUNNING/PID, exact wrapper and child cmdlines, CUDA_VISIBLE_DEVICES, bound run config and finite logged losses. Parent PGIDs must equal own PIDs and UIDs match current UID; both children must be in the admitted groups with the same UID. GPU0 UUID compute-client PIDs must exactly equal the live children. Launchers use start_new_session and wrapper subprocess inherits the group. Only these two group IDs are signaled; no other GPU/project/process is selected. No current liveness was queried by the reviewer.'),
    'finally_resume': dict(status='PASS', evidence=[
        'tools/profile_baseline_e0_pair.py:64-68', 'tools/profile_baseline_e0_pair.py:129-135',
        'reports/CADTRACK_BASELINE_LATENCY_SOURCE_REVIEW_20261027.json:178-184',
        'reports/CADTRACK_BASELINE_LATENCY_RESULT_AUDIT_20261027.json:143-158'],
        details='SIGSTOP and all baseline/E0 imports/loads/measurement are inside try; finally sends SIGCONT to both admitted groups, reads all four states, saves resume.json and asserts none stopped. Ordinary import, load, forward and assertion exceptions reach this block; the earlier baseline import failure and resume receipt are retained evidence for this lifecycle. This is not an uncatchable-kill recovery guarantee. Actual paired-run resume and resumed update progress still require native post-run receipts. F1 separately blocks admission of the profiler itself.'),
    'native_profile_admission': dict(status='FAIL', issue='F1'),
    'source_hash_and_compile': dict(status='PASS', evidence=[
        '.aris/traces/experiment-audit/2026-10-27_baseline_e0_pair_latency/deterministic_source_checks.json'],
        details='All113 exact hashes match.112 Python sources pass stdlib ast.parse(feature_version=(3,7)) and compile under installed CPython3.13.12 without imports or pyc writes. The inherited112 baseline-plan entries are unchanged; the only new registered source is the paired profiler. Generic python alias failed before execution with No pyvenv.cfg; the already installed uv-managed interpreter was used. No environment was changed.'),
    'import_dependency_coverage': dict(status='FAIL', issue='F2'),
    'claim_scope': dict(status='WARN', evidence=[
        'tools/profile_baseline_e0_pair.py:142-146', 'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json:160',
        'reports/PAIRED_LATENCY_E0_RESTORE_20261027.json:7'],
        details='At most a paired TRAIN timing sample on one loaded RTX3090/host: four fixed-order phases,600 measured frames per phase and initialization separate. Report before/after baseline results and clearly define any arithmetic bracket-mean FPS comparison. Do not conflate arithmetic mean FPS with pooled throughput. Other6 server25 GPU trainers and other CPU/GPU/host tasks remain active. No full-test FPS, isolated-host maximum, new accuracy, universal cost selection, multi-seed or current32 model result follows. All actual paired numbers remain pending.')
}

metadata = dict(deterministic['metadata_sha256'])
for name in ['reports/ALL882_FULL_TEST_RESULTS.json', 'reports/ALL882_TRAINING_RESULT_AUDIT.json',
             'reports/ADOPTED_CANDIDATE_LATENCY_SOURCE_REVIEW.json']:
    metadata[name] = sha((ROOT / name).read_bytes())
report = dict(
    audit_skill='experiment-audit', review_round='R1', status='SOURCE_FAIL', verdict='FAIL',
    blocking_issues=2, issues=issues, generated_at=datetime.now(timezone.utc).isoformat(), date='2026-10-27',
    agent_id='/root/audit_baseline_e0_pair_latency_20261027',
    reviewer_model_requested='gpt-6-astra', reviewer_reasoning_requested='max',
    actual_model='UNATTESTED', reviewer_model='UNATTESTED', reviewer_reasoning_actual='UNATTESTED',
    reviewer_family='openai', review_independence='same-family', acceptance_status='provisional',
    review_scope='Fresh source-only semantic and control-plane audit of paired baseline/historical-E0 profiler and actual dependencies. Local source/metadata/hash/AST/compile only. No source fixes, SSH, remote state, GPU, model execution, checkpoint payload or active-weight reads.',
    plan_sha256=PLAN_SHA, profiler_sha256=SOURCE_SHA, source_sha256=plan['source_sha256'],
    audited_input_sha256=metadata, source_count=113, python_sources_compiled=112,
    checks=checks, evaluation_type='real_gt initialization only; causal TRAIN latency sample; no accuracy evaluation',
    claim_ceiling='No paired performance result exists yet. After source fixes and successful native execution, only specified600-frame-per-phase TRAIN latency/totalP with separately reported initialization and loaded-host scope may be claimed. Any bracket-meanFPS comparison must define its arithmetic and retain before/after values.',
    provenance_observation='The current restore/plan note says original native training-exit provenance remains missing. An existing ALL882_TRAINING_RESULT_AUDIT.json:219-228 records COMPLETE/exit0 for that historical training. This review does not reopen those native artifacts or re-audit training completion; keep restoration-only scope and do not interpret this audit as adjudicating historical training provenance.',
    trace_path='.aris/traces/experiment-audit/2026-10-27_baseline_e0_pair_latency/',
    execution_performed=dict(local_source_reads=True, source_hashing=True, static_ast_and_compile=True,
                             git_source_history_read=True, ssh=False, remote_query=False, gpu=False,
                             model=False, checkpoint_payload_read=False, active_weight_read=False),
    artifacts_written_only_in_trace=True)
(TRACE / 'round1.review.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
(TRACE / 'round1.source_guard_payload.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
summary = '''# Baseline / historical E0 paired-latency source audit — R1

Verdict: **FAIL — two source-readiness blockers.** Requested reviewer: Astra / max. Actual model/effort: UNATTESTED. Same-family, provisional. No SSH, GPU, project-module import, model execution or checkpoint/active-weight read.

1. **F1 (P1): the profiler does not verify its own native wrapper.** `tools/profile_baseline_e0_pair.py:24-35` checks its plan, source and environment, and `:42-66` checks/signals the two training groups, but there is no profiler task/status, expected command or parent-PID admission. `scripts/run_benchmark_tracking.py:160-165` shows the existing native binding pattern. Bind the profile's own task, RUNNING state, GPU, parent PID and exact command before any SIGSTOP, then execute through the detached registered wrapper.
2. **F2 (P2): six actual package initializers are outside the source hash map.** Missing: `dual_rgbt/__init__.py`, `lib/__init__.py`, `lib/models/__init__.py`, `lib/models/layers/__init__.py`, `lib/test/tracker/__init__.py`, `lib/utils/__init__.py`. The model and utility initializers contain executed imports (`lib/models/__init__.py:1`, `lib/utils/__init__.py:1`). Add their current hashes only; no neural source edit is needed.

All 113 declared hashes match and all 112 Python files pass offline AST/compile checks. The 112 inherited source entries are unchanged. The original baseline uses strict SOTep14 loading and its complete causal tracker. Historical E0 uses the exact ee60704e... checkpoint, strict loading, frozen eval, and the official benchmark factories. The prior model source differs only by a default checkpoint path relocation, verified byte-for-byte. Total parameter assertions are 103,138,437 and 103,374,215.

The E0 crop binding is correct on the reviewed import path: earlier imports load global/evidence trackers, never `dual_rgbt.tracker`; `build_model('e0')` sets `data.crop_pair` before `build_tracker` first imports that tracker. Its local binding therefore resolves to `six_channel_crop` (`scripts/run_benchmark_tracking.py:21-35`, `dual_rgbt/tracker.py:6`).

Each phase resets causal state/templates, warms TRAIN frames 1–20 and measures 21–220 for the same three sequences: 600 rows per phase, 2,400 overall. Order is baseline-before / E0 cost0 / E0 cost0.05 / baseline-after. CUDA synchronization correctly brackets decode-inclusive and service timers; initialization is separate. No accuracy, optimizer or trainable-parameter evaluation is called.

Trainer ownership is restricted to the two hash-bound own GPU0 groups, with exact live wrapper/child commands, UID/PGID and compute-client equality before pause. The try/finally restores both groups after ordinary import/load/forward/assertion exceptions and records all four states (`tools/profile_baseline_e0_pair.py:42-68`, `:129-135`). Native execution and post-resume progress remain pending; this audit does not claim current liveness or resilience to uncatchable termination.

Claim ceiling: a loaded-host TRAIN timing sample. Preserve baseline-before/after values and explicitly define any bracket-mean FPS comparison; arithmetic mean FPS is distinct from pooled throughput. Other GPU and CPU jobs remain active. This is not full-test FPS or a result for the current 32 training models.

Plan SHA256: `b484194f296dfd5a9fdebe98fe5a4b0eda6ae92250cfafe0cb1f00fb9c32c4d6`

Profiler SHA256: `0d737c4a83a7f841657b5282921b3fbb2fd82bd61dade2f032c590839fde228b`

See `round1.review.json` for complete evidence and exact input/source hashes; `deterministic_source_checks.json` for compilation and import-closure results. The source guard payload is deliberately SOURCE_FAIL and is written only inside this trace.
'''
(TRACE / 'round1.review.md').write_text(summary, encoding='utf-8')
print(json.dumps(dict(status=report['status'], blocking_issues=2, plan_sha256=PLAN_SHA,
                     profiler_sha256=SOURCE_SHA, trace_path=report['trace_path']), indent=2))
