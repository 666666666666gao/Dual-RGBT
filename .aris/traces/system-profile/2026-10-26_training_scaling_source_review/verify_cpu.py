"""CPU-only spawn/IPC review of the exact profiler. No Torch/NN or remote queries."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
SOURCE = 'scripts/profile_candidate_training_scaling.py'
PLAN = 'reports/TRAIN_CONCURRENCY_PLAN_20261026.json'
SOURCE_SHA = 'c76b74b3efd4828f909a63842ec94c33e9da6eb52c65a9ce674fdc80baaf34fa'
PLAN_SHA = '0f9a31ac23c4a3c1450ef683e3c386e2f138756985ae4a71864a116543c03ce5'
# CPython direct execution of /project/scripts/profile.py puts scripts on sys.path.
# The real profiler prepends ROOT and leaves that direct-script entry intact.
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('training_scaling_under_review', ROOT / SOURCE)
profile = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = profile
spec.loader.exec_module(profile)

class BoundedQueue:
    def __init__(self, raw):
        self.raw = raw
        self.requested_timeouts = []
    def put(self, value):
        return self.raw.put(value)
    def get(self, timeout):
        self.requested_timeouts.append(timeout)
        return self.raw.get(timeout=min(timeout, 5.0))
    def close(self):
        self.raw.close()
    def join_thread(self):
        self.raw.join_thread()

class BoundedProcess:
    def __init__(self, raw):
        self.raw, self.terminated, self.joins = raw, False, []
    @property
    def pid(self): return self.raw.pid
    @property
    def exitcode(self): return self.raw.exitcode
    def start(self): self.raw.start()
    def join(self, timeout=None):
        self.joins.append(timeout)
        self.raw.join(timeout=.25 if timeout is not None else 5.)
    def is_alive(self): return self.raw.is_alive()
    def terminate(self):
        self.terminated = True
        self.raw.terminate()

class BoundedContext:
    def __init__(self):
        self.raw = mp.get_context('spawn')
        self.processes, self.messages, self.events = [], [], []
    def Event(self):
        event = self.raw.Event(); self.events.append(event); return event
    def Queue(self):
        messages = BoundedQueue(self.raw.Queue()); self.messages.append(messages); return messages
    def Process(self, **kwargs):
        process = BoundedProcess(self.raw.Process(**kwargs)); self.processes.append(process); return process

def cpu_worker(plan, arm, index, release, finish, messages):
    import os
    helpers = ('train_candidate_evidence_policy', 'train_candidate_joint_gainpair_trajectory', 'train_global_evidence_policy')
    resolved = {name: str(Path(importlib.util.find_spec(name).origin).resolve()) for name in helpers}
    assert all(Path(path).parent == ROOT / 'scripts' for path in resolved.values())
    assert 'torch' not in sys.modules
    mode = plan['fixture_mode']
    messages.put(dict(state='BAD_READY' if mode == 'bad_ready' and index == 0 else 'READY', index=index, pid=os.getpid()))
    assert release.wait(15)
    if mode == 'bad_ready':
        assert finish.wait(15)
        return
    if mode == 'work_exit7' and index == 0:
        os._exit(7)
    if mode == 'work_hang' and index == 0:
        while True:
            time.sleep(.1)
    streams = [dict(index=i, sequence=plan['sequence_names'][i], updates=32, tracked_frames=160)
               for i in range(index, 8, plan['current_processes'])]
    time.sleep(.01)
    ended = time.perf_counter()
    time.sleep(.01)  # stand-in for excluded CPU validation
    messages.put(dict(state='RESULT', index=index, pid=os.getpid(), streams=streams,
        updates=32 * len(streams), tracked_frames=160 * len(streams), initializes=len(streams),
        training_finished_perf_counter=ended, helpers_resolved=resolved))
    assert finish.wait(15)

def main():
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(ROOT / SOURCE) == SOURCE_SHA and sha(ROOT / PLAN) == PLAN_SHA
    plan = json.loads((ROOT / PLAN).read_text())
    base = json.loads((ROOT / plan['training_plan']).read_text())
    assert len(plan['source_sha256']) == 51 and len(plan['input_sha256']) == 28
    assert {k: v for k, v in plan['source_sha256'].items() if k != SOURCE} == base['source_sha256']
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
        ast.parse((ROOT / name).read_bytes(), filename=name)
    assert all(plan['input_sha256'][k] == v for k, v in base['input_sha256'].items())
    previous = json.loads((ROOT / '.aris/traces/experiment-audit/2026-10-26_joint_gainpair_source_final/binding_verification.json').read_text())
    local_inputs, reused_remote = [], []
    for name, digest in plan['input_sha256'].items():
        if (ROOT / name).is_file():
            assert sha(ROOT / name) == digest, name
            local_inputs.append(name)
        else:
            assert previous['remote_only_input_bindings_from_sealed_prior_verification'][name] == digest
            reused_remote.append(name)
    assert plan['input_sha256']['reports/JOINT_GAINPAIR_TRAIN_SOURCE_REVIEW_20261026.json'] == '29f167345f4f66c924a5969bf0dd532f8d4768678cf427338803e8bd8b8cabf3'
    assert plan['physical_gpu'] == 1 and plan['updates_per_stream'] == 32
    assert len(plan['sequence_names']) == len(set(plan['sequence_names'])) == 8
    assert set(plan['sequence_names']).issubset(base['sequence_names'])
    assert [a['processes'] for a in plan['arms']] == [1, 2, 4, 8, 1]
    assert len({a['name'] for a in plan['arms']}) == 5
    execution = json.loads((ROOT / 'reports/JOINT_GAINPAIR_TRAIN_EXECUTION_20261026.json').read_text())
    startup = json.loads((ROOT / 'reports/JOINT_GAINPAIR_TRAIN_STARTUP_20261026.json').read_text())
    assert execution['plan_sha256'] == plan['input_sha256'][plan['training_plan']]
    admission = execution['sanity_admission']
    assert admission['native']['exit_code'] == 0 and admission['complete']['updates'] == 4
    assert admission['probe']['strict_reload_prediction_bit_equal'] and admission['complete']['selected_execution_bit_equal']
    assert admission['complete']['teacher_preserves_tracker_state'] and admission['complete']['frozen_state_tensors'] == 251
    assert len(admission['complete']['changed_state_tensors']) == 14
    assert startup['native']['pid'] == plan['coexisting_wrapper_pid'] == execution['formal']['pid']
    assert startup['child_pid'] == plan['coexisting_child_pid']
    assert startup['source_input_model_admission_passed'] and startup['finite_logged_losses'] and startup['logged_updates'] == 150
    help_result = subprocess.run([sys.executable, str(ROOT / SOURCE), '--help'], text=True, capture_output=True, timeout=10)
    assert help_result.returncode == 0 and '--plan' in help_result.stdout
    (TRACE / 'direct_script_help.txt').write_text(help_result.stdout, encoding='utf-8')
    source_ast = ast.parse((ROOT / SOURCE).read_text())
    worker_ast = next(n for n in source_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'save' for n in ast.walk(source_ast))
    calls = [n for n in ast.walk(worker_ast) if isinstance(n, ast.Call)]
    for name in ('sampled_targets', 'selected_target', 'controller_loss', 'step_optimizer'):
        assert sum(isinstance(n.func, ast.Name) and n.func.id == name for n in calls) == 1, name
    assert 'if k != \'seconds\'' in (ROOT / SOURCE).read_text()
    fixture_root = TRACE / 'cpu_fixture_outputs'
    fixture_root.mkdir(exist_ok=False)
    profile.ROOT = fixture_root
    profile.worker = cpu_worker
    profile.gpu_snapshot = lambda: 'CPU_FIXTURE_NO_GPU_QUERY'
    observations = []
    def fake_observation(plan):
        observations.append(time.perf_counter())
        return dict(observed=observations[-1], runs=[dict(gpu=0, logged_updates=0), dict(gpu=1, logged_updates=0)])
    profile.training_observation = fake_observation
    contexts, cases = [], []
    def get_context(method):
        assert method == 'spawn'
        context = BoundedContext(); contexts.append(context); return context
    profile.mp = type('MPProxy', (), {'get_context': staticmethod(get_context)})
    fixture_plan = dict(output='profile', sequence_names=plan['sequence_names'], updates_per_stream=32,
                        fixture_mode='normal', coexisting_runs=[])
    transcript = io.StringIO()
    try:
        with contextlib.redirect_stdout(transcript):
            for i, count in enumerate((1, 2, 4, 8, 1)):
                before = len(observations)
                result = profile.run_arm(fixture_plan, 'normal_%d_%d' % (i, count), count)
                assert result['total_updates'] == 256 and sum(w['initializes'] for w in result['workers']) == 8
                assert sum(w['tracked_frames'] for w in result['workers']) == 1280
                assert result['training_and_validation_wall_seconds'] >= result['training_wall_seconds'] > 0
                assert len(observations) - before == 2
                assert all(p.exitcode == 0 and not p.is_alive() for p in contexts[-1].processes)
                assert all(e.is_set() for e in contexts[-1].events)
                cases.append(dict(mode='normal', workers=count, updates=256, streams=8,
                    training_wall_seconds=result['training_wall_seconds'], validation_wall_seconds=result['training_and_validation_wall_seconds'],
                    no_live_children=True, helper_resolution=True))
            for mode in ('bad_ready', 'work_exit7', 'work_hang'):
                failed = False
                try:
                    profile.run_arm(dict(fixture_plan, fixture_mode=mode), mode, 2)
                except (AssertionError, queue.Empty):
                    failed = True
                assert failed
                context = contexts[-1]
                assert all(not p.is_alive() for p in context.processes)
                assert all(e.is_set() for e in context.events)
                assert not (fixture_root / 'profile' / mode / 'summary.json').exists()
                if mode == 'work_exit7': assert any(p.exitcode == 7 for p in context.processes)
                if mode == 'work_hang': assert any(p.terminated for p in context.processes)
                cases.append(dict(mode=mode, workers=2, expected_failure=True,
                    exit_codes=[p.exitcode for p in context.processes], terminated=[p.terminated for p in context.processes],
                    no_live_children=True, both_barriers_released=True))
    finally:
        # Fault-injection harness owns these CPU children only; guard the harness itself.
        for context in contexts:
            for process in context.processes:
                if process.is_alive():
                    process.terminate(); process.join()
    (TRACE / 'cpu_run_arm_transcript.jsonl').write_text(transcript.getvalue(), encoding='utf-8')
    assert 'torch' not in sys.modules
    report = dict(status='CPU_SOURCE_FIXTURES_PASS', plan_sha256=PLAN_SHA, new_source_sha256=SOURCE_SHA,
        source_count=51, inherited_source_count=50, input_count=28,
        current_local_input_count=len(local_inputs), reused_remote_input_count=len(reused_remote),
        direct_script_help_pass=True, direct_script_and_spawn_helper_resolution=True,
        cases=cases, fixture_queue_timeouts_capped_seconds=5, fixture_join_timeout_cap_seconds=.25,
        production_queue_timeouts_observed=[180, 900], production_join_timeout_requested=30,
        torch_import=False, neural_execution=False, remote_query=False, actual_nn_runtime_admission=False)
    (TRACE / 'cpu_verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
