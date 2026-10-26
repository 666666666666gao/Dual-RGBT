"""Execute only the runner's AST-extracted orchestration with CPU fakes."""
import ast
import hashlib
import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace

ROOT = Path(r'C:\Users\gb\projects\Dual-RGBT')
OUT = Path(__file__).resolve().parent
source = (ROOT / 'scripts/profile_rankstop_scaling.py').read_bytes()
plan_bytes = (ROOT / 'reports/RANKSTOP_INFERENCE_SCALING_PLAN_20261026.json').read_bytes()
source_sha = hashlib.sha256(source).hexdigest()
assert source_sha == '8a17f686d892cb7e1ca47b2f1648bc042ffc14e106d73a642249b60f726496e6', source_sha
(OUT / 'R1_profile_rankstop_scaling.py').write_bytes(source)
(OUT / 'R1_plan.json').write_bytes(plan_bytes)
tree = ast.parse(source)
run_arm_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_arm')
code = compile(ast.Module(body=[run_arm_node], type_ignores=[]), '<reviewed-run-arm>', 'exec')


def case(mode):
    events = []
    processes = []
    stop = SimpleNamespace(flag=False)
    stop.set = lambda: (events.append('stop.set'), setattr(stop, 'flag', True))
    sampler = SimpleNamespace(started=False, joined=False, daemon=threading.Thread().daemon)

    def make_thread(target, args):
        def start():
            sampler.started = True
            events.append('sampler.start')
            args[1].extend([
                {'epoch': 101., 'gpus': [[0, 2200, 21927, 25], [1, 2328, 21799, 30]]},
                {'epoch': 105., 'gpus': [[0, 4200, 19927, 45], [1, 2328, 21799, 30]]},
            ])
        sampler.start = start
        sampler.join = lambda: (events.append('sampler.join'), setattr(sampler, 'joined', True))
        return sampler

    class Ready:
        def __init__(self):
            self.n = 0

        def get(self, timeout):
            events.append('ready.get')
            self.n += 1
            if mode == 'readiness_timeout' and self.n == 2:
                raise queue.Empty('injected CPU readiness timeout')
            return {'pid': 700 + self.n, 'allocated': 400000000, 'reserved': 500000000}

    class Barrier:
        def __init__(self, parties):
            self.parties = parties

        def wait(self, timeout):
            events.append('barrier.wait')
            if mode == 'barrier_failure':
                raise threading.BrokenBarrierError('injected CPU barrier failure')

    class Process:
        def __init__(self, target, args):
            self.args = args
            self.pid = 700 + len(processes)
            self.exitcode = None
            self.joined = False
            processes.append(self)

        def start(self):
            index = len([p for p in processes if p is not self])
            events.append('process.start:%d' % self.pid)
            assigned = self.args[1]
            report = {
                'total_parameters': 103218887,
                'started': 100. + index * .5,
                'finished': 110. + index,
                'frames': len(assigned) * 128,
                'results': [{'sequence': row['name'], 'predictions': [row['name']]} for row in assigned],
            }
            Path(self.args[4]).write_text(json.dumps(report))

        def join(self):
            events.append('process.join:%d' % self.pid)
            self.joined = True
            self.exitcode = 1 if mode == 'worker_failure' and self is processes[0] else 0

    context = SimpleNamespace(Barrier=Barrier, Queue=Ready, Process=Process)
    times = iter([80., 99., 115.])
    usages = iter([SimpleNamespace(ru_utime=0., ru_stime=0.), SimpleNamespace(ru_utime=10., ru_stime=2.)])
    namespace = {
        'gpu_sample': lambda: {'gpus': [[0, 26, 24101, 0], [1, 2328, 21799, 13]]},
        'mp': SimpleNamespace(get_context=lambda method: context),
        'threading': SimpleNamespace(Event=lambda: stop, Thread=make_thread),
        'monitor': None, 'worker': None,
        'resource': SimpleNamespace(RUSAGE_CHILDREN=-1, getrusage=lambda which: next(usages)),
        'time': SimpleNamespace(time=lambda: next(times)),
        'json': json, 'print': lambda *args, **kwargs: None,
    }
    exec(code, namespace)
    result = None
    exception = None
    with tempfile.TemporaryDirectory(prefix='rankstop_review_fixture_') as temporary:
        try:
            result, predictions = namespace['run_arm'](
                {'frames_per_sequence': 128}, [{'name': 'stream%d' % i} for i in range(8)],
                2, Path(temporary) / 'arm', 80)
        except (queue.Empty, threading.BrokenBarrierError, AssertionError) as error:
            exception = type(error).__name__
    facts = dict(mode=mode, exception=exception, sampler_started=sampler.started,
                 sampler_daemon=sampler.daemon, sampler_stop_set=stop.flag,
                 sampler_joined=sampler.joined, workers_joined=[p.joined for p in processes],
                 worker_exit_codes=[p.exitcode for p in processes], events=events)
    if mode == 'success':
        assert exception is None and stop.flag and sampler.joined
        assert all(p.joined for p in processes)
        assert result['total_frames'] == 1024 and result['inference_wall_seconds'] == 11.
        assert result['aggregate_fps'] == 1024. / 11.
        assert result['parent_makespan_seconds'] == 16.
        assert result['end_to_end_seconds'] == 35.
        assert len(predictions) == 8
        facts['timing_and_equal_work_checks'] = 'PASS'
    else:
        assert exception is not None
        assert sampler.started and not sampler.daemon and not stop.flag and not sampler.joined
        facts['leaked_non_daemon_sampler_reproduced'] = True
    return facts


facts = [case(mode) for mode in ('success', 'readiness_timeout', 'barrier_failure', 'worker_failure')]
report = {
    'status': 'COUNTEREXAMPLE_CONFIRMED', 'source_sha256': source_sha,
    'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
    'execution_scope': 'AST-extracted run_arm only with fake processes, GPU sampler, timing and resources; no Torch/OpenCV/GPU/SSH or real worker/benchmark execution.',
    'cases': facts,
}
(OUT / 'R1_CPU_LIFECYCLE_FIXTURES.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
