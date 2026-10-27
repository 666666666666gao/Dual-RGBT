"""Source-only R4 hash/compile checks and in-memory terminal-race regression."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[5]
TRACE = Path(__file__).resolve().parent
REMOTE = '/data2/gb/Dual-RGBT'
PYTHON = REMOTE + '/conda-envs/dual_rgbt/bin/python'


def read(name):
    return json.loads((ROOT / name).read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def programs(source, profile):
    tree = ast.parse(source)
    compile(tree, 'profile_source', 'exec')
    assignments = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'code' for t in n.targets)
                   and isinstance(n.value, ast.BinOp)]
    sampler = next(v for v in assignments if v.left.value.startswith('import csv'))
    equality = next(v for v in assignments if v.left.value.startswith('import json,torch'))
    values = {}
    for case in profile['cases']:
        values[case['name']] = eval(compile(ast.Expression(sampler), '<template-only>', 'eval'),
            {'__builtins__': {}}, {'REMOTE': REMOTE, 'case': case, 'plan': profile, 'PYTHON': PYTHON})
    values['CPU_equivalence'] = eval(compile(ast.Expression(equality), '<template-only>', 'eval'),
        {'__builtins__': {}}, {'REMOTE': REMOTE, 'plan': profile})
    return values


profile_name = 'reports/HOST_ALLOCATOR_PROFILE_R4_PLAN_20261027.json'
profile = read(profile_name)
old_profile = read('reports/HOST_ALLOCATOR_PROFILE_PLAN_20261027.json')
expected = {
    'tools/profile_host_allocator_r4.py': 'b7f837f89827843f5a3ab8200bcfd584466c820c83174db99aa611cf6d9f1cd4',
    profile_name: 'c8a439d3a113740a3d21668baf9249787c8f6d6832d3d3562813f5250bba68b7',
    'reports/host_allocator_r4_default_plan_20261027.json': 'f6ed9381a8141c08c5aa41371c9e9321da4594e5d6891a7114025936023c0d03',
    'reports/host_allocator_r4_compact_plan_20261027.json': 'c3d46362069ffab78dc725f63e1c35679a3f1dc81e29c2154ad6db41c0269f3b',
}
assert all(sha(ROOT / n) == h for n, h in expected.items())
assert all(sha(ROOT / n) == h for n, h in profile['source_sha256'].items())
original = read('reports/concurrent_joint_quality_k4_cost0_plan_20261026.json')
assert {n: h for n, h in profile['source_sha256'].items() if n != 'tools/profile_host_allocator_r4.py'} == original['source_sha256']
assert profile['owners'] == old_profile['owners']
old_guard_names = ['reports/HOST_ALLOCATOR_PROFILE_SOURCE_REVIEW_20261027.json',
                   'reports/HOST_ALLOCATOR_DEFAULT_SOURCE_REVIEW_20261027.json',
                   'reports/HOST_ALLOCATOR_COMPACT_SOURCE_REVIEW_20261027.json']
retained = {n: sha(ROOT / n) for n in old_guard_names}
assert retained[old_guard_names[0]] == '7dd80df9bec45597fa8e379346d39864f5e89c4abeb898bb8cb8534d72bbb7a6'
old_verification = read('.aris/traces/system-profile/2026-10-27_host_allocator_source_review/round3/verification.json')
assert all(sha(ROOT / n) == h for n, h in old_verification['hashes'].items())
case_checks = []
for case in profile['cases']:
    plan = read(case['plan'])
    old_case = read(next(c['plan'] for c in old_profile['cases'] if c['name'] == case['name']))
    assert sha(ROOT / case['plan']) == case['plan_sha256']
    assert plan['source_sha256'] == profile['source_sha256']
    assert plan['input_sha256'] == original['input_sha256']
    assert plan['sequence_names'] == original['sequence_names']
    changed = sorted(k for k in set(plan) | set(old_case) if plan.get(k) != old_case.get(k))
    assert changed == ['registered_grid', 'runs', 'sanity_native_status', 'scope', 'source_review', 'source_sha256', 'tasks']
    assert set(plan['runs'].values()).isdisjoint(old_case['runs'].values())
    assert set(plan['tasks'].values()).isdisjoint(old_case['tasks'].values())
    case_checks.append(dict(case=case['name'], plan_sha256=case['plan_sha256'], changed_fields=changed,
                            original_input_bindings_unchanged=True, fresh_run_task_names=True))

source = (ROOT / 'tools/profile_host_allocator_r4.py').read_text()
generated = programs(source, profile)
generated_checks = []
for name, code in generated.items():
    tree = ast.parse(code)
    compile(tree, 'generated_' + name, 'exec')
    (TRACE / ('generated_' + name + '.py')).write_text(code, encoding='utf-8')
    generated_checks.append(dict(program=name, sha256=hashlib.sha256(code.encode()).hexdigest(), compile='PASS', executed=False))
    if name == 'CPU_equivalence':
        continue
    case = next(c for c in profile['cases'] if c['name'] == name)
    plan = read(case['plan'])
    command = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'command' for t in n.targets))
    argv = eval(compile(ast.Expression(command), '<argv-only>', 'eval'), {'__builtins__': {'str': str}},
        dict(p=plan, gpu=plan['physical_gpu'], r=PurePosixPath(REMOTE), trainer='scripts/train_candidate_joint_quality_trajectory.py', case=case))
    assert argv == [PYTHON, plan['launch_wrapper'], '--task', plan['tasks']['sanity'], '--gpu', '2', '--',
                    PYTHON, REMOTE + '/scripts/train_candidate_joint_quality_trajectory.py', '--plan',
                    case['plan'], '--run', plan['runs']['sanity'], '--sanity']
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    assert len(handlers) == 1
    assert [e.id for e in handlers[0].type.elts] == ['ProcessLookupError', 'FileNotFoundError']
    generated_checks[-1]['argv'] = argv
    generated_checks[-1]['exception_types'] = ['ProcessLookupError', 'FileNotFoundError']


class Fixture:
    """Every path, clock, PID and signal below is synthetic in-memory state."""
    def __init__(self, missing_type, terminal_exit=0):
        self.missing_type = missing_type
        self.terminal_exit = terminal_exit
        self.iteration = 0
        self.returncode = None
        self.wait_calls = 0
        self.signals = []
        self.writes = {}
        self.stdout = []
        self.pid = 7654321
        self.plan = read(profile['cases'][0]['plan'])

    def poll(self):
        return self.returncode

    def wait(self):
        self.wait_calls += 1
        assert self.returncode is not None
        return self.returncode

    def killpg(self, pid, sig):
        assert pid == self.pid and sig == 9
        self.signals.append([pid, sig])
        self.returncode = -9

    def path(self, name):
        return FakePath(self, str(name))


class FakePath:
    def __init__(self, fixture, name):
        self.fixture = fixture
        self.name = name

    def __truediv__(self, name):
        return FakePath(self.fixture, self.name.rstrip('/') + '/' + str(name))

    def exists(self):
        return True

    def read_bytes(self):
        f = self.fixture
        if self.name.endswith('/status.json'):
            f.iteration += 1
            terminal = f.iteration >= 3
            if terminal:
                f.returncode = f.terminal_exit
            return json.dumps(dict(pid=f.pid, state=('COMPLETE' if f.terminal_exit == 0 else 'FAILED') if terminal else 'RUNNING', exit_code=f.terminal_exit if terminal else None)).encode()
        if self.name.endswith('/environ'):
            return ('\x00'.join(k + '=' + v for k, v in f.plan['execution_env'].items()) + '\x00').encode()
        if self.name.endswith('/COMPLETE.json'):
            return json.dumps(dict(sanity=True, updates=4, tracked_frames=20)).encode()
        if self.name.endswith('/SANITY.json'):
            return json.dumps(dict(status='PASS', strict_reload_prediction_bit_equal=True)).encode()
        raise AssertionError('Unregistered fake path: ' + self.name)

    def read_text(self):
        f = self.fixture
        if self.name.endswith('/children'):
            return '7654322'
        if self.name.endswith('/smaps_rollup'):
            if f.iteration == 2:
                raise f.missing_type('Synthetic disappearing child metadata')
            return 'header\nRss: 100 kB\nPss: 90 kB\nPrivate_Dirty: 80 kB\nSwap: 0 kB\n'
        raise AssertionError('Unregistered fake path: ' + self.name)

    def write_text(self, value):
        self.fixture.writes[self.name] = value


class FakeTime:
    @staticmethod
    def monotonic():
        return 0

    @staticmethod
    def time():
        return 1

    @staticmethod
    def sleep(seconds):
        assert seconds == 2


class FakeSignal:
    SIGKILL = 9


def exercise(label, code, missing_type, terminal_exit=0, expect_pass=False):
    """Execute only the exact post-launch sampler AST with all effects mocked."""
    module = ast.parse(code)
    outer = next(n for n in module.body if isinstance(n, ast.Try))
    f = Fixture(missing_type, terminal_exit)
    state = dict(success=False, deadline=540, child=f, r=f.path(REMOTE), p=f.plan,
                 case=profile['cases'][0], time=FakeTime, Path=f.path, json=json, os=f,
                 signal=FakeSignal, print=lambda value: f.stdout.append(value))
    error = None
    try:
        exec(compile(ast.Module(body=[outer], type_ignores=[]), '<exact-post-launch-sampler>', 'exec'), state)
    except (ProcessLookupError, FileNotFoundError, AssertionError) as exc:
        error = type(exc).__name__
    if expect_pass:
        assert error is None and state['success'] and f.returncode == 0 and f.wait_calls == 1
        assert f.signals == [] and f.writes == {} and len(f.stdout) == 1
        assert len(json.loads(f.stdout[0])['samples']) == 1
    else:
        assert error is not None and not state['success'] and not f.stdout
        assert len(f.writes) == 1
        receipt = json.loads(next(iter(f.writes.values())))
        assert receipt['wrapper_exit_code'] == f.returncode
        if 'samples' in receipt:
            assert len(receipt['samples']) == 1
    return dict(case=label, result='PASS', expected_profile_success=expect_pass,
                raised=error, synthetic_wrapper_exit=f.returncode, synthetic_signals=f.signals,
                native_terminal_required=True, recorded_samples=len(state['samples']),
                failure_samples_retained='samples' in json.loads(next(iter(f.writes.values()))) if f.writes else None,
                fixture_only=True, real_processes_started=0)


old_generated = programs((ROOT / 'tools/profile_host_allocator.py').read_text(), old_profile)
regressions = [
    exercise('R3 reproduced ProcessLookupError rejection', old_generated['default'], ProcessLookupError),
    exercise('R4 ProcessLookupError then genuine native0', generated['default'], ProcessLookupError, expect_pass=True),
    exercise('R4 FileNotFoundError then genuine native0', generated['default'], FileNotFoundError, expect_pass=True),
    exercise('R4 vanished metadata then native failure stays rejected', generated['default'], ProcessLookupError, terminal_exit=1),
]
diagnosis = read('reports/HOST_ALLOCATOR_R3_RUNTIME_FAILURE_DIAGNOSIS_20261027.json')
transcript = read('reports/HOST_ALLOCATOR_DEFAULT_SSH_20261027.json')
assert transcript['ssh_exit_code'] == 1 and not transcript['stdout'] and 'ProcessLookupError' in transcript['stderr']
native = diagnosis['files']['runs/pipeline_tasks/train_host_allocator_default_20261027_sanity/status.json']
failure = diagnosis['files']['reports/train_host_allocator_default_20261027_sanity.profile_failure.json']
assert native['state'] == 'RUNNING' and failure['wrapper_exit_code'] == -9 and failure['wrapper_pid'] == native['pid']
checks = dict(status='PASS', expected_hashes=expected, source_hash_count=51,
              original_50_sources_and_24_input_bindings_unchanged=True, original_owner_records_unchanged=True,
              R3_source_plans_guards_unchanged=True, R3_guard_hashes=retained, cases=case_checks,
              generated=generated_checks, mocked_terminal_race_regressions=regressions,
              R3_runtime_disposition='FAILED_REJECTED_NATIVE_ZERO_NOT_OBTAINED',
              real_NN_execution=False, remote_queries=False, real_processes_started=0)
for name in list(expected) + ['reports/HOST_ALLOCATOR_DEFAULT_SSH_20261027.json', 'reports/HOST_ALLOCATOR_R3_RUNTIME_FAILURE_DIAGNOSIS_20261027.json']:
    p = TRACE / 'inputs' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((ROOT / name).read_bytes())
(TRACE / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
print(json.dumps(checks, indent=2))
