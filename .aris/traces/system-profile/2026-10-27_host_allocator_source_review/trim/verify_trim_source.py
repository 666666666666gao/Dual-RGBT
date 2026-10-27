"""Targeted source, archived-reference, generated-code and mocked callback audit."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
TRACE = Path(__file__).resolve().parent
REMOTE = '/data2/gb/Dual-RGBT'
PYTHON = REMOTE + '/conda-envs/dual_rgbt/bin/python'
PROFILE = 'reports/HOST_ALLOCATOR_TRIM_PROFILE_PLAN_20261027.json'
CASE = 'reports/host_allocator_trim_case_plan_20261027.json'


def read(name):
    return json.loads((ROOT / name).read_bytes())


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


expected = {
    'scripts/train_candidate_host_compact.py': 'd62266395ca43967f959787850892db0c0ab08c3666b11fc67eeb50328c162d3',
    'tools/profile_host_allocator_trim.py': '83608f5d00e54c7be68d11ec4e95700fdedf7e3b23d260772cc612c02f72801f',
    PROFILE: '32a5035c80ddd54d07f68fc681fa49562bfe5e974acf9941dc9517ef6bbdfda5',
    CASE: 'afbd6498e8b65a46a9b8a69fcc40f49f2ee77432d95e9a5682b88935cbdb9b3f',
}
assert all(sha(n) == h for n, h in expected.items())
profile, case = read(PROFILE), read(CASE)
assert len(profile['source_sha256']) == 53 and len(case['source_sha256']) == 52
assert all(sha(n) == h for n, h in profile['source_sha256'].items())
assert all(profile['source_sha256'][n] == h for n, h in case['source_sha256'].items())
original = read('reports/concurrent_joint_quality_k4_cost0_plan_20261026.json')
assert all(case['source_sha256'][n] == h for n, h in original['source_sha256'].items())
assert case['input_sha256'] == original['input_sha256'] and case['sequence_names'] == original['sequence_names']
r4 = read('reports/HOST_ALLOCATOR_PROFILE_R4_PLAN_20261027.json')
assert profile['owners'] == r4['owners']
old_compact = read(r4['cases'][1]['plan'])
changed = sorted(k for k in set(case) | set(old_compact) if case.get(k) != old_compact.get(k))
assert changed == ['launch_trainer', 'registered_grid', 'runs', 'sanity_native_status', 'scope', 'source_review', 'source_sha256', 'tasks']
assert case['execution_env'] == old_compact['execution_env']
assert sha('reports/HOST_ALLOCATOR_PROFILE_R4_SOURCE_REVIEW_20261027.json') == '1b77749e77fa1a19fb1e2bc82eee570e32fd7281f0d8277f411e2f04180a15e5'

reference = profile['cases'][0]
assert reference['kind'] == 'completed_reference' and profile['cases'][1]['kind'] == 'new_admission'
sample = read(reference['reference_sample'])
assert sha(reference['reference_sample']) == reference['reference_sample_sha256']
reference_plan = read(reference['plan'])
reference_review = read(reference_plan['source_review'])
assert sha(reference['plan']) == reference['plan_sha256'] == sample['case']['plan_sha256'] == reference_review['plan_sha256']
assert sample['case']['plan'] == reference['plan']
assert reference_plan['source_sha256'] == reference_review['source_sha256']
assert sample['native']['state'] == 'COMPLETE' and sample['native']['exit_code'] == 0
assert sample['native']['task'] == reference_plan['tasks']['sanity']
assert sample['native']['command'] == [PYTHON, REMOTE + '/scripts/train_candidate_joint_quality_trajectory.py',
    '--plan', reference['plan'], '--run', reference_plan['runs']['sanity'], '--sanity']
assert sample['complete']['sanity'] and sample['complete']['updates'] == 4 and sample['complete']['tracked_frames'] == 20
assert sample['probe']['status'] == 'PASS' and sample['probe']['strict_reload_prediction_bit_equal']

diagnosis_envelope = read('reports/HOST_ALLOCATOR_R4_COMPARISON_DIAGNOSIS_20261027.json')
assert diagnosis_envelope['ssh_exit_code'] == 0
diagnosis = json.loads(diagnosis_envelope['stdout'])
assert diagnosis['state_tensors'] == 265 and diagnosis['network_differences'] == []
assert len(diagnosis['optimizer_differences']) == 5 and all(diagnosis['csv_equal'].values())
assert all(n['state'] == 'COMPLETE' and n['exit_code'] == 0 for n in diagnosis['native'])
heap = read('reports/VRAM18_HOST_PRIVATE_MAPPING_DIAGNOSIS_20261027.json')
assert '[heap]' in heap[0]['mapping'] and heap[0]['Private_Dirty'] == 4396536

source = (ROOT / 'tools/profile_host_allocator_trim.py').read_text()
module = ast.parse(source)
assignments = [n.value for n in ast.walk(module) if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == 'code' for t in n.targets) and isinstance(n.value, ast.BinOp)]
sampler = next(v for v in assignments if v.left.value.startswith('import csv'))
comparison = next(v for v in assignments if v.left.value.startswith('import json,torch'))
generated = []
new_argv = None
for c in profile['cases']:
    code = eval(compile(ast.Expression(sampler), '<source-template-only>', 'eval'), {'__builtins__': {}},
        dict(REMOTE=REMOTE, case=c, plan=profile, PYTHON=PYTHON))
    tree = ast.parse(code)
    compile(tree, 'generated_sampler_' + c['name'], 'exec')
    (TRACE / ('generated_sampler_' + c['name'] + '.py')).write_text(code, encoding='utf-8')
    generated.append(dict(program='sampler_' + c['name'], compile='PASS', sha256=hashlib.sha256(code.encode()).hexdigest(),
                          dispatched=c['kind'] == 'new_admission', executed=False))
    if c['kind'] == 'completed_reference':
        generated[-1]['note'] = 'Dormant template instantiation only; local completed_reference branch continues before any launch.'
        continue
    command = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'command' for t in n.targets))
    new_argv = eval(compile(ast.Expression(command), '<argv-only>', 'eval'), {'__builtins__': {'str': str}},
        dict(p=case, gpu=case['physical_gpu'], r=PurePosixPath(REMOTE), trainer=case['launch_trainer'], case=c))
    assert new_argv == [PYTHON, case['launch_wrapper'], '--task', case['tasks']['sanity'], '--gpu', '2', '--',
        PYTHON, REMOTE + '/' + case['launch_trainer'], '--plan', CASE, '--run', case['runs']['sanity'], '--sanity']
    generated[-1]['argv'] = new_argv
    assert 'deadline=time.monotonic()+540' in code and 'os.killpg(child.pid,signal.SIGKILL)' in code
    assert "n['exit_code']==0 and child.wait()==0" in code and 'except (ProcessLookupError,FileNotFoundError):' in code
    assert 'samples=samples' in code
comparison_code = eval(compile(ast.Expression(comparison), '<source-template-only>', 'eval'), {'__builtins__': {}}, dict(REMOTE=REMOTE, plan=profile))
compile(comparison_code, 'generated_comparison', 'exec')
(TRACE / 'generated_comparison.py').write_text(comparison_code, encoding='utf-8')
generated.append(dict(program='CPU_comparison', compile='PASS', sha256=hashlib.sha256(comparison_code.encode()).hexdigest(), dispatched=True, executed=False))
assert "torch.allclose(a,b,atol=1e-8,rtol=1e-6)" in comparison_code
assert 'torch.isfinite(a).all().item() and torch.isfinite(b).all().item()' in comparison_code
assert 'optimizer_bit_equal=' in comparison_code and 'optimizer_max_abs=' in comparison_code
assert "assert all(torch.equal(a['net'][n],b['net'][n]) for n in a['net'])" in comparison_code

# All callback effects are in-memory mocks. Neither ctypes.CDLL nor gc.collect is called for real.
events, writes, forwarded = [], [], []
native_command = new_argv[new_argv.index('--') + 1:]
fake_sys = SimpleNamespace(argv=native_command[1:].copy(), executable=native_command[0])
argv_before = fake_sys.argv.copy()
sentinel = object()


class MemoryPath:
    def read_text(self):
        events.append('memory')
        return 'header\nRss: 100 kB\nPss: 90 kB\nPrivate_Dirty: 80 kB\n'


class RunPath:
    def __truediv__(self, name):
        assert name == 'host_memory_startup.json'
        return self

    def write_text(self, value):
        events.append('write_startup_memory')
        writes.append(json.loads(value))


args = [object(), object(), case, object(), RunPath(), object(), object(), True]


def original_stage(*received):
    assert len(received) == 8 and all(a is b for a, b in zip(received, args))
    events.append('original_trajectory_stage')
    forwarded.append(received)
    return sentinel


fake_module = SimpleNamespace(trajectory_stage=original_stage)


def fake_original_main():
    events.append('original_main')
    assert native_command == [fake_sys.executable] + fake_sys.argv
    assert fake_module.trajectory_stage(*args) is sentinel


fake_module.main = fake_original_main


def import_module(name):
    assert name == 'train_candidate_joint_quality_trajectory'
    events.append('import_original_quality_trainer')
    return fake_module


class MallocTrim:
    def __call__(self, pad):
        assert pad == 0
        assert self.argtypes == ['c_size_t'] and self.restype == 'c_int'
        events.append('malloc_trim_0')
        return 1


malloc_trim = MallocTrim()


def load_libc(name):
    assert name == 'libc.so.6'
    events.append('load_libc')
    return SimpleNamespace(malloc_trim=malloc_trim)


def fake_path(name):
    assert name == '/proc/self/smaps_rollup'
    return MemoryPath()


wrapper = ast.parse((ROOT / 'scripts/train_candidate_host_compact.py').read_text())
main_def = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
state = dict(ROOT=ROOT, sys=fake_sys, json=json, importlib=SimpleNamespace(import_module=import_module),
    gc=SimpleNamespace(collect=lambda: events.append('gc_collect')), Path=fake_path,
    ctypes=SimpleNamespace(CDLL=load_libc, c_size_t='c_size_t', c_int='c_int'))
exec(compile(ast.Module(body=[main_def], type_ignores=[]), '<exact-wrapper-main-mocked>', 'exec'), state)
state['main']()
assert events == ['import_original_quality_trainer', 'original_main', 'memory', 'gc_collect', 'load_libc',
                  'malloc_trim_0', 'memory', 'write_startup_memory', 'original_trajectory_stage']
assert fake_sys.argv == argv_before and len(forwarded) == len(writes) == 1
assert writes[0]['malloc_trim_return'] == 1
trainer_tree = ast.parse((ROOT / 'scripts/train_candidate_joint_quality_trajectory.py').read_text())
trainer_main = next(n for n in trainer_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
calls = [n for n in ast.walk(trainer_main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'trajectory_stage']
assert len(calls) == 1 and calls[0].lineno == 332
checks = dict(status='PASS', hashes=expected, aggregate_source_count=53, new_case_source_count=52,
    original_50_sources_and_24_inputs_unchanged=True, original_owner_records_unchanged=True,
    delta_from_R4_compact=changed, generated_programs=generated,
    completed_reference=dict(sample_sha256=sha(reference['reference_sample']), plan_sha256=reference['plan_sha256'],
        native_exit_code=0, native_argv_bound=True, actual_recorded_updates=4, actual_recorded_frames=20,
        checkpoint_sha256_recorded_only=sample['complete']['checkpoint_sha256'], reference_is_not_rerun=True),
    callback_fixture=dict(status='PASS', events=events, original_main_calls=1, original_stage_calls=1,
        exact_argument_object_identity=True, argv_unchanged=True, native_command_equality=True,
        synthetic_memory_receipts=1, real_libc_calls=0, real_gc_calls=0, real_neural_calls=0),
    prior_R4_exact_policy='FAILED_PRESERVED_NOT_RETROACTIVELY_RELAXED',
    optimizer_tolerance=dict(atol=1e-8, rtol=1e-6, finite_required=True, exact_flag_and_max_error_reported=True,
        predeclared_for_new_trim_probe_only=True, comparison_executed=False),
    intervention='compact allocator environment plus one post-load gc/trim, relative to completed default reference; not isolated trim attribution',
    remote_queries=False, active_job_queries=False, weight_reads=False, torch_import=False, neural_execution=False)
for name in list(expected) + [reference['reference_sample'], 'reports/HOST_ALLOCATOR_R4_COMPACT_SAMPLES_20261027.json',
    'reports/HOST_ALLOCATOR_R4_COMPARISON_DIAGNOSIS_20261027.json', 'reports/VRAM18_HOST_PRIVATE_MAPPING_DIAGNOSIS_20261027.json']:
    p = TRACE / 'inputs' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((ROOT / name).read_bytes())
(TRACE / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
print(json.dumps(checks, indent=2))
