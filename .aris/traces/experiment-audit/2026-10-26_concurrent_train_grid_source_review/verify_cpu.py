"""Local stdlib-only review of sealed grid bindings and CPU wrapper routing."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
INDEX = 'reports/CONCURRENT_TRAIN_PLAN_INDEX_20261026.json'
GRID = 'reports/TRAIN_CONCURRENCY_CANDIDATE_GRID_20261026.json'
WRAPPER = 'scripts/run_task_four_gpu.py'
WRAPPER_SHA = 'cc2b87b38d3b5c48a9f875e78218e48abb3640891beb0b5570c68b81a25afde3'
index, grid = read(ROOT / INDEX), read(ROOT / GRID)
assert sha(ROOT / INDEX) == 'de4fe9afa72168e27b28a5bdf24f67a1989beabb62e51b788f67415904dc9993'
assert sha(ROOT / GRID) == 'a22c609fdfd2ea00a8fff9d155ce227a262cd9ffb75e56aece30f68534e3694c'
assert sha(ROOT / WRAPPER) == index['wrapper_sha256'] == WRAPPER_SHA
assert index['status'] == 'PLANNED_NOT_LAUNCHED'
assert grid['status'] == 'BOUNDED_GRID_PENDING_MEASURED_CAPACITY'
assert len(index['plans']) == 6 and len(grid['rows']) == 8
original = ast.parse((ROOT / 'scripts/run_task.py').read_text())
extended = ast.parse((ROOT / WRAPPER).read_text())
assert isinstance(original.body[0].value, ast.Constant)
extended.body[0] = copy.deepcopy(original.body[0])
gpu_call = lambda tree: next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--gpu')
old_choices = next(k for k in gpu_call(original).keywords if k.arg == 'choices')
new_choices = next(k for k in gpu_call(extended).keywords if k.arg == 'choices')
assert ast.literal_eval(old_choices.value) == ('0', '1')
assert ast.literal_eval(new_choices.value) == ('0', '1', '2', '3')
new_choices.value = copy.deepcopy(old_choices.value)
assert ast.dump(extended, include_attributes=False) == ast.dump(original, include_attributes=False)
receipt_path = ROOT / '.aris/traces/experiment-audit/2026-10-26_joint_quality_source_review/remote_cpu_stdout.json'
assert sha(receipt_path) == 'cf08f70f15f085d225490ae77e09cb0f392a45a765f7d597adeedf470c577326'
receipt = read(receipt_path)
prior_binding = read(ROOT / '.aris/traces/experiment-audit/2026-10-26_joint_gainpair_source_final/binding_verification.json')
assert receipt['formal_counts'] == dict(tracked_frames=917106, updates=182182, initializes=1764)
assert receipt['sanity_counts'] == dict(tracked_frames=20, updates=4, initializes=1)
assert receipt['train_sequences'] == 882 and receipt['dev_sequences'] == 97 and receipt['train_dev_disjoint']
prior_hashes = {
    'quality': ('014101511170dc2742c1cca4662035bae7a2b106f0f6e38cc10310eda090d42f', '9651f561a030a3ce800bdc46a230d95cd564b8bf39a8ee733c04663aad7107ac'),
    'gainpair': ('931b2ce6c4f50c49b923cfd7192419fdac64594d324df5e4f55b2f455234be77', '29f167345f4f66c924a5969bf0dd532f8d4768678cf427338803e8bd8b8cabf3'),
}
bases = {}
for family, (plan_digest, review_digest) in prior_hashes.items():
    path = f'reports/global_candidate_joint_{family}_trajectory_plan_20261026.json'
    base = read(ROOT / path)
    review_path = f'reports/JOINT_{family.upper()}_TRAIN_SOURCE_REVIEW_20261026.json'
    review = read(ROOT / review_path)
    assert sha(ROOT / path) == plan_digest and sha(ROOT / review_path) == review_digest
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == plan_digest and review['source_sha256'] == base['source_sha256']
    bases[family] = base
allowed_deltas = {'cost_weight', 'max_candidates', 'physical_gpu', 'runs', 'tasks', 'sanity_native_status', 'source_review', 'scope', 'comparison_control', 'registered_grid', 'launch_wrapper', 'source_sha256', 'input_sha256'}
all_source_hashes, all_local_inputs, remote_inputs = {}, {}, {}
all_runs = {v for base in bases.values() for v in base['runs'].values()}
all_tasks = {v for base in bases.values() for v in base['tasks'].values()}
all_reviews = {base['source_review'] for base in bases.values()}
cases = []
for entry in index['plans']:
    path = entry['path']; plan = read(ROOT / path)
    assert sha(ROOT / path) == entry['sha256']
    family = 'gainpair' if 'gainpair' in path else 'quality'; base = bases[family]
    changed = {k for k in set(base) | set(plan) if base.get(k) != plan.get(k)}
    assert changed <= allowed_deltas, (path, changed - allowed_deltas)
    assert plan['status'] == 'READY'
    assert plan['physical_gpu'] == entry['gpu']
    matches = [r for r in grid['rows'] if r['objective'] == 'joint_' + family and r['max_candidates'] == plan['max_candidates'] and r['trajectory_cost_weight'] == plan['cost_weight']]
    assert len(matches) == 1 and matches[0]['status'] == 'PLANNED_NOT_LAUNCHED'
    assert matches[0]['preferred_gpu'] == plan['physical_gpu']
    expected_source = dict(base['source_sha256'])
    if entry['gpu'] >= 2:
        expected_source[WRAPPER] = WRAPPER_SHA
    assert plan['source_sha256'] == expected_source and len(expected_source) == entry['source_count']
    assert plan['launch_wrapper'] == (WRAPPER if entry['gpu'] >= 2 else 'scripts/run_task.py')
    assert plan['registered_grid'] == GRID
    assert plan['comparison_control'] == f'reports/global_candidate_joint_{family}_trajectory_plan_20261026.json'
    assert plan['input_sha256'] == dict(base['input_sha256'], **{GRID: sha(ROOT / GRID)})
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
        ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)
        all_source_hashes[name] = digest
    for name, digest in plan['input_sha256'].items():
        p = ROOT / name
        if p.is_file():
            assert sha(p) == digest, name
            all_local_inputs[name] = digest
        else:
            assert receipt['archived_input_sha256'][name] == prior_binding['remote_only_input_bindings_from_sealed_prior_verification'][name] == digest
            remote_inputs[name] = digest
    assert plan['seed'] == 2027 and plan['rounds'] == 2 and plan['label_stride'] == 5
    assert len(plan['sequence_names']) == len(set(plan['sequence_names'])) == 882
    assert plan['expected_formal_counts'] == receipt['formal_counts']
    assert plan['sanity_updates'] == 4 and plan['sanity_expected_counts'] == receipt['sanity_counts']
    assert plan['expected_frozen_state_tensors'] == 251 and plan['expected_changed_state_tensors'] == 14
    assert plan['preview_learning_rate'] == 1e-5 and plan['learning_rate'] == 1e-4
    assert plan['checkpoint'] == 'runs/global_candidate_policy_formal_20261014/latest.pth'
    assert plan['input_sha256'][plan['checkpoint']] == 'b84d00454717cae56048753e0fe54c1d6b92cfcd2820c01fa65296508488ccb3'
    assert not any('sanity' in n and n.endswith('.pth') for n in plan['input_sha256'])
    for name in plan['runs'].values():
        assert name not in all_runs; all_runs.add(name)
    for name in plan['tasks'].values():
        assert name not in all_tasks; all_tasks.add(name)
    assert plan['source_review'] == entry['review'] and plan['source_review'] not in all_reviews
    all_reviews.add(plan['source_review'])
    assert plan['sanity_native_status'] == f"runs/pipeline_tasks/{plan['tasks']['sanity']}/status.json"
    source = f'scripts/train_candidate_joint_{family}_trajectory.py'
    tree = ast.parse((ROOT / source).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    native_asserts = [n for n in main.body if isinstance(n, ast.Assert) and any(isinstance(x, ast.Name) and x.id == 'own_native' for x in ast.walk(n))]
    env_asserts = [n for n in main.body if isinstance(n, ast.Assert) and 'os.environ' in ast.unparse(n)]
    b_asserts = [n for n in main.body if isinstance(n, ast.Assert) and any(isinstance(x, ast.Name) and x.id == 'b_native' for x in ast.walk(n))]
    assert len(native_asserts) == 3 and len(env_asserts) == 2 and len(b_asserts) == 1
    system = SimpleNamespace(executable='/cpu/python', argv=[source, '--plan', path, '--run', plan['runs']['formal']])
    ns = dict(plan=plan, task=plan['tasks']['formal'], sys=system,
        os=SimpleNamespace(getppid=lambda: 1234, environ={'CUDA_VISIBLE_DEVICES':str(entry['gpu']), 'CUDA_DEVICE_ORDER':'PCI_BUS_ID'}),
        own_native=dict(state='RUNNING', pid=1234, task=plan['tasks']['formal'], physical_gpu=str(entry['gpu']), command=[system.executable]+system.argv),
        b_native=dict(state='COMPLETE', exit_code=0, physical_gpu='1'))
    code = compile(ast.fix_missing_locations(ast.Module(body=env_asserts+native_asserts+b_asserts, type_ignores=[])), '<CPU-admission-assertions>', 'exec')
    exec(code, ns)
    cases.append(dict(plan_path=path, plan_sha256=entry['sha256'], source_count=len(expected_source), input_count=len(plan['input_sha256']), family=family, gpu=entry['gpu'], max_candidates=plan['max_candidates'], cost_weight=plan['cost_weight'], inherited_sources=len(base['source_sha256']), inherited_inputs=len(base['input_sha256']), changed_fields=sorted(changed), native_environment_and_historical_B_assertions='PASS'))

# Execute the exact wrapper bytes in an isolated trace root. Child only reads stdlib environment.
fixture = TRACE / 'cpu_wrapper_fixture_direct_python'
native_python = sys._base_executable
(fixture / 'scripts').mkdir(parents=True, exist_ok=False)
(fixture / WRAPPER).write_bytes((ROOT / WRAPPER).read_bytes())
child = fixture / 'cpu_child.py'
child.write_text("import json,os,sys\nfrom pathlib import Path\ntask,exitcode=sys.argv[1],int(sys.argv[2])\nr=json.loads((Path('runs/pipeline_tasks')/task/'status.json').read_text())\nassert r['state']=='RUNNING' and r['pid']==os.getppid()\nassert r['command']==[sys.executable]+sys.argv\nassert r['physical_gpu']==os.environ['CUDA_VISIBLE_DEVICES']\nassert os.environ['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'\nassert 'torch' not in sys.modules\nprint(json.dumps(dict(gpu=os.environ['CUDA_VISIBLE_DEVICES'],pid=os.getpid(),ppid=os.getppid(),torch_imported=False)))\nsys.exit(exitcode)\n", encoding='utf-8')
wrapper_results = []
for gpu, code in [(2, 0), (3, 7)]:
    task = f'CPU_ONLY_gpu{gpu}_exit{code}'
    command = [native_python, str(fixture / WRAPPER), '--task', task, '--gpu', str(gpu), '--', native_python, str(child), task, str(code)]
    result = subprocess.run(command, cwd=fixture, capture_output=True, text=True, timeout=20)
    record = read(fixture / 'runs/pipeline_tasks' / task / 'status.json')
    log = read(fixture / 'runs/pipeline_tasks' / task / 'run.log')
    assert result.returncode == record['exit_code'] == code, (result, record)
    assert record['state'] == ('COMPLETE' if code == 0 else 'FAILED')
    assert record['physical_gpu'] == log['gpu'] == str(gpu)
    assert log['ppid'] == record['pid'] and record['command'] == command[7:]
    assert not log['torch_imported'] and record['completed'] >= record['started']
    wrapper_results.append(dict(gpu=gpu, exit_code=code, state=record['state'], actual_parent_and_argv_verified=True, torch_imported=False))
assert sha(fixture / WRAPPER) == WRAPPER_SHA
assert 'torch' not in sys.modules
summary = dict(status='CPU_SOURCE_GRID_PASS', plan_index_sha256=sha(ROOT/INDEX), grid_sha256=sha(ROOT/GRID),
    wrapper_sha256=WRAPPER_SHA, wrapper_AST_delta_only_docstring_and_gpu_choices=True,
    unique_local_source_hashes=len(all_source_hashes), unique_local_AST_parses=len(all_source_hashes),
    current_local_input_hashes=all_local_inputs, remote_only_input_bindings_from_prior_sealed_receipt=remote_inputs,
    reused_remote_receipt_sha256=sha(receipt_path), cases=cases, wrapper_cpu_native_fixtures=wrapper_results,
    full_counts=receipt['formal_counts'], sanity_counts=receipt['sanity_counts'],
    no_audited_source_or_plan_edits=True, no_remote_queries=True, no_torch_import=True, no_neural_execution=True,
    runtime_capacity_verified=False, new_plan_actual_M0_verified=False,
    fixture_execution_note='First CPU fixture used a Windows venv redirector, whose child parent PID was the redirector rather than its caller. Separate stdlib reproduction confirmed caller48492/child38572/parent48736 versus direct-base caller48492/child33712/parent48492. Final fixtures use the direct base Python executable. Production source is unchanged; this is not a Linux runtime receipt.')
(TRACE / 'cpu_verification.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['current_local_input_hashes','remote_only_input_bindings_from_prior_sealed_receipt','cases']}, indent=2))
