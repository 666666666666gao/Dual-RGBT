"""Local inert source verification; no SSH, runtime samples, NN or wall-time waits."""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('C:/Users/gb/projects/Dual-RGBT')
OUT = Path(__file__).parent
source = 'tools/observe_vram18_capacity.py'
serial = 'tools/launch_vram18_serial_training_wave.py'
pipeline = 'scripts/complete_pipeline.py'
sha = lambda name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
trees = {name: ast.parse((ROOT / name).read_bytes()) for name in [source, serial, pipeline]}
for name, tree in trees.items(): compile(tree, name, 'exec')
assert sha(serial) == 'c1ef685e700b57d3a7754c4730edb889702abc235ab8b1b20e563eab217fb4b1'
assert sha(pipeline) == '370c629998b21dcdd210df4c7f61710286ce860b9a5caa02c5b412c85116105b'
main = next(n for n in trees[source].body if isinstance(n, ast.FunctionDef) and n.name == 'main')
cpu_assignment = next(n for n in main.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'cpu_code' for t in n.targets))
cpu_code = ast.literal_eval(cpu_assignment.value)
compile(cpu_code, 'exact_generated_cpu_code', 'exec')
formula_nodes = [n for n in main.body if 24 <= n.lineno <= 26]
formula = compile(ast.Module(body=formula_nodes, type_ignores=[]), 'exact_cpu_formula', 'exec')
results = []
for name, delta, expected in [('idle_only', [0, 0, 0, 70, 30, 0, 0, 0], 0), ('busy_only', [30, 10, 30, 0, 0, 10, 10, 10], 100), ('mixed_guest_excluded', [30, 0, 20, 40, 10, 0, 0, 0], 50)]:
    env = dict(before=dict(cpu_ticks=[100] * 10, observed=10), after=dict(cpu_ticks=[100 + d for d in delta] + [99999, 99999], observed=23), snapshot={})
    exec(formula, env)
    assert env['snapshot']['cpu_sample']['busy_percent'] == expected
    results.append(dict(fixture=name, status='PASS', busy_percent=expected))

events = []; now = [1000]; calls = [0]
before = dict(observed=1000, cpu_ticks=[100, 0, 100, 800, 20, 0, 0, 0, 500, 100])
after = dict(observed=1013, cpu_ticks=[130, 0, 120, 840, 30, 0, 0, 0, 700, 300])
plan_path = str((OUT / 'inert_node_plan.json').relative_to(ROOT)).replace(chr(92), '/')
output_path = str((OUT / 'inert_output.json').relative_to(ROOT)).replace(chr(92), '/')
node = dict(scope='SYNTHETIC_LOCAL_TEST_ONLY', observe_at_epoch=1000, observer_sha256=sha(source), output=output_path, tasks=[dict(task='INERT_TASK_%d' % n, pid=n + 1, gpu=n % 4) for n in range(24)])
(ROOT / plan_path).write_text(json.dumps(node), encoding='utf-8')
raw = dict(scope='SYNTHETIC_LOCAL_TEST_ONLY', observed=1013, tasks=copy.deepcopy(node['tasks']), meminfo_kib={'MemAvailable': 12345, 'SwapFree': 6789}, gpu_csv=['INERT_GPU_DATA'])
def wait_until(deadline):
    assert deadline == node['observe_at_epoch']
    events.append('wait_until'); now[0] = deadline
def sleep(seconds):
    assert seconds == 10
    events.append('sleep10'); now[0] += seconds
def remote(code):
    assert now[0] >= node['observe_at_epoch'] and code == cpu_code
    calls[0] += 1; events.append('cpu_before' if calls[0] == 1 else 'cpu_after')
    assert calls[0] <= 2
    return copy.deepcopy(before if calls[0] == 1 else after)
def measurement(tasks):
    assert tasks == node['tasks'] and events == ['wait_until', 'cpu_before', 'sleep10']
    events.append('measurement'); now[0] += 3
    return copy.deepcopy(raw)
namespace = dict(argparse=argparse, ROOT=ROOT, Path=Path, __file__=str(ROOT/source), hashlib=hashlib, json=json,
                 remote=remote, measurement=measurement, wait_until=wait_until, time=SimpleNamespace(sleep=sleep))
exec(compile(ast.Module(body=[main], type_ignores=[]), 'exact_observer_main', 'exec'), namespace)
sys.argv = [source, '--node', plan_path]
with contextlib.redirect_stdout(io.StringIO()): namespace['main']()
actual = json.loads((ROOT / output_path).read_bytes())
assert events == ['wait_until', 'cpu_before', 'sleep10', 'measurement', 'cpu_after']
assert {key: actual[key] for key in raw} == raw
assert actual['cpu_sample']['before'] == before and actual['cpu_sample']['after'] == after
assert actual['cpu_sample']['busy_percent'] == 50 and actual['cpu_sample']['interval_seconds'] == 13
assert actual['node_plan_sha256'] == sha(plan_path)
result = dict(status='PASS', source_only=True, runtime_certified=False, source_sha256={name: sha(name) for name in trees},
              generated_cpu_code_parses=True, arithmetic_fixtures=results, inert_events=events,
              registered_tasks=24, raw_cpu_endpoints_preserved=True, same_measurement_meminfo_gpu_tasks_preserved=True,
              interval_includes_snapshot_overhead=True, network_calls=0, nn_calls=0)
(OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
