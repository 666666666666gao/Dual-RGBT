"""Local-only source/receipt binding and inert resume lifecycle verification."""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('C:/Users/gb/projects/Dual-RGBT')
OUT = Path(__file__).parent
read = lambda name: json.loads((ROOT / name).read_bytes())
sha = lambda name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
resume_name = 'tools/resume_vram18_serial_training_wave.py'
serial_name = 'tools/launch_vram18_serial_training_wave.py'
v1_name = 'tools/launch_vram18_training_wave.py'
index_name = 'reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json'
archive_name = 'reports/VRAM18_WAVE6_INTERRUPTED_EXECUTION_20261027.json'
execution_name = 'reports/VRAM18_WAVE6_EXECUTION_20261027.json'
index = read(index_name)
review = read('reports/VRAM18_SERIAL_CONTROLLER_SOURCE_REVIEW_20261027.json')
state0 = read(archive_name)
prior = read('reports/VRAM18_WAVE4_EXECUTION_20261027.json')
health = read('reports/VRAM18_WAVE4_HEALTH_20261027.json')
diag = read('reports/VRAM18_WAVE6_PARTIAL_NATIVE_DIAGNOSIS_20261027.json')
assert sha(index_name) == review['index_sha256'] == state0['index_sha256'] == health['index_sha256']
assert sha(serial_name) == review['launcher_sha256'] == state0['controller_sha256']
assert sha(v1_name) == review['preserved_v1_launcher_sha256']
assert sha(archive_name) == sha(execution_name)
assert health['execution_sha256'] == sha('reports/VRAM18_WAVE4_EXECUTION_20261027.json')
assert health['status'] == 'ADMIT_NEXT_WAVE' and prior['state'] == 'FORMAL_STARTUP_PASS'
preserved = {}
for key, mapping in review.items():
    if key.startswith('preserved_') and isinstance(mapping, dict):
        assert all(sha(name) == value for name, value in mapping.items()), key
        preserved[key] = len(mapping)
assert state0['state'] == 'FORMAL_IDENTITY_PERSISTED' and state0['phase'] == 6
cases = state0['cases']
expected = {row['path']: row for row in index['plans'] if row['phase'] == 6}
assert len(cases) == 8 and {c['plan'] for c in cases} == set(expected)
assert sum('formal' in c for c in cases) == 3
existing = {c['formal']['task']: c['formal'] for c in cases if 'formal' in c}
prior_owners = [dict(e, cuda_ready=True) for e in index['original_owners']]
prior_owners += [dict(task=c['formal']['task'], pid=c['formal']['pid'], gpu=c['gpu'], cuda_ready=True) for c in prior['cases']]
assert len(prior_owners) == 16 and len({c['task'] for c in prior_owners}) == 16
diagnosed = {row['task']: row for row in diag['tasks']}
assert set(diagnosed) == {r['task'] for r in prior_owners} | set(existing)
for owner in prior_owners + list(existing.values()):
    actual = diagnosed[owner['task']]
    assert actual['native']['pid'] == owner['pid'] and actual['native']['state'] == 'RUNNING' and actual['alive']
    assert len(actual['children']) == 1 and actual['updates'] > 0
for case in cases:
    plan = read(case['plan']); guard = read(plan['source_review']); receipt = case['sanity_admission']
    assert case['plan_sha256'] == expected[case['plan']]['sha256'] == sha(case['plan']) == guard['plan_sha256']
    assert case['gpu'] == expected[case['plan']]['gpu'] == plan['physical_gpu']
    assert case['tasks'] == plan['tasks'] and case['runs'] == plan['runs']
    assert guard['status'] == 'SOURCE_PASS' and guard['blocking_issues'] == 0 and guard['source_sha256'] == plan['source_sha256']
    assert all(sha(name) == value for name, value in plan['source_sha256'].items())
    n = receipt['native']; a = receipt['complete']; s = receipt['probe']
    assert n['state'] == 'COMPLETE' and n['exit_code'] == 0 and n['physical_gpu'] == str(case['gpu'])
    assert n['task'] == case['tasks']['sanity'] and n['pid'] == case['sanity']['pid']
    assert a['sanity'] and {k: a[k] for k in plan['sanity_expected_counts']} == plan['sanity_expected_counts']
    assert a['selected_execution_bit_equal'] and a['teacher_preserves_tracker_state']
    assert s['status'] == 'PASS' and s['strict_reload_prediction_bit_equal'] and s['frozen_state_tensors'] == 251 and len(s['changed_state_tensors']) == 14
trees = {name: ast.parse((ROOT / name).read_bytes()) for name in [resume_name, serial_name, v1_name]}
for name, tree in trees.items():
    compile(tree, name, 'exec')
functions = lambda tree: {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
rf = functions(trees[resume_name]); sf = functions(trees[serial_name]); vf = functions(trees[v1_name])
for name in ['measurement', 'score_health', 'launch', 'admit', 'startup']:
    assert ast.dump(rf[name]) == ast.dump(sf[name]), name
for name in ['measurement', 'score_health']:
    assert ast.dump(rf[name]) == ast.dump(vf[name]), name
schedule = [n for n in rf['main'].body if n.lineno >= 168]
profile = read(index['host_profile'])
programs = []
def fixture(failure=None):
    state = copy.deepcopy(state0); cases = state['cases']; owners = copy.deepcopy(prior_owners)
    events = []; snapshots = []; elapsed = [0]; launches = []; startup_calls = [0]
    class Report:
        def write_text(self, text):
            snapshots.append(json.loads(text)); events.append('persist')
    def sleep(seconds):
        assert seconds == 180
        elapsed[0] += seconds; events.append('sleep')
    def remote(code):
        tree = ast.parse(code)
        value = lambda key: next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == key for t in n.targets))
        case = value('case'); plan = read(case['plan'])
        if code.startswith('import hashlib,json\n'):
            events.append('admit')
            result = copy.deepcopy(case['sanity_admission'])
            if failure == 'admission_mismatch': result['complete_sha256'] = '0' * 64
        elif code.startswith('import csv,json,math,time\n'):
            startup_calls[0] += 1; events.append('startup')
            if failure == 'reattach_failure' and startup_calls[0] == 3: raise AssertionError('INERT_REATTACH_FAILURE')
            if failure == 'new_startup_failure' and len(launches) == 1: raise AssertionError('INERT_STARTUP_FAILURE')
            result = dict(child_pid=case['formal']['pid'] + 1, logged_updates=25, finite_logged_losses=True)
        else:
            assert value('new_slots') == 1 and value('limit') == 6 and all(o['cuda_ready'] for o in value('owned'))
            assert not any(n['task'] == case['tasks']['formal'] for n in value('owned'))
            assert case['tasks']['formal'] not in existing and case['tasks']['formal'] not in {o['task'] for o in prior_owners}
            if failure == 'third_launch_gate_failure' and len(launches) == 2: raise AssertionError('INERT_CAPACITY_FAILURE')
            popen = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'Popen')
            command = ast.literal_eval(popen.args[0])
            assert command[-2:] == ['--sanity-complete-sha256', case['sanity_admission']['complete_sha256']]
            assert '--sanity' not in command
            result = dict(task=plan['tasks']['formal'], pid=3000000 + len(launches), gpu=case['gpu'], command=command, launched=2000000000 + elapsed[0])
            launches.append(result); events.append('launch')
        if failure is None: programs.append(code)
        return result
    def measurement(owned):
        assert len(owned) == (19 if not launches else 24) and all(row['cuda_ready'] for row in owned)
        events.append('measurement'); return dict(observed=2000000000 + elapsed[0], tasks=[])
    ns = dict(ROOT=ROOT, Path=Path, json=json, REMOTE='/data2/gb/Dual-RGBT', PYTHON='/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python',
              cases=cases, state=state, owned=owners, temporary=[], args=SimpleNamespace(phase=6), compact_peak=profile['cases'][1]['peak_sampled_pss_kib'],
              report_path=Report(), time=SimpleNamespace(sleep=sleep), remote=remote, measurement=measurement)
    exec(compile(ast.Module(body=[rf[n] for n in ['launch', 'admit', 'startup']], type_ignores=[]), 'exact_resume_helpers', 'exec'), ns)
    caught = False
    with contextlib.redirect_stdout(io.StringIO()):
        try: exec(compile(ast.Module(body=schedule, type_ignores=[]), 'exact_resume_schedule', 'exec'), ns)
        except AssertionError:
            if failure is None: raise
            caught = True
    for i, event in enumerate(events):
        if event == 'launch': assert events[i + 1] == 'persist'
    if failure is None:
        assert not caught and state['state'] == 'FORMAL_STARTUP_PASS'
        assert len(launches) == 5 and [v['gpu'] for v in launches] == [3, 0, 1, 2, 3]
        assert set(existing) | {v['task'] for v in launches} == {c['tasks']['formal'] for c in cases}
        assert elapsed[0] == 900 and events.count('admit') == 8 and events.count('startup') == 8
        assert all(next(c['formal'] for c in cases if c['tasks']['formal'] == name) == value for name, value in existing.items())
    else:
        assert caught and len(launches) == {'admission_mismatch': 0, 'reattach_failure': 0, 'new_startup_failure': 1, 'third_launch_gate_failure': 2}[failure]
        if launches: assert sum('formal' in c for c in snapshots[-1]['cases']) == 3 + len(launches)
    return dict(fixture=failure or 'success', status='PASS', new_formals=len(launches), inert_wait_seconds=elapsed[0], events=events)
fixtures = [fixture(name) for name in [None, 'admission_mismatch', 'reattach_failure', 'new_startup_failure', 'third_launch_gate_failure']]
assert len(programs) == 21
result = dict(status='PASS', scope='LOCAL_SOURCE_AND_SAVED_RECEIPTS_ONLY; inert programs were never executed remotely', runtime_certified=False,
              source_sha256={name: sha(name) for name in trees}, index_sha256=sha(index_name), archive_sha256=sha(archive_name), preserved_maps=preserved,
              original_and_prior_formals=16, reattached_phase6_formals=3, remaining_phase6_formals=5, existing_native0_m0_receipts=8,
              exact_serial_helper_ast_match=True, generated_remote_programs_parsed=21, fixtures=fixtures,
              startup_delay_metadata_seconds=180)
(OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'fixtures'}, indent=2))
