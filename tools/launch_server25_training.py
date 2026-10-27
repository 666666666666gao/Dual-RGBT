"""Admit the eight new25 tasks without restarting any existing task or queue."""
import argparse
import base64
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from server25_runtime import ROOT, REMOTE, PYTHON, remote

INDEX = 'reports/SERVER25_REMAINING_GRID_20261027.json'
EXECUTION = 'reports/SERVER25_REMAINING_GRID_EXECUTION_20261027.json'
CONTROLLER_TASK = 'admit_server25_remaining_grid_20261027'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases():
    index = json.loads((ROOT / INDEX).read_bytes())
    review = json.loads((ROOT / index['source_review']).read_bytes())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['index_sha256'] == sha(ROOT / INDEX)
    accepted = json.loads((ROOT / index['environment_acceptance']).read_bytes())
    assert accepted['status'] == 'ENVIRONMENT_ACCEPTED'
    cases = []
    for row in index['plans']:
        plan = json.loads((ROOT / row['path']).read_bytes())
        guard = json.loads((ROOT / plan['source_review']).read_bytes())
        assert row['sha256'] == guard['plan_sha256'] == sha(ROOT / row['path'])
        assert guard['status'] == 'SOURCE_PASS' and guard['blocking_issues'] == 0
        assert guard['source_sha256'] == plan['source_sha256']
        assert all(sha(ROOT / n) == h for n, h in plan['source_sha256'].items())
        cases.append(dict(plan_path=row['path'], plan=plan))
    assert len(cases) == 8 and all(sum(c['plan']['physical_gpu'] == g for c in cases) == 2 for g in range(4))
    return index, cases


def admission(cases):
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], universal_newlines=True).strip()
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                      '--format=csv,noheader,nounits'], universal_newlines=True).split()
    assert len(memory) == 4 and all(int(v) < 500 for v in memory)
    available = int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines()
                         if v.startswith('MemAvailable:')))
    assert available > (8 * 6 + 16) * 1024 * 1024
    fs = os.statvfs(str(ROOT))
    assert fs.f_bavail * fs.f_frsize > 10 * 1024 ** 3
    for c in cases:
        assert all(sha(ROOT / n) == h for n, h in c['plan']['input_sha256'].items())
    return dict(observed=time.time(), available_kib=available,
                free_bytes=fs.f_bavail * fs.f_frsize)


def startup(case):
    plan = case['plan']
    native = json.loads((ROOT / 'runs/pipeline_tasks' / plan['tasks']['formal'] / 'status.json').read_bytes())
    assert native['state'] == 'RUNNING' and native['pid'] == case['formal']['pid']
    assert native['command'] == case['formal']['command'][7:]
    parent = Path('/proc') / str(native['pid'])
    assert parent.joinpath('cmdline').read_bytes().decode().split(chr(0))[:-1] == case['formal']['command']
    children = (parent / 'task' / str(native['pid']) / 'children').read_text().split()
    assert len(children) == 1
    child = Path('/proc') / children[0]
    assert child.joinpath('cmdline').read_bytes().decode().split(chr(0))[:-1] == native['command']
    env = dict(v.split('=', 1) for v in child.joinpath('environ').read_bytes().decode().split(chr(0)) if v)
    assert env['CUDA_VISIBLE_DEVICES'] == str(plan['physical_gpu'])
    directory = ROOT / 'runs' / plan['runs']['formal']
    config = json.loads((directory / 'config.json').read_bytes())
    assert not config['sanity'] and config['plan_sha256'] == sha(ROOT / case['plan_path'])
    assert config['source_sha256'] == plan['source_sha256'] and config['input_sha256'] == plan['input_sha256']
    with (directory / 'trajectory_steps.csv').open(newline='') as f:
        rows = [v for v in csv.DictReader(f) if None not in v.values()]
    assert rows and len(rows) == int(rows[-1]['step'])
    assert all(all(math.isfinite(float(v[k])) for k in v if k == 'loss' or k.endswith('_loss')) for v in rows)
    return dict(native=native, child_pid=int(children[0]), logged_updates=len(rows),
                last_update=rows[-1], finite_logged_losses=True,
                host_memory={v.split(':', 1)[0]:v.split(':', 1)[1].strip()
                             for v in child.joinpath('smaps_rollup').read_text().splitlines()[1:]})


def worker():
    index, cases = load_cases()
    output = ROOT / EXECUTION
    assert not output.exists()
    state = dict(state='STARTING_SANITY_BATCH', cases=cases, source_index_sha256=sha(ROOT / INDEX),
                 sanity_wait_seconds=300, formal_wait_seconds=180, started=time.time())
    def save():
        output.write_text(json.dumps(state, indent=2) + '\n')
    def launch(stage):
        state[stage + '_admission'] = admission(cases)
        for case in cases:
            p = case['plan']
            task, run = p['tasks'][stage], p['runs'][stage]
            assert not (ROOT / 'runs/pipeline_tasks' / task).exists() and not (ROOT / 'runs' / run).exists()
            command = [PYTHON, p['launch_wrapper'], '--task', task, '--gpu', str(p['physical_gpu']),
                       '--', PYTHON, REMOTE+'/'+p['launch_trainer'], '--plan', case['plan_path'], '--run', run]
            command += ['--sanity'] if stage == 'sanity' else ['--sanity-complete-sha256', case['sanity_admission']['complete_sha256']]
            with (ROOT / 'reports' / (task + '.launcher.log')).open('wb') as log:
                process = subprocess.Popen(command, cwd=str(ROOT), env=dict(os.environ, **p['execution_env']),
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            case[stage] = dict(pid=process.pid, command=command, launched=time.time())
            save()  # Persist partial identities before another task can start.
    launch('sanity')
    state['state'] = 'SANITY_BATCH_LAUNCHED'; save()
    time.sleep(300)
    for c in cases:
        p = c['plan']; directory = ROOT / 'runs' / p['runs']['sanity']
        n = json.loads((ROOT / p['sanity_native_status']).read_bytes())
        complete = json.loads((directory / 'COMPLETE.json').read_bytes())
        probe = json.loads((directory / 'SANITY.json').read_bytes())
        config = json.loads((directory / 'config.json').read_bytes())
        assert n['state'] == 'COMPLETE' and n['exit_code'] == 0 and n['pid'] == c['sanity']['pid']
        assert n['physical_gpu'] == str(p['physical_gpu'])
        assert complete['sanity'] and {k:complete[k] for k in p['sanity_expected_counts']} == p['sanity_expected_counts']
        assert complete['selected_execution_bit_equal'] and complete['teacher_preserves_tracker_state']
        assert probe['status'] == 'PASS' and probe['strict_reload_prediction_bit_equal']
        assert probe['frozen_state_tensors'] == 251 and len(probe['changed_state_tensors']) == 14
        assert config['plan_sha256'] == sha(ROOT / c['plan_path'])
        assert config['source_sha256'] == p['source_sha256'] and config['input_sha256'] == p['input_sha256']
        c['sanity_admission'] = dict(native=n, complete=complete, probe=probe,
                                     complete_sha256=sha(directory / 'COMPLETE.json'))
        save()
    state['state'] = 'EIGHT_SANITIES_COMPLETE'; save()
    launch('formal')
    state['state'] = 'FORMAL_BATCH_LAUNCHED'; save()
    time.sleep(180)
    for c in cases:
        c['startup'] = startup(c); save()
    state.update(state='EIGHT_FORMALS_STARTUP_PASS', observed=time.time(),
                 next_observation_epoch=max(c['formal']['launched'] for c in cases)+1800,
                 gpu_csv=subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                                  '--format=csv,noheader,nounits'], universal_newlines=True).splitlines())
    save()
    print(json.dumps(dict(state=state['state'], cases=8, next_observation_epoch=state['next_observation_epoch'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(); return
    index, cases = load_cases()
    assert not (ROOT / EXECUTION).exists()
    names = {INDEX, index['source_review'], index['environment_acceptance']}
    for case in cases:
        p = case['plan']; names.update([case['plan_path'], p['source_review']]); names.update(p['source_sha256'])
    payload = {n:base64.b64encode((ROOT / n).read_bytes()).decode() for n in names}
    code = """import json,base64,subprocess,time
from pathlib import Path
r=Path(%r);payload=%r
for n,v in payload.items():
 p=r/n;b=base64.b64decode(v)
 if p.exists():assert p.read_bytes()==b,n
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
task=%r;assert not (r/'runs/pipeline_tasks'/task).exists() and not (r/%r).exists()
command=[%r,'scripts/run_task_four_gpu.py','--task',task,'--gpu','0','--',%r,str(r/'tools/launch_server25_training.py'),'--worker']
with (r/'reports/server25_grid_controller.launcher.log').open('wb') as log:
 p=subprocess.Popen(command,cwd=str(r),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(task=task,pid=p.pid,command=command,launched=time.time(),first_observation_after_seconds=600)))
""" % (REMOTE, payload, CONTROLLER_TASK, EXECUTION, PYTHON, PYTHON)
    receipt = remote(code)
    (ROOT / 'reports/SERVER25_GRID_CONTROLLER_LAUNCH_20261027.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
