"""Wait for a free allowed GPU, then finish one smoke and three full dev runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_inputs import sha
from complete_pipeline import ROOT, REMOTE, SCP, launch, remote, wait_until


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--toolkit', required=True)
    args = parser.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_bytes())
    review = json.loads((ROOT / plan['source_review']).read_bytes())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(path) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['local_input_sha256'].items())
    evaluation = json.loads((ROOT / plan['evaluation_plan']).read_bytes())
    toolkit = Path(args.toolkit)
    assert all(sha(toolkit / name) == value for name, value in evaluation['toolkit_code_sha256'].items())
    status_path = ROOT / plan['status_path']
    assert not status_path.exists()
    state = dict(state='STARTING', controller_pid=os.getpid(), started=time.time(),
                 plan_sha256=sha(path), tasks=[], metrics=[])

    def save():
        status_path.write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)

    def sleep_until(seconds):
        state['next_observation_epoch'] = time.time() + seconds
        save()
        wait_until(state['next_observation_epoch'])

    def free_gpu():
        first_busy = True
        while True:
            observed = remote("import json,subprocess\ns=subprocess.check_output(['nvidia-smi','-i','0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],universal_newlines=True)\nprint(json.dumps([tuple(int(v.strip()) for v in line.split(',')) for line in s.strip().splitlines()]))")
            state['last_gpu_memory_mib'] = observed
            available = [index for index, used in observed if index in (0, 1) and used < 500]
            if available:
                return available[0]
            state['state'] = 'WAITING_FOR_FREE_GPU'
            sleep_until(plan['first_busy_wait_seconds'] if first_busy else plan['resource_poll_seconds'])
            first_busy = False

    def finish(entry, task, smoke):
        sleep_until(plan['smoke_wait_seconds'] if smoke else plan['full_dev_first_wait_seconds'])
        while True:
            observed = remote("""import json,time
from pathlib import Path
r=Path(%r)
n=json.loads((r/'runs/pipeline_tasks'/%r/'status.json').read_bytes())
result=dict(native=n,observed=time.time())
if n['state']=='RUNNING':
 p=Path('/proc')/str(n['pid'])
 result['wrapper_command']=p.joinpath('cmdline').read_bytes().decode().split('\\0')[:-1]
 progress=r/'runs'/%r/'progress.json'
 result['done_frames']=sum(row['frames'] for row in json.loads(progress.read_bytes())) if progress.exists() else 0
print(json.dumps(result))
""" % (REMOTE, entry['task'], entry['run']))
            native = observed['native']
            assert native['task'] == entry['task'] and native['pid'] == task['pid']
            assert native['physical_gpu'] == str(task['gpu'])
            state['last_task_observation'] = observed
            save()
            if native['state'] != 'RUNNING':
                assert native['state'] == 'COMPLETE' and native['exit_code'] == 0, native
                return native
            wrapper = observed['wrapper_command']
            assert wrapper[1:5] == ['scripts/run_task.py', '--task', entry['task'], '--gpu']
            assert wrapper[5] == str(task['gpu']) and wrapper[6] == '--'
            assert wrapper[7:] == native['command']
            done = observed['done_frames']
            if smoke or not done:
                delay = plan['resource_poll_seconds']
            else:
                remaining = (observed['observed'] - native['started']) * (54646 - done) / done
                delay = max(180, min(3600, remaining - 180))
            sleep_until(delay)

    save()
    for key in ('smoke', '0.0', '0.05', '0.15'):
        entry = evaluation['runs'][key]
        gpu = free_gpu()
        command = [REMOTE + '/scripts/run_rankstop_dev.py', '--plan', plan['evaluation_plan'],
                   '--task', entry['task'], '--run', entry['run'], '--cost-weight', str(entry['cost_weight'])]
        smoke = key == 'smoke'
        if smoke:
            command.append('--smoke')
        task = launch(entry['task'], gpu, command)
        state['tasks'].append(task)
        state['state'] = 'SMOKE_RUNNING' if smoke else 'FULL_DEV_RUNNING'
        save()
        native = finish(entry, task, smoke)
        destination = ROOT / 'runs' / entry['run']
        assert not destination.exists()
        subprocess.run(SCP + ['-r', '2026:' + REMOTE + '/runs/' + entry['run'], str(destination)], check=True)
        local_task = ROOT / 'runs/pipeline_tasks' / entry['task']
        local_task.mkdir(parents=True, exist_ok=False)
        (local_task / 'status.json').write_text(json.dumps(native, indent=2))
        if smoke:
            receipt = json.loads((destination / 'SMOKE.json').read_bytes())
            assert receipt['reload']['status'] == 'PASS'
        else:
            subprocess.run([sys.executable, str(ROOT / 'tools/score_rankstop_dev.py'),
                            '--plan', plan['evaluation_plan'], '--cost-weight', key, '--toolkit', args.toolkit],
                           cwd=str(ROOT), check=True)
            state['metrics'].append(json.loads((ROOT / entry['metrics']).read_bytes()))
        save()
    best = max(state['metrics'], key=lambda row: (row['SR'], row['PR'], row['NPR'], -row['mean_ms']))
    state.update(state='FULL_DEV_COMPLETE_PENDING_RESULT_AUDIT', best_cost_weight=best['cost_weight'],
                 best_metrics={key: best[key] for key in ('PR', 'NPR', 'SR')},
                 exceeds_e1_dev_sr=best['SR'] > evaluation['selection']['reference_dev']['SR'],
                 completed=time.time(), test_launched=False)
    save()


if __name__ == '__main__':
    main()
