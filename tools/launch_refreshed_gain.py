"""Launch one admitted refreshed-cache stage on a measured fitting GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha


def launch(plan_path, run, gpu):
    memory = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        universal_newlines=True).splitlines()
    # The matched control reserved 572 MiB; allow 1 GiB for this cached head.
    assert int(memory[int(gpu)]) >= 1024
    task = 'train_' + run
    assert not (ROOT / 'runs' / run).exists()
    assert not (ROOT / 'runs/pipeline_tasks' / task).exists()
    python = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
    command = [python, str(ROOT / 'scripts/run_task_four_gpu.py'), '--task', task,
               '--gpu', gpu, '--', python, str(ROOT / 'scripts/train_refreshed_query_gain.py'),
               '--plan', plan_path.relative_to(ROOT).as_posix(),
               '--run', run, '--task', task]
    with (ROOT / 'reports' / (run + '_wrapper.log')).open('xb') as handle:
        process = subprocess.Popen(
            command, cwd=str(ROOT), env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'),
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    entry = dict(run=run, task=task, gpu=gpu, pid=process.pid, command=command,
                 plan_sha256=sha(plan_path), started=time.time(),
                 observed_free_mib=int(memory[int(gpu)]), reserved_budget_mib=1024)
    with (ROOT / 'reports' / (run + '_launch.json')).open('x') as handle:
        handle.write(json.dumps(entry, indent=2) + '\n')
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'formal'), required=True)
    parser.add_argument('--gpu', choices=('0', '1', '2', '3'), required=True)
    args = parser.parse_args()
    path = ROOT / ('reports/REFRESHED_GAIN_PARENT_LOSS_%s_PLAN_20261030.json' % args.mode.upper())
    plan = json.loads(path.read_text())
    assert plan['mode'] == args.mode and plan['launch_allowed']
    run = 'refreshed_query_gain_parent_loss_%s_20261030' % args.mode
    entry = launch(path, run, args.gpu)
    print(json.dumps(entry, indent=2))


if __name__ == '__main__':
    main()
