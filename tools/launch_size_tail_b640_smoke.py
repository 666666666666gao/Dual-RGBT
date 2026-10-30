"""Measure the requested larger batch with the unchanged admitted trainer."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
from joint_evaluation_contract import check_files


def main():
    name = 'size_tail_b640_lr8e5_smoke_20261030'
    task = 'train_' + name
    plan_path = ROOT / 'reports/SIZE_TAIL_B640_SMOKE_PLAN_20261030.json'
    assert not plan_path.exists()
    assert not (ROOT / 'runs' / name).exists()
    assert not (ROOT / 'runs/pipeline_tasks' / task).exists()
    plan = json.loads((ROOT / 'reports/SIZE_TAIL_EPOCH_RESUME_SANITY_PLAN_20261030.json').read_text())
    check_files(ROOT, plan['source_sha256'])
    used = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        universal_newlines=True)
    assert int(used.splitlines()[0]) < 500
    plan.update(status='B640_REAL_UPDATE_MEMORY_SMOKE', batch_size=640, lr=0.00008,
                epochs=1, samples_per_epoch=2560, formal_launch_allowed=False,
                epoch_definition='Four real TRAIN updates; numerical and memory admission only',
                memory_choice='B576 reserved18.77GiB; B640 targets about20.9GiB. Actual result determines admission.',
                retention='Preserve this smoke separately; formal training must retain the latest five checkpoints',
                test_eligible=False)
    with plan_path.open('x') as handle:
        handle.write(json.dumps(plan, indent=2) + '\n')
    python = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
    command = [python, str(ROOT / 'scripts/run_task_four_gpu.py'), '--task', task, '--gpu', '0',
               '--', python, str(ROOT / 'scripts/train_size_tail_epoch.py'),
               '--plan', plan_path.relative_to(ROOT).as_posix(), '--run', name,
               '--task', task, '--epoch', '1', '--sanity']
    with (ROOT / 'reports/SIZE_TAIL_B640_SMOKE_WRAPPER_20261030.log').open('xb') as handle:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'),
                                   start_new_session=True)
    receipt = dict(name=name, task=task, gpu='0', pid=process.pid, command=command,
                   plan_sha256=sha(plan_path), started=time.time(), batch_size=640, lr=0.00008)
    with (ROOT / 'reports/SIZE_TAIL_B640_SMOKE_LAUNCH_20261030.json').open('x') as handle:
        handle.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
