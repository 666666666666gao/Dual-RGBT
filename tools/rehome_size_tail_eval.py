"""Replace only a verified CPU controller, preserving its live native GPU child."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
from joint_evaluation_contract import check_files
from run_size_tail_shared_eval import identity, task_terminal

CHAINS = {
    'size_tail_b576_formal_epochs_20261030':
        ('reports/SIZE_TAIL_EPOCH_FORMAL_PLAN_20261030.json', 'tools/run_size_tail_epochs.py'),
    'size_tail_b640_formal_epochs_20261030':
        ('reports/SIZE_TAIL_B640_FORMAL_PLAN_20261030.json', 'tools/run_size_tail_epochs_keep_five.py'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', choices=tuple(CHAINS), required=True)
    parser.add_argument('--eval-gpu', choices=('2', '3'), required=True)
    args = parser.parse_args()
    review_path = ROOT / 'reports/SIZE_TAIL_SHARED_EVAL_SOURCE_REVIEW_20261030.json'
    review = json.loads(review_path.read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    check_files(ROOT, review['source_sha256'])
    directory = ROOT / 'runs' / args.chain
    state_path = directory / 'status.json'
    state = json.loads(state_path.read_text())
    assert state['status'] == 'RUNNING' and state['phase'] in ('TRAIN', 'DEV')
    plan, original_controller = CHAINS[args.chain]
    assert sha(ROOT / plan) == state['plan_sha256']
    old_pid, child_pid = state['controller_pid'], state['active_pid']
    command = (Path('/proc') / str(old_pid) / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    assert command == [str(ROOT / 'conda-envs/dual_rgbt/bin/python'),
                       str(ROOT / original_controller), '--plan', plan,
                       '--name', args.chain, '--gpu', state['physical_gpu']]
    native_path = ROOT / 'runs/pipeline_tasks' / state['active_task'] / 'status.json'
    task_terminal(json.loads(native_path.read_text()), child_pid)
    handoff = dict(
        chain=args.chain, plan=plan, eval_gpu=args.eval_gpu, identity=identity(state),
        source_review=review_path.relative_to(ROOT).as_posix(),
        source_review_sha256=sha(review_path), previous_controller_command=command,
        created=time.time())
    handoff_path = directory / 'shared_eval_handoff.json'
    with handoff_path.open('x') as handle:
        handle.write(json.dumps(handoff, indent=2) + '\n')
    ready_path = directory / 'shared_eval_ready.json'
    assert not ready_path.exists()
    new_command = [str(ROOT / 'conda-envs/dual_rgbt/bin/python'),
                   str(ROOT / 'tools/run_size_tail_shared_eval.py'),
                   '--handoff', handoff_path.relative_to(ROOT).as_posix()]
    with (directory / 'shared_eval_controller.log').open('xb') as handle:
        process = subprocess.Popen(new_command, cwd=str(ROOT),
            env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'),
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + 10
    while not ready_path.exists() or ready_path.stat().st_size == 0:
        assert process.poll() is None and time.time() < deadline
        time.sleep(.1)
    ready = json.loads(ready_path.read_text())
    assert ready['pid'] == process.pid and ready['handoff_sha256'] == sha(handoff_path)
    assert ready['status'] == 'READY_TO_ADOPT'
    assert ready['identity'] == identity(json.loads(state_path.read_text())) == handoff['identity']
    # This is a CPU scheduler replacement, never a signal to a GPU child/group.
    os.kill(old_pid, signal.SIGTERM)
    task_terminal(json.loads(native_path.read_text()), child_pid)
    receipt = dict(
        status='TAKEOVER_DISPATCHED', previous_controller_pid=old_pid,
        new_controller_pid=process.pid, command=new_command,
        preserved_task=state['active_task'], preserved_wrapper_pid=child_pid,
        original_plan_sha256=state['plan_sha256'], train_gpu=state['physical_gpu'],
        future_eval_gpu=args.eval_gpu, handoff_sha256=sha(handoff_path),
        source_review_sha256=sha(review_path), terminated_only_cpu_controller=True,
        created=time.time())
    with (directory / 'shared_eval_takeover.json').open('x') as handle:
        handle.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
