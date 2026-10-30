"""Admit the measured B640 run and retain its last five full-dev checkpoints."""
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
    smoke_run = ROOT / 'runs/size_tail_b640_lr8e5_smoke_20261030'
    smoke_path = smoke_run / 'COMPLETE.json'
    smoke = json.loads(smoke_path.read_text())
    native_path = ROOT / 'runs/pipeline_tasks/train_size_tail_b640_lr8e5_smoke_20261030/status.json'
    native = json.loads(native_path.read_text())
    config = json.loads((smoke_run / 'config.json').read_text())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert native['task'] == config['task'] == 'train_' + smoke_run.name
    python = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
    smoke_plan = 'reports/SIZE_TAIL_B640_SMOKE_PLAN_20261030.json'
    assert native['command'] == [
        python, str(ROOT / 'scripts/train_size_tail_epoch.py'), '--plan', smoke_plan,
        '--run', smoke_run.name, '--task', native['task'], '--epoch', '1', '--sanity']
    assert sha(ROOT / smoke_plan) == config['plan_sha256']
    assert json.loads((ROOT / smoke_plan).read_text()) == config['plan']
    assert smoke['status'] == 'SANITY_PASS' and smoke['strict_reload_bit_equal']
    assert sha(smoke_run / 'epoch_001.pth') == smoke['checkpoint_sha256']
    assert smoke['updates'] == 4 and smoke['optimizer_steps_verified']
    assert set(smoke['changed_state_tensors']) == {
        'executor.box_head.conv5_size.weight', 'executor.box_head.conv5_size.bias'}
    assert config['sanity'] and config['plan']['batch_size'] == 640 and config['plan']['lr'] == 0.00008
    assert smoke['peak_reserved_bytes'] < 23 * 1024 ** 3
    check_files(ROOT, config['plan']['source_sha256'])
    plan = json.loads((ROOT / 'reports/SIZE_TAIL_EPOCH_FORMAL_PLAN_20261030.json').read_text())
    check_files(ROOT, plan['source_sha256'])
    assert plan['parent_checkpoint_sha256'] == config['plan']['parent_checkpoint_sha256']
    assert sha(ROOT / plan['parent_control_score']) == plan['parent_control_score_sha256']
    assert sha(ROOT / plan['diagnostic_score']) == plan['diagnostic_score_sha256']
    used = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        universal_newlines=True)
    assert int(used.splitlines()[0]) < 500
    controller = 'tools/run_size_tail_epochs_keep_five.py'
    del plan['source_sha256']['tools/run_size_tail_epochs.py']
    plan['source_sha256'][controller] = sha(ROOT / controller)
    plan.update(status='ADMITTED_B640_SIZE_ONLY', batch_size=640, lr=0.00008,
                samples_per_epoch=9600, formal_launch_allowed=True,
                epoch_definition='9600 deterministic TRAIN pairs per epoch; 15 updates, not an exhaustive frame pass',
                retention='Retain the latest five weights and every full97dev score; select SR/PR/NPR using dev only',
                smoke_result=smoke_path.relative_to(ROOT).as_posix(), smoke_result_sha256=sha(smoke_path),
                smoke_native_status=native_path.relative_to(ROOT).as_posix(), smoke_native_sha256=sha(native_path),
                memory_choice='B640 four-update smoke measured; no concurrent workload on this training GPU',
                comparison_caveat='B576 uses9216pairs/epoch; B640 uses9600 for integral batches. This is not an equal-budget causal batch comparison.',
                test_eligible=False)
    path = ROOT / 'reports/SIZE_TAIL_B640_FORMAL_PLAN_20261030.json'
    with path.open('x') as handle:
        handle.write(json.dumps(plan, indent=2) + '\n')
    name = 'size_tail_b640_formal_epochs_20261030'
    assert not (ROOT / 'runs' / name).exists()
    command = [str(ROOT / 'conda-envs/dual_rgbt/bin/python'), str(ROOT / controller),
               '--plan', path.relative_to(ROOT).as_posix(), '--name', name, '--gpu', '0']
    with (ROOT / 'reports/SIZE_TAIL_B640_FORMAL_CONTROLLER_20261030.log').open('xb') as handle:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'),
                                   start_new_session=True)
    result = dict(pid=process.pid, gpu='0', name=name, command=command,
                  plan_sha256=sha(path), started=time.time(), batch_size=640, lr=0.00008)
    with (ROOT / 'reports/SIZE_TAIL_B640_FORMAL_LAUNCH_20261030.json').open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
