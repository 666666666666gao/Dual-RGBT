"""Bind completed controls and launch the existing registered size-tail chain."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
from joint_evaluation_contract import check_files, evaluation_plan


def main():
    plan = json.loads((ROOT / 'reports/SIZE_TAIL_EPOCH_PILOT_DRAFT_20261030.json').read_text())
    parent, _, _ = evaluation_plan(ROOT, plan['parent_evaluation_plan'])
    assert parent['checkpoint_sha256'] == plan['parent_checkpoint_sha256']
    check_files(ROOT, plan['source_sha256'])
    controls = {}
    for label, name in (('parent', 'parent_control'), ('b384', 'b384_sanity'), ('b576', 'b576_sanity')):
        path = ROOT / 'reports/size_tail_dev_20261030' / (name + '.json')
        report = json.loads(path.read_text())
        assert report['status'] == 'SCORED_COMPLETE'
        assert (report['sequences'], report['frames']) == (97, 54646)
        config_name = next(key for key in report['run_artifacts_sha256'] if key == 'config.json')
        # Each control is a separate full dev run, never an epoch mix.
        run_names = {'parent': 'size_tail_parent_control_dev_20261030',
                     'b384': 'size_tail_b384_dev_20261030',
                     'b576': 'size_tail_b576_dev_20261030'}
        run = ROOT / 'runs' / run_names[label]
        assert sha(run / config_name) == report['run_artifacts_sha256'][config_name]
        check_files(run, report['run_artifacts_sha256'])
        assert sha(path.with_suffix('.npz')) == report['curves_sha256']
        native = ROOT / 'runs/pipeline_tasks' / ('eval_' + run.name) / 'status.json'
        assert sha(native) == report['native_status_sha256']
        controls[label] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                               **{key: report[key] for key in ('PR', 'NPR', 'SR')})
    smoke_path = ROOT / 'runs/size_tail_sanity_b576_20261030/COMPLETE.json'
    smoke = json.loads(smoke_path.read_text())
    assert smoke['status'] == 'SANITY_PASS' and smoke['strict_reload_bit_equal']
    assert smoke['updates'] == 4 and smoke['trainable_parameters'] == 66
    assert smoke['peak_reserved_bytes'] < 22 * 1024 ** 3
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                   '--format=csv,noheader,nounits'], universal_newlines=True)
    assert int(used.splitlines()[1]) < 500
    plan.update(status='ADMITTED_FORMAL_SIZE_ONLY', formal_launch_allowed=True,
                parent_control_score_sha256=controls['parent']['sha256'],
                diagnostic_score_sha256=controls['b576']['sha256'],
                verified_controls=controls, max_existing_gpu_mib=500,
                test_eligible=False,
                admission_decision='Parent exact replay and both four-update controls verified. B576 SR is0.334849pp below parent; this is not convergence evidence. Run preregistered multi-epoch size-only chain, stop on declared full-dev regression/plateau rules.',
                convergence_caveat='Minimum-five/maximum-ten sampled epochs; every epoch full97dev. Not a full pass over every frame or a backbone fine-tune.',
                expected_minutes_per_training_epoch=30, expected_minutes_per_dev=85)
    path = ROOT / 'reports/SIZE_TAIL_EPOCH_FORMAL_PLAN_20261030.json'
    with path.open('x') as handle:
        handle.write(json.dumps(plan, indent=2) + '\n')
    name = 'size_tail_b576_formal_epochs_20261030'
    assert not (ROOT / 'runs' / name).exists()
    command = [str(ROOT / 'conda-envs/dual_rgbt/bin/python'),
               str(ROOT / 'tools/run_size_tail_epochs.py'), '--plan',
               path.relative_to(ROOT).as_posix(), '--name', name, '--gpu', '1']
    with (ROOT / 'reports/SIZE_TAIL_FORMAL_CONTROLLER_20261030.log').open('xb') as handle:
        process = subprocess.Popen(command, cwd=str(ROOT), env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'),
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=process.pid, gpu='1', name=name, command=command,
                   plan_sha256=sha(path), started=time.time(), batch_size=576, lr=plan['lr'])
    with (ROOT / 'reports/SIZE_TAIL_FORMAL_CONTROLLER_LAUNCH_20261030.json').open('x') as handle:
        handle.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
