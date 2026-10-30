"""Register and launch separate fixed-state dev diagnostics on server25."""
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
from joint_evaluation_contract import evaluation_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('parent', 'e6'), required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--gpu', choices=('0', '1', '2', '3'), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    available = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                         '--format=csv,noheader,nounits']).decode().splitlines()
    assert int(available[int(args.gpu)]) < 500
    base = json.loads((ROOT / 'reports/CACHED_QUERY_GAIN_UTILITY_FOCAL_E6_DEV_PLAN_20261030.json').read_text())
    parent, spec, _ = evaluation_plan(ROOT, base['parent_evaluation_plan'])
    assert ROOT.as_posix() == spec['provider']['project']
    source = 'scripts/diagnose_fixed_state_dev.py'
    names = ['belowrightwhiteboy', 'moto2north1', 'midboyblue', 'browncar2north']
    assert args.sequence in names
    model = args.model
    reference = 'size_tail_parent_control_dev_20261030' if model == 'parent' else \
                'cached_query_gain_utility_focal_e6_lasher_dev_20261030'
    checkpoint_sha = parent['checkpoint_sha256'] if model == 'parent' else base['checkpoint_sha256']
    run = 'fixed_state_%s_%s_%s_20261030' % (model, args.sequence, 'smoke' if args.smoke else 'full')
    task = 'diagnose_' + run
    plan_path = ROOT / 'reports' / (run + '_plan.json')
    assert not plan_path.exists() and not (ROOT / 'runs' / run).exists()
    plan = dict(mode='fixed_state_dev_diagnostic', training_allowed=False, test_eligible=False,
                parent_evaluation_plan=base['parent_evaluation_plan'],
                parent_checkpoint_sha256=parent['checkpoint_sha256'],
                checkpoint_sha256=checkpoint_sha, overlay=None if model == 'parent' else base['checkpoint'],
                manifest_sha256=sha(ROOT / spec['datasets']['LasHeR']['dev']),
                source_sha256={source: sha(ROOT / source)}, reference_run=reference,
                sequence_names=names, cost_weight=.15, label_stride=5, smoke_frames=11)
    if not args.smoke:
        smoke_run = 'fixed_state_%s_browncar2north_smoke_20261030' % model
        smoke_result = ROOT / 'runs' / smoke_run / 'SMOKE.json'
        smoke_native = ROOT / 'runs/pipeline_tasks' / ('diagnose_' + smoke_run) / 'status.json'
        terminal = json.loads(smoke_native.read_text())
        assert terminal['state'] == 'COMPLETE' and terminal['exit_code'] == 0
        result = json.loads(smoke_result.read_text())
        assert result['status'] == 'SMOKE_PASS' and result['observer_control_equal']
        assert result['source_sha256'] == plan['source_sha256']
        plan.update(smoke_result=smoke_result.relative_to(ROOT).as_posix(),
                    smoke_result_sha256=sha(smoke_result),
                    smoke_native_status=smoke_native.relative_to(ROOT).as_posix(),
                    smoke_native_sha256=sha(smoke_native))
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    py = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
    command = [py, str(ROOT / 'scripts/run_task_four_gpu.py'), '--task', task, '--gpu',
               args.gpu, '--', py, str(ROOT / source), '--plan',
               plan_path.relative_to(ROOT).as_posix(), '--run', run, '--task', task,
               '--sequence', args.sequence]
    if args.smoke:
        command.append('--smoke')
    with (ROOT / 'reports' / (run + '_wrapper.log')).open('wb') as handle:
        process = subprocess.Popen(command, cwd=str(ROOT),
                                   env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'),
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=process.pid, task=task, run=run, physical_gpu=args.gpu,
                   command=command, plan_sha256=sha(plan_path), started=time.time())
    (ROOT / 'reports' / (run + '_launch.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
