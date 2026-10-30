"""Launch the matched original-loss comparator without changing bound sources."""
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


def write_new(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2) + '\n')


def launch(plan, run, script, gpu, smoke_frames=None):
    memory = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        universal_newlines=True).splitlines()
    assert int(memory[int(gpu)]) < 500
    task = ('train_' if script.startswith('train_') else 'eval_') + run
    assert not (ROOT / 'runs' / run).exists()
    assert not (ROOT / 'runs/pipeline_tasks' / task).exists()
    plan_path = ROOT / 'reports' / (run + '_plan.json')
    write_new(plan_path, plan)
    python = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
    command = [python, str(ROOT / 'scripts/run_task_four_gpu.py'), '--task', task,
               '--gpu', gpu, '--', python, str(ROOT / 'scripts' / script),
               '--plan', plan_path.relative_to(ROOT).as_posix(),
               '--run', run, '--task', task]
    if smoke_frames is not None:
        command += ['--smoke-frames', str(smoke_frames)]
    with (ROOT / 'reports' / (run + '_wrapper.log')).open('xb') as handle:
        process = subprocess.Popen(command, cwd=str(ROOT),
                                   env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'),
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    entry = dict(run=run, task=task, gpu=gpu, pid=process.pid, command=command,
                 plan_sha256=sha(plan_path), started=time.time())
    write_new(ROOT / 'reports' / (run + '_launch.json'), entry)
    return entry


def terminal(run, prefix, receipt):
    native_path = ROOT / 'runs/pipeline_tasks' / (prefix + run) / 'status.json'
    native = json.loads(native_path.read_text())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    result_path = ROOT / 'runs' / run / receipt
    return result_path, native_path, json.loads(result_path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('smoke', 'formal', 'eval-smoke', 'dev'), required=True)
    parser.add_argument('--gpu', choices=('0', '1', '2', '3'), required=True)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--smoke-attempt', default='')
    args = parser.parse_args()
    stem = 'cached_query_gain_parent_loss'
    smoke_run = stem + '_largebatch_smoke_20261030'
    formal_run = stem + '_formal_20261030'
    if args.phase in ('smoke', 'formal'):
        plan = json.loads((ROOT / 'reports/CACHED_QUERY_GAIN_UTILITY_FOCAL_FORMAL_PLAN_20261030.json').read_text())
        plan['variant'] = 'parent_loss'
        for key in ('smoke_result', 'smoke_result_sha256', 'smoke_native_status', 'smoke_native_sha256'):
            plan.pop(key)
        if args.phase == 'smoke':
            plan.update(mode='smoke', formal_launch_allowed=False, min_epochs=2, max_epochs=2)
            run = smoke_run
        else:
            result_path, native_path, result = terminal(smoke_run, 'train_', 'COMPLETE.json')
            assert result['status'] == 'SANITY_PASS' and result['strict_reload_bit_equal']
            plan.update(smoke_result=result_path.relative_to(ROOT).as_posix(),
                        smoke_result_sha256=sha(result_path),
                        smoke_native_status=native_path.relative_to(ROOT).as_posix(),
                        smoke_native_sha256=sha(native_path))
            run = formal_run
        entry = launch(plan, run, 'train_cached_query_gain.py', args.gpu)
    else:
        assert args.epoch is not None
        result_path, native_path, result = terminal(formal_run, 'train_', 'COMPLETE.json')
        assert result['status'] == 'TRAIN_COMPLETE_PENDING_DEV'
        checkpoint = next(item for item in result['retained'] if item['epoch'] == args.epoch)
        plan = json.loads((ROOT / 'reports/CACHED_QUERY_GAIN_UTILITY_FOCAL_E10_EVAL_SMOKE_PLAN_20261030.json').read_text())
        plan.update(checkpoint=checkpoint['checkpoint'], checkpoint_sha256=checkpoint['sha256'],
                    training_task='train_' + formal_run,
                    training_native_sha256=sha(native_path), training_result_sha256=sha(result_path))
        eval_smoke_run = stem + '_e%d_eval_smoke_20261030' % args.epoch + args.smoke_attempt
        if args.phase == 'eval-smoke':
            run, frames = eval_smoke_run, 11
        else:
            smoke_path, smoke_native, smoke = terminal(eval_smoke_run, 'eval_', 'SMOKE.json')
            assert smoke['checkpoint_sha256'] == checkpoint['sha256']
            plan.update(full_evaluation_allowed=True,
                        smoke_result=smoke_path.relative_to(ROOT).as_posix(),
                        smoke_result_sha256=sha(smoke_path),
                        smoke_native_status=smoke_native.relative_to(ROOT).as_posix(),
                        smoke_native_sha256=sha(smoke_native))
            run, frames = stem + '_e%d_lasher_dev_20261030' % args.epoch, 0
        entry = launch(plan, run, 'eval_cached_query_gain.py', args.gpu, frames)
        if frames == 0:
            output = 'reports/cached_query_gain_parent_loss_dev_20261030/e%d.json' % args.epoch
            command = [str(ROOT / 'conda-envs/cpu-scoring/bin/python'),
                       str(ROOT / 'tools/score_size_tail.py'), '--run', run,
                       '--output', output, '--wait']
            with (ROOT / 'reports' / (run + '_scorer.log')).open('xb') as handle:
                process = subprocess.Popen(command, cwd=str(ROOT), stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            entry.update(scorer_pid=process.pid, scorer_command=command)
            write_new(ROOT / 'reports' / (run + '_scorer_launch.json'), entry)
    print(json.dumps(entry, indent=2))


if __name__ == '__main__':
    main()
