"""Run remaining comparator epochs on independently available server26 GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha


def launch(job, gpu):
    epoch, phase = job['epoch'], job['phase']
    formal = 'cached_query_gain_parent_loss_formal_20261030'
    result_path = ROOT / 'runs' / formal / 'COMPLETE.json'
    result = json.loads(result_path.read_text())
    assert result['status'] == 'TRAIN_COMPLETE_PENDING_DEV'
    training_native = ROOT / 'runs/pipeline_tasks' / ('train_' + formal) / 'status.json'
    native = json.loads(training_native.read_text())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    checkpoint = next(item for item in result['retained'] if item['epoch'] == epoch)
    base = ROOT / 'reports/CACHED_QUERY_GAIN_UTILITY_FOCAL_E10_EVAL_SMOKE_PLAN_20261030.json'
    plan = json.loads(base.read_text())
    mapping = ROOT / 'reports/CACHED_PARENT_LOSS_SERVER26_MAPPING_20261030.json'
    plan.update(checkpoint=checkpoint['checkpoint'], checkpoint_sha256=checkpoint['sha256'],
                training_task='train_' + formal, training_native_sha256=sha(training_native),
                training_result_sha256=sha(result_path),
                runtime_provider=dict(ssh_alias='2026', project=str(ROOT)),
                dataset_mapping_receipt=mapping.relative_to(ROOT).as_posix(),
                dataset_mapping_receipt_sha256=sha(mapping))
    plan['source_sha256']['tools/run_cached_gain_server26.py'] = sha(ROOT / 'tools/run_cached_gain_server26.py')
    smoke_run = 'cached_query_gain_parent_loss_e%d_server26_eval_smoke_20261030' % epoch
    if phase == 'smoke':
        run, frames = smoke_run, 11
    else:
        run, frames = 'cached_query_gain_parent_loss_e%d_server26_lasher_dev_20261030' % epoch, 0
        smoke_result = ROOT / 'runs' / smoke_run / 'SMOKE.json'
        smoke_native = ROOT / 'runs/pipeline_tasks' / ('eval_' + smoke_run) / 'status.json'
        terminal = json.loads(smoke_native.read_text())
        assert terminal['state'] == 'COMPLETE' and terminal['exit_code'] == 0
        plan.update(full_evaluation_allowed=True,
                    smoke_result=smoke_result.relative_to(ROOT).as_posix(),
                    smoke_result_sha256=sha(smoke_result),
                    smoke_native_status=smoke_native.relative_to(ROOT).as_posix(),
                    smoke_native_sha256=sha(smoke_native))
    task = 'eval_' + run
    assert not (ROOT / 'runs' / run).exists()
    assert not (ROOT / 'runs/pipeline_tasks' / task).exists()
    plan_path = ROOT / 'reports' / (run + '_plan.json')
    with plan_path.open('x') as handle:
        handle.write(json.dumps(plan, indent=2) + '\n')
    python = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
    command = [python, str(ROOT / 'scripts/run_task_four_gpu.py'), '--task', task,
               '--gpu', gpu, '--', python, str(ROOT / 'tools/run_cached_gain_server26.py'),
               '--plan', plan_path.relative_to(ROOT).as_posix(),
               '--run', run, '--task', task, '--smoke-frames', str(frames)]
    with (ROOT / 'reports' / (run + '_wrapper.log')).open('xb') as handle:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=handle,
                                   env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'),
                                   stderr=subprocess.STDOUT, start_new_session=True)
    entry = dict(job, gpu=gpu, run=run, task=task, pid=process.pid,
                 command=command, plan_sha256=sha(plan_path), started=time.time())
    with (ROOT / 'reports' / (run + '_launch.json')).open('x') as handle:
        handle.write(json.dumps(entry, indent=2) + '\n')
    return entry, process


def main():
    output = ROOT / 'reports/CACHED_PARENT_LOSS_QUEUE_SERVER26_20261030.json'
    assert not output.exists()
    queue = [dict(epoch=epoch, phase='smoke') for epoch in (8, 9)]
    active, processes, completed, failed = {}, {}, [], []
    while queue or active:
        for gpu, entry in list(active.items()):
            if processes[gpu].poll() is None:
                continue
            native = json.loads((ROOT / 'runs/pipeline_tasks' / entry['task'] / 'status.json').read_text())
            if native['state'] == 'COMPLETE' and native['exit_code'] == 0:
                completed.append(entry)
                if entry['phase'] == 'smoke':
                    queue.insert(0, dict(epoch=entry['epoch'], phase='dev'))
                else:
                    command = [str(ROOT / 'conda-envs/cpu-scoring/bin/python'),
                               str(ROOT / 'tools/score_size_tail.py'), '--run', entry['run'],
                               '--output', 'reports/cached_query_gain_parent_loss_dev_20261030/e%d.json' % entry['epoch']]
                    with (ROOT / 'reports' / (entry['run'] + '_scorer.log')).open('xb') as handle:
                        scorer = subprocess.Popen(command, cwd=str(ROOT), stdout=handle,
                                                  stderr=subprocess.STDOUT, start_new_session=True)
                    entry.update(scorer_pid=scorer.pid, scorer_command=command)
            else:
                failed.append(dict(entry=entry, native=native))
            del active[gpu], processes[gpu]
        memory = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            universal_newlines=True).splitlines()
        for index, used in enumerate(memory):
            gpu = str(index)
            if gpu in active or int(used) >= 500 or not queue:
                continue
            job = queue.pop(0)
            active[gpu], processes[gpu] = launch(job, gpu)
        state = dict(pid=os.getpid(), queue=queue, active=active, completed=completed,
                     failed=failed, updated=time.time(), poll_seconds=180,
                     status='RUNNING' if queue or active else 'COMPLETE_WITH_FAILURES' if failed else 'COMPLETE')
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(output)
        if queue or active:
            time.sleep(180)


if __name__ == '__main__':
    main()
