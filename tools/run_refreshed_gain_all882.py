"""Admit the all882 head experiment and score every retained weight on full dev."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
from joint_evaluation_contract import check_files
from prepare_refreshed_gain_all882 import PREFIX, RUN_PREFIX, REVIEW, PROTOCOL

STATUS = ROOT / 'reports/REFRESHED_GAIN_ALL882_QUEUE_20261030.json'
PYTHON = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
CPU = str(ROOT / 'conda-envs/cpu-scoring/bin/python')
STATE = dict(pid=os.getpid(), poll_seconds=180, completed=[], failed=[])


def publish(phase, **values):
    STATE.update(phase=phase, updated=time.time(), **values)
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(STATE, indent=2) + '\n')
    temporary.replace(STATUS)


def free_memory(gpu):
    values = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        universal_newlines=True).splitlines()
    return int(values[int(gpu)])


def launch(run, plan, script, gpu, required_mib, frames=None):
    available = free_memory(gpu)
    assert available >= required_mib
    task = ('train_' if script.startswith('train_') else 'eval_') + run
    assert not (ROOT / 'runs' / run).exists()
    assert not (ROOT / 'runs/pipeline_tasks' / task).exists()
    command = [PYTHON, str(ROOT / 'scripts/run_task_four_gpu.py'),
               '--task', task, '--gpu', gpu, '--', PYTHON,
               str(ROOT / 'scripts' / script), '--plan', plan,
               '--run', run, '--task', task]
    if frames is not None:
        command += ['--smoke-frames', str(frames)]
    with (ROOT / 'reports' / (run + '_wrapper.log')).open('xb') as handle:
        process = subprocess.Popen(command, cwd=str(ROOT),
            env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'),
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    entry = dict(run=run, task=task, pid=process.pid, gpu=gpu,
                 command=command, plan_sha256=sha(ROOT / plan),
                 started=time.time(), observed_free_mib=available,
                 required_free_mib=required_mib)
    return process, entry


def terminal(process, entry):
    if process.poll() is None:
        return False
    path = ROOT / 'runs/pipeline_tasks' / entry['task'] / 'status.json'
    native = json.loads(path.read_text())
    if native['state'] != 'COMPLETE' or native['exit_code'] != 0:
        STATE['failed'].append(dict(entry=entry, native=native))
        publish('FAILED', status='FAILED')
        raise RuntimeError('Native task failed: ' + entry['task'])
    return True


def train(mode, required_mib):
    subprocess.run([PYTHON, str(ROOT / 'tools/prepare_refreshed_gain_all882.py'),
                    '--mode', mode], cwd=str(ROOT), check=True)
    plan = 'reports/' + PREFIX + '_' + mode.upper() + '_PLAN_20261030.json'
    run = RUN_PREFIX + '_' + mode + '_20261030'
    process, entry = launch(run, plan, 'train_refreshed_query_gain.py', '2', required_mib)
    publish('TRAIN_' + mode.upper(), status='RUNNING', active=[entry])
    while not terminal(process, entry):
        time.sleep(180)
        publish('TRAIN_' + mode.upper(), active=[entry])
    result_path = ROOT / 'runs' / run / 'COMPLETE.json'
    result = json.loads(result_path.read_text())
    STATE['completed'].append(dict(entry, result_sha256=sha(result_path)))
    return result, entry


def evaluation_plan(checkpoint, training, stage):
    template = ROOT / 'reports/CACHED_QUERY_GAIN_UTILITY_FOCAL_E10_EVAL_SMOKE_PLAN_20261030.json'
    plan = json.loads(template.read_text())
    native_path = ROOT / 'runs/pipeline_tasks' / training['task'] / 'status.json'
    result_path = ROOT / 'runs' / training['run'] / 'COMPLETE.json'
    assert sha(ROOT / checkpoint['checkpoint']) == checkpoint['sha256']
    plan.update(checkpoint=checkpoint['checkpoint'], checkpoint_sha256=checkpoint['sha256'],
                training_task=training['task'], training_native_sha256=sha(native_path),
                training_result_sha256=sha(result_path),
                source_review=REVIEW, source_review_sha256=sha(ROOT / REVIEW),
                full_evaluation_allowed=stage == 'dev')
    stem = RUN_PREFIX + '_e%03d' % checkpoint['epoch']
    smoke_run = stem + '_eval_smoke_20261030'
    if stage == 'dev':
        smoke_path = ROOT / 'runs' / smoke_run / 'SMOKE.json'
        smoke_native = ROOT / 'runs/pipeline_tasks' / ('eval_' + smoke_run) / 'status.json'
        smoke = json.loads(smoke_path.read_text())
        assert smoke['checkpoint_sha256'] == checkpoint['sha256']
        assert (smoke['sequences'], smoke['frames']) == (1, 11)
        plan.update(smoke_result=smoke_path.relative_to(ROOT).as_posix(),
                    smoke_result_sha256=sha(smoke_path),
                    smoke_native_status=smoke_native.relative_to(ROOT).as_posix(),
                    smoke_native_sha256=sha(smoke_native))
    run = smoke_run if stage == 'smoke' else stem + '_lasher_dev_20261030'
    path = ROOT / 'reports' / (run + '_plan.json')
    with path.open('x') as handle:
        handle.write(json.dumps(plan, indent=2) + '\n')
    return run, path.relative_to(ROOT).as_posix()


def evaluate(result, training):
    assert result['status'] == 'TRAIN_COMPLETE_PENDING_DEV'
    queue = [dict(checkpoint=item, stage='smoke') for item in result['retained']]
    assert len(queue) == 5
    active, processes = {}, {}
    slots = {'a': '2', 'b': '3', 'c': '3'}
    while queue or active:
        for slot, entry in list(active.items()):
            if not terminal(processes[slot], entry):
                continue
            STATE['completed'].append(entry)
            if entry['stage'] == 'smoke':
                queue.insert(0, dict(checkpoint=entry['checkpoint'], stage='dev'))
            else:
                output = 'reports/refreshed_gain_all882_dev_20261030/e%03d.json' % entry['checkpoint']['epoch']
                subprocess.run([CPU, str(ROOT / 'tools/score_size_tail.py'),
                                '--run', entry['run'], '--output', output],
                               cwd=str(ROOT), check=True)
                score = json.loads((ROOT / output).read_text())
                assert score['status'] == 'SCORED_COMPLETE'
                entry.update(score=output, score_sha256=sha(ROOT / output),
                             metrics={key: score[key] for key in ('PR', 'NPR', 'SR')})
            del active[slot], processes[slot]
        for slot, gpu in slots.items():
            if slot in active or not queue or free_memory(gpu) < 3072:
                continue
            job = queue.pop(0)
            run, plan = evaluation_plan(job['checkpoint'], training, job['stage'])
            process, entry = launch(run, plan, 'eval_cached_query_gain.py', gpu, 3072,
                                    11 if job['stage'] == 'smoke' else 0)
            entry.update(job)
            processes[slot], active[slot] = process, entry
        publish('DEV', status='RUNNING' if queue or active else 'COMPLETE',
                active=list(active.values()), queue=queue)
        if queue or active:
            time.sleep(180)


def main():
    assert not STATUS.exists()
    review = json.loads((ROOT / REVIEW).read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    check_files(ROOT, review['source_sha256'])
    assert review['protocol_sha256'] == sha(ROOT / PROTOCOL)
    refresh_path = ROOT / 'reports/POLICY_REFRESH_E6_QUEUE_20261030.json'
    while True:
        refresh = json.loads(refresh_path.read_text())
        assert not refresh['failed']
        if refresh['status'] == 'COMPLETE':
            break
        assert refresh['status'] == 'RUNNING'
        assert (Path('/proc') / str(refresh['pid'])).exists()
        publish('WAIT_E6_CACHE', status='RUNNING', active=[],
                prerequisite_pid=refresh['pid'], prerequisite_sha256=sha(refresh_path))
        time.sleep(180)
    smoke, _ = train('smoke', 8192)
    smoke_plan = json.loads((ROOT / ('reports/' + PREFIX + '_SMOKE_PLAN_20261030.json')).read_text())
    rejected = []
    for part in ('fit', 'holdout'):
        for metric, field in (('sampled_recoverable_states', 'minimum_recovery_states'),
                              ('recovery_sequences', 'minimum_recovery_sequences')):
            observed, minimum = smoke['baseline'][part][metric], smoke_plan[field][part]
            if observed < minimum:
                rejected.append(dict(partition=part, metric=metric, observed=observed, minimum=minimum))
    if rejected:
        publish('ADMISSION_REJECTED', status='ADMISSION_REJECTED', active=[], reasons=rejected)
        return
    required = int(math.ceil(smoke['peak_reserved_bytes'] / (1024 ** 2))) + 1024
    result, training = train('formal', required)
    evaluate(result, training)


if __name__ == '__main__':
    main()
