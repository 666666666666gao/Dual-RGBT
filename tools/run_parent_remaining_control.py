"""Collect the missing parent-policy TRAIN control with reviewed native jobs."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
from joint_evaluation_contract import evaluation_plan

REGISTRY = ROOT / 'reports/PARENT_REMAINING782_CONTROL_PLAN_20261030.json'
STATUS = ROOT / 'reports/PARENT_REMAINING782_CONTROL_QUEUE_20261030.json'


def prepare():
    refresh = json.loads((ROOT / 'reports/POLICY_REFRESH_E6_ALL882_PLAN_20261030.json').read_text())
    server = refresh['server']
    slots = {'26': {0: '0', 2: '1', 4: '2', 6: '3'},
             '25': {1: '2', 3: '2', 5: '3', 7: '3'}}[server]
    jobs, selected = [], []
    for item in refresh['jobs']:
        if item['phase'] != 'remaining782':
            continue
        old_path = ROOT / item['plan']
        assert sha(old_path) == item['plan_sha256']
        old = json.loads(old_path.read_text())
        parent, _, _ = evaluation_plan(ROOT, old['parent_evaluation_plan'])
        assert parent['checkpoint_sha256'] == old['parent_checkpoint_sha256']
        assert sha(ROOT / old['manifest']) == old['manifest_sha256']
        manifest = json.loads((ROOT / old['manifest']).read_text())
        names = sorted(row['name'] for row in manifest['train'])
        random.Random(2027).shuffle(names)
        fit, held = old['fit_sequences'], old['holdout_sequences']
        assert set(fit + held) <= set(names[100:])
        assert not set(fit) & set(held)
        assert set(held) == set(fit + held) & set(names[100:][4::5])
        selected.extend(fit + held)
        plan = {key: old[key] for key in (
            'parent_evaluation_plan', 'manifest', 'manifest_sha256', 'seed',
            'cost_weight', 'label_stride', 'smoke_frames', 'fit_sequences',
            'holdout_sequences', 'shard', 'expected_frames')}
        plan.update(mode='frozen_train_query_cache',
                    checkpoint_sha256=parent['checkpoint_sha256'],
                    source_sha256={name: sha(ROOT / name) for name in (
                        'scripts/collect_frozen_query_features.py',
                        'tools/summarize_frozen_query_cache.py')},
                    training_allowed=False, test_eligible=False,
                    full_collection_allowed=False,
                    matched_refresh_plan=item['plan'],
                    matched_refresh_plan_sha256=item['plan_sha256'],
                    scope='Remaining782 TRAIN parent-policy control. Same sequence partition as E6 refresh; original100 not recollected.')
        name = 'parent_remaining782_control_s%d_20261030' % item['shard']
        path = ROOT / 'reports' / (name + '_smoke_plan.json')
        with path.open('x') as handle:
            handle.write(json.dumps(plan, indent=2) + '\n')
        jobs.append(dict(name=name, plan=path.relative_to(ROOT).as_posix(),
                         plan_sha256=sha(path), gpu=slots[item['shard']],
                         sequences=len(fit + held), frames=item['frames']))
    assert len(jobs) == 4 and len(selected) == len(set(selected))
    result = dict(status='REGISTERED_SMOKE_REQUIRED', server=server, jobs=jobs,
                  sequences=len(selected), frames=sum(job['frames'] for job in jobs),
                  source_sha256=sha(Path(__file__)), poll_seconds=180,
                  training_allowed=False, test_eligible=False,
                  ownership='26 even shards on GPU0/1/2/3;25 odd shards on GPU2/2/3/3. GPU0/1 on25 remain reserved for size training.')
    with REGISTRY.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def launch(entry):
    assert sha(ROOT / entry['plan']) == entry['plan_sha256']
    run = entry['name'] + '_' + entry['stage']
    task = 'collect_' + run
    assert not (ROOT / 'runs' / run).exists()
    assert not (ROOT / 'runs/pipeline_tasks' / task).exists()
    python = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
    command = [python, str(ROOT / 'scripts/run_task_four_gpu.py'),
               '--task', task, '--gpu', entry['gpu'], '--', python,
               str(ROOT / 'scripts/collect_frozen_query_features.py'),
               '--plan', entry['plan'], '--run', run, '--task', task]
    if entry['stage'] == 'smoke':
        command.append('--smoke')
    with (ROOT / 'reports' / (run + '_wrapper.log')).open('xb') as handle:
        process = subprocess.Popen(command, cwd=str(ROOT),
            env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'),
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    return process, dict(entry, run=run, task=task, pid=process.pid,
                         command=command, started=time.time())


def audit(entry):
    report = 'reports/' + entry['run'] + '_audit.json'
    result = subprocess.run([str(ROOT / 'conda-envs/cpu-scoring/bin/python'),
        str(ROOT / 'tools/summarize_frozen_query_cache.py'),
        '--run', entry['run'], '--task', entry['task'], '--output', report], cwd=str(ROOT))
    if result.returncode:
        return None
    plan = json.loads((ROOT / entry['plan']).read_text())
    progress = json.loads((ROOT / 'runs' / entry['run'] / 'progress.json').read_text())
    manifest = json.loads((ROOT / plan['manifest']).read_text())
    by_name = {row['name']: row for row in manifest['train']}
    names = plan['fit_sequences'] + plan['holdout_sequences']
    expected = names[:1] if entry['stage'] == 'smoke' else names
    assert [row['sequence'] for row in progress] == expected
    for row in progress:
        sequence = by_name[row['sequence']]
        count = plan['smoke_frames'] if entry['stage'] == 'smoke' else len(sequence['rgb'])
        assert row['frames'] == count
        assert row['sampled_states'] == sum(
            frame in sequence['valid_frames'] for frame in range(plan['label_stride'], count, plan['label_stride']))
    return report


def run_queue():
    assert not STATUS.exists()
    registry = json.loads(REGISTRY.read_text())
    assert registry['source_sha256'] == sha(Path(__file__))
    processes, active, completed, failed = {}, {}, [], []
    for job in registry['jobs']:
        process, entry = launch(dict(job, stage='smoke'))
        processes[job['name']], active[job['name']] = process, entry
    while active:
        for name, entry in list(active.items()):
            # Native status is read only after its owning wrapper has exited.
            if processes[name].poll() is None:
                continue
            native_path = ROOT / 'runs/pipeline_tasks' / entry['task'] / 'status.json'
            native = json.loads(native_path.read_text())
            report = audit(entry) if native['state'] == 'COMPLETE' and native['exit_code'] == 0 else None
            del active[name], processes[name]
            if report is None:
                failed.append(dict(entry=entry, native=native, phase='NATIVE_OR_AUDIT'))
                continue
            completed.append(dict(entry, audit=report, audit_sha256=sha(ROOT / report)))
            if entry['stage'] == 'smoke':
                result_path = ROOT / 'runs' / entry['run'] / 'SMOKE.json'
                result = json.loads(result_path.read_text())
                assert result['observer_parent_trajectory_equal']
                assert result['frozen_state_tensors'] == 265
                plan = json.loads((ROOT / entry['plan']).read_text())
                plan.update(full_collection_allowed=True,
                            smoke_plan=entry['plan'],
                            smoke_result=result_path.relative_to(ROOT).as_posix(),
                            smoke_result_sha256=sha(result_path),
                            smoke_native_status=native_path.relative_to(ROOT).as_posix(),
                            smoke_native_sha256=sha(native_path))
                full_path = ROOT / 'reports' / (name + '_full_plan.json')
                with full_path.open('x') as handle:
                    handle.write(json.dumps(plan, indent=2) + '\n')
                job = dict(name=name, plan=full_path.relative_to(ROOT).as_posix(),
                           plan_sha256=sha(full_path), gpu=entry['gpu'],
                           sequences=entry['sequences'], frames=entry['frames'], stage='full')
                processes[name], active[name] = launch(job)
        state = dict(pid=os.getpid(), registry_sha256=sha(REGISTRY),
                     active=active, completed=completed, failed=failed,
                     updated=time.time(), poll_seconds=180,
                     status='RUNNING' if active else 'COMPLETE_WITH_FAILURES' if failed else 'COMPLETE')
        temporary = STATUS.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(STATUS)
        if active:
            time.sleep(180)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run_queue()
