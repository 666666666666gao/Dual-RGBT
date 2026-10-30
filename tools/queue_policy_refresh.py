"""Admit each TRAIN-refresh shard by its real smoke, then fill free GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha


def main():
    path = ROOT / 'reports/POLICY_REFRESH_E6_QUEUE_20261030.json'
    assert not path.exists()
    registry = json.loads((ROOT / 'reports/POLICY_REFRESH_E6_ALL882_PLAN_20261030.json').read_text())
    queue = [dict(item, stage='smoke') for item in registry['jobs']]
    active, processes, completed, failed = {}, {}, [], []
    python = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
    cpu = str(ROOT / 'conda-envs/cpu-scoring/bin/python')
    while queue or active:
        for gpu, entry in list(active.items()):
            if processes[gpu].poll() is None:
                continue
            native_path = ROOT / 'runs/pipeline_tasks' / entry['task'] / 'status.json'
            native = json.loads(native_path.read_text())
            if native['state'] != 'COMPLETE' or native['exit_code'] != 0:
                failed.append(dict(entry=entry, native=native))
            else:
                report = 'reports/' + entry['run'] + '_audit.json'
                audit = subprocess.run([cpu, str(ROOT / 'tools/audit_policy_refresh.py'),
                                       '--run', entry['run'], '--task', entry['task'], '--output', report],
                                      cwd=str(ROOT))
                if audit.returncode != 0:
                    failed.append(dict(entry=entry, native=native, phase='AUDIT',
                                       audit_exit_code=audit.returncode))
                    del active[gpu], processes[gpu]
                    continue
                completed.append(dict(entry, audit=report, audit_sha256=sha(ROOT / report)))
                if entry['stage'] == 'smoke':
                    result_path = ROOT / 'runs' / entry['run'] / 'SMOKE.json'
                    result = json.loads(result_path.read_text())
                    assert result['observer_policy_trajectory_equal'] and result['frozen_state_tensors'] == 265
                    plan = json.loads((ROOT / entry['plan']).read_text())
                    plan.update(full_collection_allowed=True,
                                smoke_result=result_path.relative_to(ROOT).as_posix(),
                                smoke_result_sha256=sha(result_path),
                                smoke_native_status=native_path.relative_to(ROOT).as_posix(),
                                smoke_native_sha256=sha(native_path))
                    full_path = ROOT / 'reports' / (entry['name'] + '_full_plan.json')
                    with full_path.open('x') as handle:
                        handle.write(json.dumps(plan, indent=2) + '\n')
                    queue.insert(0, dict(name=entry['name'], plan=full_path.relative_to(ROOT).as_posix(),
                                         plan_sha256=sha(full_path), stage='full'))
            del active[gpu], processes[gpu]
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                         '--format=csv,noheader,nounits'], universal_newlines=True).splitlines()
        for index, used in enumerate(memory):
            gpu = str(index)
            if gpu in active or int(used) >= 500 or not queue:
                continue
            entry = queue.pop(0)
            assert sha(ROOT / entry['plan']) == entry['plan_sha256']
            run = entry['name'] + '_' + entry['stage']
            task = 'collect_' + run
            assert not (ROOT / 'runs' / run).exists()
            assert not (ROOT / 'runs/pipeline_tasks' / task).exists()
            command = [python, str(ROOT / 'scripts/run_task_four_gpu.py'), '--task', task,
                       '--gpu', gpu, '--', python, str(ROOT / 'scripts/collect_policy_query_features.py'),
                       '--plan', entry['plan'], '--run', run, '--task', task]
            if entry['stage'] == 'smoke':
                command.append('--smoke')
            with (ROOT / 'reports' / (run + '_wrapper.log')).open('xb') as handle:
                process = subprocess.Popen(command, cwd=str(ROOT),
                    env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'), stdout=handle,
                    stderr=subprocess.STDOUT, start_new_session=True)
            processes[gpu] = process
            active[gpu] = dict(entry, run=run, task=task, gpu=gpu, pid=process.pid,
                               command=command, started=time.time())
        result = dict(pid=os.getpid(), queue=queue, active=active, completed=completed,
                      failed=failed, updated=time.time(), poll_seconds=180,
                      status='RUNNING' if queue or active else 'COMPLETE_WITH_FAILURES' if failed else 'COMPLETE')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(path)
        if queue or active:
            time.sleep(180)


if __name__ == '__main__':
    main()
