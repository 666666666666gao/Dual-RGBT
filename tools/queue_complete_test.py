"""Run explicitly registered full-test jobs and score after wrapper exit."""
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
from joint_evaluation_contract import check_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text())
    check_files(ROOT, manifest['source_sha256'])
    output = ROOT / manifest['status_file']
    assert not output.exists()
    python = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
    cpu = str(ROOT / 'conda-envs/cpu-scoring/bin/python')
    pending = list(manifest['jobs'])
    active, completed, failed = {}, [], []
    processes = {}
    while pending or active:
        for gpu, entry in list(active.items()):
            if processes[gpu].poll() is None:
                continue
            native_path = ROOT / 'runs/pipeline_tasks' / entry['task'] / 'status.json'
            native = json.loads(native_path.read_text())
            if native['state'] == 'COMPLETE' and native['exit_code'] == 0:
                score_command = [cpu, str(ROOT / 'tools/score_cached_gain_test.py'),
                                 '--run', entry['run'], '--output', entry['score']]
                with (ROOT / 'reports' / (entry['run'] + '_score.log')).open('xb') as handle:
                    scored = subprocess.run(score_command, cwd=str(ROOT), stdout=handle,
                                            stderr=subprocess.STDOUT)
                if scored.returncode == 0:
                    result = json.loads((ROOT / entry['score']).read_text())
                    assert result['exploratory_test'] and not result['test_eligible']
                    assert (result['sequences'], result['frames']) == (245, 220703)
                    entry.update(metrics={k: result[k] for k in ('PR', 'NPR', 'SR', 'fps')},
                                 score_sha256=sha(ROOT / entry['score']))
                    completed.append(entry)
                else:
                    failed.append(dict(entry=entry, phase='SCORING', exit_code=scored.returncode))
            else:
                failed.append(dict(entry=entry, phase='INFERENCE', native=native))
            del active[gpu], processes[gpu]
        memory = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            universal_newlines=True).splitlines()
        for gpu in manifest['gpus']:
            if gpu in active or not pending or int(memory[int(gpu)]) >= 500:
                continue
            job = pending.pop(0)
            plan_path = ROOT / job['plan']
            assert sha(plan_path) == job['plan_sha256']
            plan = json.loads(plan_path.read_text())
            assert plan['exploratory_test'] and plan['full_evaluation_allowed']
            check_files(ROOT, plan['source_sha256'])
            run, task = job['run'], 'eval_' + job['run']
            assert not (ROOT / 'runs' / run).exists()
            assert not (ROOT / 'runs/pipeline_tasks' / task).exists()
            command = [python, str(ROOT / 'scripts/run_task_four_gpu.py'),
                       '--task', task, '--gpu', gpu, '--', python,
                       str(ROOT / job['script']), '--plan', job['plan'],
                       '--run', run, '--task', task, '--split', 'test',
                       '--smoke-frames', '0']
            with (ROOT / 'reports' / (run + '_wrapper.log')).open('xb') as handle:
                process = subprocess.Popen(
                    command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT,
                    env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'),
                    start_new_session=True)
            entry = dict(job, task=task, gpu=gpu, pid=process.pid,
                         command=command, started=time.time())
            processes[gpu], active[gpu] = process, entry
        state = dict(pid=os.getpid(), manifest_sha256=sha(manifest_path),
                     pending=pending, active=active, completed=completed, failed=failed,
                     poll_seconds=180, updated=time.time(),
                     status='RUNNING' if pending or active else
                     'COMPLETE_WITH_FAILURES' if failed else 'COMPLETE')
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(output)
        if pending or active:
            time.sleep(180)


if __name__ == '__main__':
    main()
