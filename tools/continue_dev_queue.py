"""Continue an owned dev queue, adopting live wrappers without restarting them."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True)
    parser.add_argument('--launch-memory-limit-mib', type=int, default=500)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / args.state
    lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(path.read_text())
    assert state['status'] == 'RUNNING'
    tasks = [e['task'] for e in state['queue']] + [a['entry']['task'] for a in state['active'].values()]
    assert len(tasks) == len(set(tasks))
    python = str(root / 'conda-envs/dual_rgbt/bin/python')
    processes = {}

    def save():
        state.update(controller_pid=os.getpid(), poll_seconds=180, updated=time.time())
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(path)

    save()
    while state['queue'] or state['active']:
        for gpu, item in list(state['active'].items()):
            if gpu in processes:
                if processes[gpu].poll() is None:
                    continue
                del processes[gpu]
            else:
                proc = Path('/proc') / str(item['pid'])
                if proc.exists() and proc.joinpath('stat').read_text().split(') ')[1][0] != 'Z':
                    continue
            native = root / 'runs/pipeline_tasks' / item['entry']['task'] / 'status.json'
            status = json.loads(native.read_text())
            assert status['state'] in ('COMPLETE', 'FAILED'), status
            bucket = 'completed' if status['state'] == 'COMPLETE' and status['exit_code'] == 0 else 'failed'
            state[bucket].append(dict(entry=item['entry'], gpu=gpu, status=status))
            del state['active'][gpu]
            save()
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], universal_newlines=True)
        for gpu, used in enumerate(memory.splitlines()):
            gpu = str(gpu)
            if gpu in state['active'] or not state['queue'] or int(used) >= args.launch_memory_limit_mib:
                continue
            entry = state['queue'][0]
            assert not (root / 'runs/pipeline_tasks' / entry['task']).exists()
            assert not (root / 'runs' / entry['run']).exists()
            cmd = [python, str(root / 'scripts/run_task_four_gpu.py'), '--task', entry['task'], '--gpu', gpu, '--',
                   python, str(root / 'scripts/run_joint_candidate_evaluation.py'), '--plan', entry['plan'],
                   '--benchmark', 'LasHeR', '--split', 'dev', '--task', entry['task'], '--run', entry['run'],
                   '--cost-weight', str(entry['cost_weight']), '--shard', '0', '--shards', '1', '--smoke-frames', '0']
            with (path.parent / (entry['task'] + '.wrapper.log')).open('ab') as output:
                proc = subprocess.Popen(cmd, cwd=str(root), env=dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1', OPENCV_FOR_THREADS_NUM='1'), stdout=output, stderr=subprocess.STDOUT)
            processes[gpu] = proc
            state['active'][gpu] = dict(entry=entry, pid=proc.pid, launched=time.time())
            state['queue'].pop(0)
            save()
        if state['queue'] or state['active']:
            state['next_check_epoch'] = time.time() + 180
            save()
            time.sleep(180)
    state['status'] = 'COMPLETE' if not state['failed'] else 'COMPLETE_WITH_FAILURES'
    save()
    lock.close()


if __name__ == '__main__':
    main()
