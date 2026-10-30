"""Refill each free GPU with an admitted retained-epoch evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', nargs='+', type=int, required=True)
    parser.add_argument('--smoke-attempt', default='')
    args = parser.parse_args()
    path = ROOT / ('reports/CACHED_PARENT_LOSS_QUEUE_SERVER25_20261030' + args.smoke_attempt + '.json')
    assert not path.exists()
    queue = [dict(epoch=e, phase='eval-smoke') for e in args.epochs]
    active, completed, failed = {}, [], []
    while queue or active:
        for gpu, entry in list(active.items()):
            native_path = ROOT / 'runs/pipeline_tasks' / entry['task'] / 'status.json'
            if not native_path.exists():
                continue
            native = json.loads(native_path.read_text())
            if native['state'] == 'RUNNING':
                continue
            assert native['state'] in ('COMPLETE', 'FAILED')
            if native['state'] == 'COMPLETE' and native['exit_code'] == 0:
                completed.append(entry)
                if entry['phase'] == 'eval-smoke':
                    queue.insert(0, dict(epoch=entry['epoch'], phase='dev'))
            else:
                failed.append(dict(entry=entry, native=native))
            del active[gpu]
        memory = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            universal_newlines=True).splitlines()
        for index, used in enumerate(memory):
            gpu = str(index)
            if gpu in active or int(used) >= 500 or not queue:
                continue
            job = queue.pop(0)
            command = [sys.executable, str(ROOT / 'tools/launch_cached_parent_loss.py'),
                       '--phase', job['phase'], '--epoch', str(job['epoch']), '--gpu', gpu,
                       '--smoke-attempt', args.smoke_attempt]
            entry = json.loads(subprocess.check_output(command, cwd=str(ROOT), universal_newlines=True))
            active[gpu] = dict(entry, **job)
        state = dict(pid=os.getpid(), queue=queue, active=active, completed=completed,
                     failed=failed, poll_seconds=180, updated=time.time(),
                     status='RUNNING' if queue or active else 'COMPLETE_WITH_FAILURES' if failed else 'COMPLETE')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(path)
        if queue or active:
            time.sleep(180)


if __name__ == '__main__':
    main()
