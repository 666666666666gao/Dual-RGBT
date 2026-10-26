"""Run an admitted owned task on one of four GPUs and persist its actual exit status."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--gpu', choices=('0', '1', '2', '3'), required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root / 'runs/pipeline_tasks' / args.task
    directory.mkdir(parents=True, exist_ok=False)
    command = args.command[1:] if args.command[0] == '--' else args.command
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpu,
               PYTHONUNBUFFERED='1')
    started = time.time()
    record = dict(task=args.task, physical_gpu=args.gpu, command=command,
                  started=started, pid=os.getpid(), state='RUNNING')
    (directory / 'status.json').write_text(json.dumps(record, indent=2))
    with (directory / 'run.log').open('wb') as log:
        result = subprocess.run(command, cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(state='COMPLETE' if result.returncode == 0 else 'FAILED',
                  exit_code=result.returncode, completed=time.time())
    (directory / 'status.json').write_text(json.dumps(record, indent=2))
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
