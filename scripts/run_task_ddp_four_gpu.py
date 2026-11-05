"""Run one admitted DDP task across all four local GPUs."""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root / "runs/pipeline_tasks" / args.task
    directory.mkdir(parents=True, exist_ok=False)
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    env = dict(
        os.environ,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        OPENCV_FOR_THREADS_NUM="1",
        PYTHONUNBUFFERED="1",
    )
    started = time.time()
    record = dict(
        task=args.task,
        physical_gpus=["0", "1", "2", "3"],
        hostname=socket.gethostname(),
        command=command,
        started=started,
        pid=os.getpid(),
        state="RUNNING",
    )
    (directory / "status.json").write_text(json.dumps(record, indent=2) + "\n")
    with (directory / "run.log").open("wb") as log:
        result = subprocess.run(
            command, cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT
        )
    record.update(
        state="COMPLETE" if result.returncode == 0 else "FAILED",
        exit_code=result.returncode,
        completed=time.time(),
    )
    (directory / "status.json").write_text(json.dumps(record, indent=2) + "\n")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
