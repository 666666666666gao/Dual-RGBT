"""Run the eight preregistered diagnostics on server25's three free GPUs."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'reports/FIXED_STATE_DEV_QUEUE_20261030.json'
    assert not output.exists()
    jobs = [(model, sequence) for sequence in
            ('midboyblue', 'belowrightwhiteboy', 'moto2north1', 'browncar2north')
            for model in ('parent', 'e6')]
    active, complete = {}, []
    while jobs or active:
        for gpu, entry in list(active.items()):
            native_path = ROOT / 'runs/pipeline_tasks' / entry['task'] / 'status.json'
            native = json.loads(native_path.read_text())
            if native['state'] != 'RUNNING':
                assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
                result = ROOT / 'runs' / entry['run'] / 'COMPLETE.json'
                assert json.loads(result.read_text())['status'] == 'DIAGNOSTIC_COMPLETE'
                complete.append(entry)
                del active[gpu]
        for gpu in ('0', '2', '3'):
            if gpu in active or not jobs:
                continue
            model, sequence = jobs.pop(0)
            command = [sys.executable, str(ROOT / 'tools/launch_fixed_state_dev.py'),
                       '--model', model, '--sequence', sequence, '--gpu', gpu]
            entry = json.loads(subprocess.check_output(command, cwd=str(ROOT)).decode())
            active[gpu] = entry
        output.write_text(json.dumps(dict(pid=__import__('os').getpid(), active=active,
                                          queue=jobs, complete=complete,
                                          updated=time.time(), poll_seconds=180), indent=2))
        if jobs or active:
            time.sleep(180)


if __name__ == '__main__':
    main()
