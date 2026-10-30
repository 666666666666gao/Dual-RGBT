"""Save measured queue, native-task and GPU state for this continuation."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', choices=('25', '26'), required=True)
    args = parser.parse_args()
    refresh = json.loads((ROOT / 'reports/POLICY_REFRESH_E6_QUEUE_20261030.json').read_text())
    tasks = {}
    for entry in list(refresh['active'].values()) + refresh['completed']:
        native = ROOT / 'runs/pipeline_tasks' / entry['task'] / 'status.json'
        directory = ROOT / 'runs' / entry['run']
        progress = directory / 'progress.json'
        rows = json.loads(progress.read_text()) if progress.exists() else []
        tasks[entry['task']] = dict(native=json.loads(native.read_text()),
                                   completed_sequences=len(rows),
                                   completed_frames=sum(row['frames'] for row in rows))
    sizes = {}
    if args.server == '25':
        for label in ('b576', 'b640'):
            name = 'size_tail_' + label + '_formal_epochs_20261030'
            state = json.loads((ROOT / 'runs' / name / 'status.json').read_text())
            native = ROOT / 'runs/pipeline_tasks' / state['active_task'] / 'status.json'
            sizes[label] = dict(controller=state, native=json.loads(native.read_text()))
        retention = json.loads((ROOT / 'runs/size_tail_b576_formal_epochs_20261030/retained_last_five/status.json').read_text())
    else:
        retention = None
    result = dict(server=args.server, root=str(ROOT), observed_epoch=time.time(),
                  refresh=refresh, tasks=tasks, sizes=sizes, legacy_retention=retention,
                  gpu_csv=subprocess.check_output(
                      ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                       '--format=csv,noheader'], universal_newlines=True),
                  host_memory_csv=subprocess.check_output(['free', '-m'], universal_newlines=True))
    output = ROOT / ('reports/CONTINUATION_SERVER%s_STATUS_20261030.json' % args.server)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(server=args.server, output=str(output), refresh_status=refresh['status'],
                          active=len(refresh['active']), completed=len(refresh['completed']),
                          failed=len(refresh['failed']), gpu_csv=result['gpu_csv']), indent=2))


if __name__ == '__main__':
    main()
