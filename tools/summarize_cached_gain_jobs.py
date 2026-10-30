"""Save actual task receipts and compact GPU/progress evidence for handoff."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', choices=('25', '26'), required=True)
    args = parser.parse_args()
    records = {}
    for directory in (ROOT / 'runs').glob('cached_query_gain_parent_loss*'):
        for name in ('config.json', 'COMPLETE.json', 'SMOKE.json'):
            path = directory / name
            if path.exists():
                records[path.relative_to(ROOT).as_posix()] = dict(sha256=sha(path), value=json.loads(path.read_text()))
    for path in (ROOT / 'runs/pipeline_tasks').glob('*cached_query_gain_parent_loss*/status.json'):
        records[path.relative_to(ROOT).as_posix()] = dict(sha256=sha(path), value=json.loads(path.read_text()))
    for pattern in ('cached_query_gain_parent_loss*_plan.json', 'cached_query_gain_parent_loss*_launch.json',
                    'CACHED_PARENT_LOSS_QUEUE_SERVER*.json', 'CACHED_PARENT_LOSS_SERVER26_MAPPING_*.json',
                    'CACHED_PARENT_LOSS_SERVER26_SNAPSHOT_INVENTORY_*.json'):
        for path in (ROOT / 'reports').glob(pattern):
            records[path.relative_to(ROOT).as_posix()] = dict(sha256=sha(path), value=json.loads(path.read_text()))
    active = []
    for name, record in records.items():
        native = record['value']
        if not name.endswith('/status.json') or native['state'] != 'RUNNING':
            continue
        command = native['command']
        run = command[command.index('--run') + 1]
        path = ROOT / 'runs' / run / 'progress.json'
        progress = json.loads(path.read_text()) if path.exists() else []
        active.append(dict(task=native['task'], gpu=native['physical_gpu'], run=run,
                           completed_sequences=len(progress), completed_frames=sum(x['frames'] for x in progress),
                           wrapper_pid=native['pid'], wrapper_alive=(Path('/proc') / str(native['pid'])).exists()))
    smoke_rows = {}
    for directory in (ROOT / 'runs').glob('cached_query_gain_parent_loss*_eval_smoke*'):
        if not (directory / 'SMOKE.json').exists():
            continue
        predictions = next((directory / 'predictions').glob('*.txt'))
        trace = next((directory / 'traces').glob('*.csv'))
        with trace.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        integer_fields = ('frame', 'modality', 'action', 'primary_tokens', 'auxiliary_tokens',
                          'request_index', 'candidate_rank', 'candidate_count')
        smoke_rows[directory.name] = dict(predictions=predictions.read_text(),
            routes=[{key: row[key] for key in integer_fields} for row in rows])
    scored = {}
    if args.server == '25':
        for epoch in range(6, 11):
            path = ROOT / 'reports/cached_query_gain_dev_20261030' / ('e%d.json' % epoch)
            report = json.loads(path.read_text())
            run = ROOT / 'runs' / ('cached_query_gain_utility_focal_e%d_lasher_dev_20261030' % epoch)
            assert all(sha(run / name) == digest for name, digest in report['run_artifacts_sha256'].items())
            native = ROOT / 'runs/pipeline_tasks' / ('eval_' + run.name) / 'status.json'
            assert sha(native) == report['native_status_sha256']
            assert sha(path.with_suffix('.npz')) == report['curves_sha256']
            scored[str(epoch)] = {key: report[key] for key in ('status', 'PR', 'NPR', 'SR', 'fps')}
    result = dict(server=args.server, root=str(ROOT), observed_epoch=time.time(), records=records,
                  active=active, smoke_rows=smoke_rows, verified_focal_dev=scored,
                  gpu_csv=subprocess.check_output(
                      ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                       '--format=csv,noheader'], universal_newlines=True))
    output = ROOT / ('reports/CACHED_PARENT_LOSS_SERVER%s_RECEIPTS_20261030.json' % args.server)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), active=active, gpu_csv=result['gpu_csv'], verified_focal_dev=scored), indent=2))


if __name__ == '__main__':
    main()
