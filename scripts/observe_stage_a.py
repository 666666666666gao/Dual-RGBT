"""Read the existing run once and save a small progress/terminal snapshot."""
import argparse
import csv
import json
import time
from pathlib import Path

root = Path('/home/gaob/Dual-RGBT')
parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--launch', required=True)
parser.add_argument('--report', required=True)
args = parser.parse_args()
run = root / 'runs' / args.run
launch = json.loads((root / args.launch).read_text())
pid = launch['pid']
with (run / 'steps.csv').open() as handle:
    rows = list(csv.DictReader(handle))
report = {'observed_at_server_epoch': time.time(), 'pid': pid,
          'process_exists': Path('/proc/{}'.format(pid)).exists(),
          'optimizer_updates_logged': len(rows), 'last_step': rows[-1],
          'mean_seconds_last50': sum(float(r['seconds']) for r in rows[-50:]) / len(rows[-50:]),
          'completion': json.loads((run / 'COMPLETE.json').read_text()) if (run / 'COMPLETE.json').exists() else None,
          'dev': {}, 'utility': {}}
for key, prefix in [('dev', 'dev'), ('utility', 'utility')]:
    for path in sorted(run.glob('{}_epoch*.json'.format(prefix))):
        epoch = int(path.stem.split('epoch')[1])
        report[key][str(epoch)] = json.loads(path.read_text())
(root / args.report).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
