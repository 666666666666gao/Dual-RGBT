"""Read the existing run once and save a small progress/terminal snapshot."""
import csv
import json
import os
import time
from pathlib import Path

root = Path('/home/gaob/Dual-RGBT')
run = root / 'runs/stage_a_diagnostic_20261013'
launch = json.loads((root / 'reports/launch_stage_a.json').read_text())
pid = launch['pid']
with (run / 'steps.csv').open() as handle:
    rows = list(csv.DictReader(handle))
report = {'observed_at_server_epoch': time.time(), 'pid': pid,
          'process_exists': Path('/proc/{}'.format(pid)).exists(),
          'optimizer_updates_logged': len(rows), 'last_step': rows[-1],
          'mean_seconds_last50': sum(float(r['seconds']) for r in rows[-50:]) / len(rows[-50:]),
          'completion': json.loads((run / 'COMPLETE.json').read_text()) if (run / 'COMPLETE.json').exists() else None,
          'dev': {}, 'utility': {}}
for epoch in (1, 2):
    for key, prefix in [('dev', 'dev'), ('utility', 'utility')]:
        path = run / '{}_epoch{:02d}.json'.format(prefix, epoch)
        if path.exists():
            report[key][str(epoch)] = json.loads(path.read_text())
(root / 'reports/stage_a_latest.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
