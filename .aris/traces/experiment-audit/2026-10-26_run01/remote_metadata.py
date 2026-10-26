import csv
import hashlib
import json
from pathlib import Path

root = Path('/data2/gb/Dual-RGBT')
run = root / 'runs/global_candidate_rankstop_trajectory_formal_20261018'
sources = ['tools/analyze_rankstop_training_decisions.py',
           'scripts/train_candidate_rankstop_trajectory.py',
           'scripts/train_candidate_evidence_policy.py',
           'dual_rgbt_global/candidate_tracker.py']

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

artifacts = {}
for name in ['query_labels.csv', 'trajectory_steps.csv', 'COMPLETE.json', 'config.json']:
    p = run / name
    artifacts[name] = dict(bytes=p.stat().st_size, sha256=sha(p))
    if name.endswith('.csv'):
        with p.open(newline='') as f:
            reader = csv.DictReader(f)
            artifacts[name]['fields'] = reader.fieldnames
            artifacts[name]['first_row'] = next(reader)
config = json.loads((run / 'config.json').read_text())
complete = json.loads((run / 'COMPLETE.json').read_text())
source_records = {}
for name in sources:
    p = root / name
    source_records[name] = dict(sha256=sha(p), bytes=p.stat().st_size,
                               recorded_training_sha256=config['source_sha256'].get(name))
print(json.dumps(dict(artifacts=artifacts, sources=source_records,
                     config=config, complete=complete), indent=2))
