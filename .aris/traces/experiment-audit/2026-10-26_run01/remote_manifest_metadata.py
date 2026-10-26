import hashlib
import json
from pathlib import Path

root = Path('/data2/gb/Dual-RGBT')
report = {}
for name in ['data/manifest.json', 'inputs/server25/manifest.json']:
    p = root / name
    b = p.read_bytes()
    doc = json.loads(b)
    train = doc['train']
    row = train[0]
    report[name] = dict(sha256=hashlib.sha256(b).hexdigest(), keys=list(doc),
        train_count=len(train), first_sequence={key: value[:2] if isinstance(value, list) else value
        for key, value in row.items()}, sequence_keys=list(row))
for name in ['scripts/train_candidate_rankstop_trajectory.py',
             'scripts/train_candidate_evidence_policy.py',
             'dual_rgbt/data.py', 'dual_rgbt_evidence/regions.py']:
    b = (root / name).read_bytes()
    report[name] = dict(sha256=hashlib.sha256(b).hexdigest(),
        lf_sha256=hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest(),
        crlf_count=b.count(b'\r\n'), lf_count=b.count(b'\n'))
print(json.dumps(report, indent=2))
