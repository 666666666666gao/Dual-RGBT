import hashlib
import json
from pathlib import Path
import subprocess

trace = Path('.aris/traces/experiment-audit/2026-10-26_run01')
meta = json.loads((trace / 'remote_metadata.json').read_text())
names = ['scripts/train_candidate_rankstop_trajectory.py', 'scripts/train_candidate_evidence_policy.py']
commits = subprocess.check_output(['git', 'log', '--all', '--format=%H', '--'] + names).decode().splitlines()
records = []
for name in names:
    expected = meta['config']['source_sha256'][name]
    for commit in commits:
        obj = subprocess.run(['git', 'show', commit + ':' + name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if obj.returncode:
            records.append(dict(path=name, commit=commit, exists=False))
            continue
        b = obj.stdout
        hashes = {'raw': hashlib.sha256(b).hexdigest(),
                  'lf': hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest(),
                  'crlf': hashlib.sha256(b.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()}
        matching = [key for key, value in hashes.items() if value == expected]
        records.append(dict(path=name, commit=commit, exists=True, hashes=hashes, expected=expected, matching=matching))
        if matching:
            exact = b if matching[0] == 'raw' else b.replace(b'\r\n', b'\n')
            if matching[0] == 'crlf':
                exact = exact.replace(b'\n', b'\r\n')
            (trace / ('historical_' + Path(name).name)).write_bytes(exact)
result = dict(records=records)
(trace / 'source_history.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
