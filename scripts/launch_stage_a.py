"""Launch the one reviewed diagnostic run; records the actual child PID."""
import json
import subprocess
from datetime import datetime
from pathlib import Path

root = Path('/home/gaob/Dual-RGBT')
assert json.loads((root / 'reports/sanity.json').read_text())['status'] == 'PASS'
assert not (root / 'runs/stage_a_diagnostic_20261013').exists()
(root / 'runs').mkdir(exist_ok=True)
log = root / 'runs/stage_a_diagnostic_20261013.log'
with log.open('w') as handle:
    child = subprocess.Popen(['/bin/bash', str(root / 'scripts/run_stage_a.sh')],
                             cwd=str(root), stdout=handle, stderr=subprocess.STDOUT,
                             start_new_session=True)
record = {'pid': child.pid, 'launched_at_server': datetime.now().isoformat(),
          'physical_gpu': 0, 'command': 'bash scripts/run_stage_a.sh',
          'log': str(log), 'source_commit': (root / 'CODE_REVISION').read_text().strip(),
          'scope': '2 epochs / 2000 samples per epoch / 1000 AdamW updates, diagnostic only'}
(root / 'reports/launch_stage_a.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
