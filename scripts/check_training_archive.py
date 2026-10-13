import json
from pathlib import Path
import torch

root = Path('/home/gaob/Dual-RGBT')
path = root / 'runs/stage_a_diagnostic_20261013/first_preserved.pth'
state = torch.load(path, map_location='cpu')
report = {'path': str(path), 'bytes': path.stat().st_size,
          'epoch': state['epoch'], 'optimizer_updates': state['step']}
(root / 'reports/preserved_archive.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
