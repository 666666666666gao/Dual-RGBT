import json
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
checkpoint = torch.load('/home/gaob/CADTrack_ep0014.pth.tar', map_location='cpu')
state = checkpoint['net']
report = {
    'checkpoint': '/home/gaob/CADTrack_ep0014.pth.tar',
    'metadata_types': {k: type(v).__name__ for k, v in checkpoint.items()},
    'epoch': checkpoint['epoch'],
    'state_shapes': {k: list(v.shape) for k, v in state.items()},
    'state_elements': sum(v.numel() for v in state.values()),
    'torch': torch.__version__,
    'source': '/data2/zhouy/CADTrack-main',
}
(root / 'reports').mkdir(exist_ok=True)
(root / 'reports' / 'checkpoint_structure.json').write_text(
    json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'state_shapes'}, indent=2))
print('first keys:', list(state)[:12])
