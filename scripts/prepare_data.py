import json
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
data = Path('/data2/wangwj/dataset/LasHeR-0428')
report = {'annotation': 'init.txt (same as CADTrack training loader)',
          'train': [], 'dev': [], 'issues': [], 'seed': 2027}

for split, source in [('train', 'train'), ('dev', 'val')]:
    names = (root / 'lib/train/data_specs/lasher_{}.txt'.format(source)).read_text().splitlines()
    for name in names:
        directory = data / 'trainingset' / name
        rgb = sorted((directory / 'visible').glob('*.jpg'))
        tir = sorted((directory / 'infrared').glob('*.jpg'))
        boxes = np.loadtxt(str(directory / 'init.txt'), delimiter=',', ndmin=2)
        if len(rgb) != len(tir) or len(rgb) != len(boxes) or not np.isfinite(boxes).all():
            report['issues'].append({'sequence': name, 'rgb': len(rgb), 'tir': len(tir),
                                     'annotations': len(boxes), 'split': split})
            continue
        valid = np.flatnonzero((boxes[:, 2] > 0) & (boxes[:, 3] > 0))
        assert len(valid) >= 3, name
        report[split].append({'name': name, 'rgb': list(map(str, rgb)), 'tir': list(map(str, tir)),
                              'boxes': boxes.tolist(), 'valid_frames': valid.tolist()})

assert not (set(s['name'] for s in report['train']) & set(s['name'] for s in report['dev']))
test_names = sorted(p.name for p in (data / 'testingset/testingset').iterdir() if p.is_dir())
report['test_sequences'] = test_names
assert not (set(test_names) & set(s['name'] for s in report['train'] + report['dev']))
(root / 'data/LasHeR').mkdir(parents=True, exist_ok=True)
for name, target in [('train', data / 'trainingset'), ('test', data / 'testingset/testingset')]:
    link = root / 'data/LasHeR' / name
    if not link.exists():
        link.symlink_to(target, target_is_directory=True)
(root / 'data/manifest.json').write_text(json.dumps(report), encoding='utf-8')
summary = {'train_sequences': len(report['train']), 'dev_sequences': len(report['dev']),
           'test_sequences': len(test_names), 'issues': report['issues'], 'annotation': report['annotation']}
(root / 'reports/data_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
assert not report['issues'], 'Resolve recorded dataset inconsistencies before training.'
