"""Read archived bindings and installed TRAIN initialization only; stdlib only."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data2/gb/Dual-RGBT')
PLAN = ROOT / 'reports/RANKSTOP_INFERENCE_SCALING_PLAN_20261026.json'
plan = json.loads(PLAN.read_bytes())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


inputs = {name: dict(sha256=sha(ROOT / name), expected=value,
                    size=(ROOT / name).stat().st_size)
          for name, value in plan['input_sha256'].items()}
root = ROOT / 'profile_output' / plan['run']
paths = [PLAN, ROOT / plan['source_review'], root / 'config.json', root / 'COMPLETE.json',
         ROOT / 'runs/pipeline_tasks' / plan['task'] / 'status.json',
         ROOT / 'runs/pipeline_tasks' / plan['task'] / 'run.log']
for directory in sorted(root.iterdir()):
    if directory.is_dir():
        paths.extend(sorted(directory.glob('*.json')))
        paths.extend(sorted(directory.glob('*.prof')))
artifacts = {str(path.relative_to(ROOT)): sha(path) for path in paths}
manifest = json.loads((ROOT / plan['manifest']).read_bytes())
by_name = {row['name']: row for row in manifest['train']}
dev = json.loads((ROOT / 'data/benchmarks/server26/lasher_dev.json').read_bytes())
dev_names = [row['name'] for row in dev['sequences']]
rows = []
for name in plan['sequence_names']:
    row = by_name[name]
    init = Path(row['rgb'][0]).parent.parent / 'init.txt'
    first = [float(value) for value in init.read_text().splitlines()[0].replace(',', ' ').split()]
    frames = plan['frames_per_sequence']
    rows.append(dict(name=name, rgb_frames=len(row['rgb']), tir_frames=len(row['tir']),
                     boxes_frames=len(row['boxes']), manifest_initial_box=row['boxes'][0],
                     init_path=str(init), init_sha256=sha(init), installed_initial_box=first,
                     image_paths_present=all(Path(p).is_file() for key in ('rgb', 'tir') for p in row[key][:frames]),
                     first_rgb=str(row['rgb'][0]), first_tir=str(row['tir'][0]),
                     first_rgb_sha256=sha(row['rgb'][0]), first_tir_sha256=sha(row['tir'][0]),
                     rgb_prefix=row['rgb'][:frames], tir_prefix=row['tir'][:frames]))
print(json.dumps(dict(scope='Archived profile/input hashes and TRAIN initial-box provenance only; no active status, process, resource or NN calls.',
                      plan_sha256=sha(PLAN), inputs=inputs, artifacts=artifacts,
                      train_sequences=len(manifest['train']), dev_sequences=len(dev_names),
                      selected_dev_overlap=sorted(set(plan['sequence_names']) & set(dev_names)),
                      selected_train=rows), indent=2))
