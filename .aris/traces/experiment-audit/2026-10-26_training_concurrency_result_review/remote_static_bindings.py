import hashlib
import json
from pathlib import Path

root = Path('/data2/gb/Dual-RGBT')
plan_path = root / 'reports/TRAIN_CONCURRENCY_PLAN_20261026.json'
plan = json.loads(plan_path.read_bytes())

def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()

source = {k:sha(root/k) for k in plan['source_sha256']}
inputs = {k:sha(root/k) for k in plan['input_sha256']}
out = root / plan['output']
files = list(out.rglob('*'))
files += [plan_path, root/plan['source_review'], root/'runs/pipeline_tasks'/plan['task']/'status.json', root/'runs/pipeline_tasks'/plan['task']/'run.log']
artifacts = {str(p.relative_to(root)):sha(p) for p in sorted(files) if p.is_file()}
manifest=json.loads((root/'data/manifest.json').read_bytes())
by_name={row['name']:row for row in manifest['train']}
base=json.loads((root/plan['training_plan']).read_bytes())
selected=[]
for name in plan['sequence_names']:
    row=by_name[name]
    eligible=[f for f in row['valid_frames'] if f>0 and f%base['label_stride']==0]
    selected.append(dict(name=name,total_frames=len(row['boxes']),initial_box=row['boxes'][0],
                         eligible_update_frames=eligible[:plan['updates_per_stream']],
                         recorded_targets=[row['boxes'][f] for f in eligible[:plan['updates_per_stream']]],
                         first_rgb=row['rgb'][0],first_tir=row['tir'][0]))
print(json.dumps(dict(scope='Only frozen source/input and finished-profile files; no active status or process/resource query.',
    source_sha256=source,input_sha256=inputs,artifacts_sha256=artifacts,selected_train=selected),indent=2))
