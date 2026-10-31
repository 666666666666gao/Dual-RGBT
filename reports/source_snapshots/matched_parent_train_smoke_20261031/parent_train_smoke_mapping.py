"""Hash the two provider copies of exactly the declared eleven TRAIN frames."""
import hashlib
import json
from pathlib import Path
import sys

root, provider = Path(sys.argv[1]), sys.argv[2]
manifest_path = root / 'data/benchmarks/lasher_train_smoke.json'
manifest = json.loads(manifest_path.read_text())
assert manifest['split'] == 'train_smoke'
sequence = manifest['sequences'][0]
old = '/data2/wangwj/dataset/LasHeR-0428/trainingset/'
local = '/home/gaob/Dual-RGBT/data/LasHeR/train/' if provider == '2025' else '/data2/gb/Dual-RGBT/data/LasHeR/train/'
files = {}
for key in ('rgb', 'tir'):
    for name in sequence[key][:11]:
        assert name.startswith(old)
        relative = name[len(old):]
        path = Path(local + relative)
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
assert len(files) == 22
record = dict(provider=provider, sequence=sequence['name'], frames=11, images=22,
    original_prefix=old, local_prefix=local, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    files_sha256=files)
if provider == '2026':
    source_path = root / 'reports/MATCHED_PARENT_TRAIN_SMOKE_IMAGES_P25_20261031.json'
    source = json.loads(source_path.read_text())
    assert source['sequence'] == record['sequence'] and source['files_sha256'] == files
    assert (source['frames'], source['images']) == (11, 22)
    record.update(status='TRAIN_SMOKE_MAPPING_VERIFIED', sequences=1,
        source_receipt=source_path.relative_to(root).as_posix(),
        source_receipt_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
print(json.dumps(record, indent=2))
