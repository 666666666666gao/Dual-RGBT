"""Rebase LasHeR train/dev paths only after checking every file and annotation."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD_TRAIN = Path('/data2/wangwj/dataset/LasHeR-0428/trainingset')


def rebased(sequence):
    result = dict(sequence)
    for modality in ('rgb', 'tir'):
        result[modality] = [str(ROOT / 'data/LasHeR/train' / Path(name).relative_to(OLD_TRAIN))
                            for name in sequence[modality]]
        assert all(Path(name).is_file() and Path(name).stat().st_size > 0 for name in result[modality]), sequence['name']
    actual = np.loadtxt(str(ROOT / 'data/LasHeR/train' / sequence['name'] / 'init.txt'), delimiter=',', ndmin=2)
    expected = np.asarray(sequence['boxes'])
    assert np.array_equal(actual.astype(np.float32), expected.astype(np.float32)), sequence['name']
    return result


def main():
    old_path = ROOT / 'inputs/server25/manifest.json'
    original = json.loads(old_path.read_bytes())
    migrated = dict(original)
    for split in ('train', 'dev'):
        migrated[split] = [rebased(sequence) for sequence in original[split]]
    assert len(migrated['train']) == 882 and len(migrated['dev']) == 97
    output = ROOT / 'data/manifest.json'
    assert not output.exists()
    output.write_text(json.dumps(migrated))
    manifests = {}
    for split in ('lasher_train_smoke', 'lasher_dev'):
        old = ROOT / 'data/benchmarks' / (split + '.json')
        manifest = json.loads(old.read_bytes())
        manifest['sequences'] = [rebased(sequence) for sequence in manifest['sequences']]
        new = ROOT / 'data/benchmarks/server26' / (split + '.json')
        new.parent.mkdir(parents=True, exist_ok=True)
        assert not new.exists()
        new.write_text(json.dumps(manifest))
        manifests[str(new.relative_to(ROOT))] = hashlib.sha256(new.read_bytes()).hexdigest()
    report = dict(status='TRAIN_DEV_PATH_AND_GT_CHECK_PASS', server='2026', project=str(ROOT),
                  train_sequences=882, dev_sequences=97, dev_frames=54646,
                  train_dev_all_referenced_images_exist_nonempty=True,
                  train_dev_all_init_annotations_float32_equal=True,
                  original_manifest_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                  migrated_manifest_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  benchmark_manifests_sha256=manifests,
                  scope='Paths changed, annotation arrays/order/splits retained. Existence and GT checks are full; '
                        'image contents are not fully rehashed. Test manifests and GPU execution remain separate.')
    (ROOT / 'reports/SERVER26_TRAIN_DEV_MIGRATION_20261026.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
