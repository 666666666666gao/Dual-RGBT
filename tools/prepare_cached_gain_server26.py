"""Verify the sealed snapshot and map the existing dev image copy."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = Path('/data2/gb/Dual-RGBT')
ORIGINAL = '/data2/wangwj/dataset/LasHeR-0428/trainingset/'
LOCAL = str(PROVIDER / 'data/LasHeR-0428/trainingset') + '/'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert ROOT == PROVIDER / 'snapshots/cached_parent_loss_20261030'
    inventory_path = ROOT / 'reports/CACHED_PARENT_LOSS_SERVER26_SNAPSHOT_INVENTORY_20261030.json'
    inventory = json.loads(inventory_path.read_text())
    for name, item in inventory['files'].items():
        path = ROOT / name
        if not path.exists():
            original = PROVIDER / name
            assert original.is_file() and sha(original) == item['sha256'], name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(original)
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
    (ROOT / 'conda-envs').symlink_to(PROVIDER / 'conda-envs', target_is_directory=True)
    sys.path.insert(0, str(ROOT / 'scripts'))
    from joint_evaluation_contract import evaluation_plan
    parent_path = 'reports/joint_candidate_evaluation/joint_quality_k8_cost0p1_server25_20261027_formal/plan.json'
    evaluation_plan(ROOT, parent_path)
    manifest_path = ROOT / 'data/benchmarks/lasher_dev.json'
    manifest = json.loads(manifest_path.read_text())
    for row in manifest['sequences']:
        for key in ('rgb', 'tir'):
            for value in row[key]:
                assert value.startswith(ORIGINAL), value
                assert Path(LOCAL + value[len(ORIGINAL):]).is_file(), value
    verified = {}
    for name, digest in inventory['origin_image_sha256'].items():
        local = Path(LOCAL + name[len(ORIGINAL):])
        assert sha(local) == digest, name
        verified[str(local)] = digest
    result = dict(status='MAPPING_VERIFIED', original_prefix=ORIGINAL, local_prefix=LOCAL,
                  sequences=manifest['sequence_count'], frames=manifest['frame_count'],
                  checked_image_paths=2 * manifest['frame_count'],
                  manifest_sha256=sha(manifest_path), sampled_image_sha256=verified,
                  snapshot_inventory_sha256=sha(inventory_path),
                  scope='All image paths exist; both modalities first/last frame per sequence match source25 bytes. Not an all-image hash audit.',
                  historical_training_provider='2025', runtime_provider='2026')
    with (ROOT / 'reports/CACHED_PARENT_LOSS_SERVER26_MAPPING_20261030.json').open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in
                      ('status', 'sequences', 'frames', 'checked_image_paths', 'snapshot_inventory_sha256')}))


if __name__ == '__main__':
    main()
