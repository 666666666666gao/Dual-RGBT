"""Export immutable server25 provenance and code for server26 evaluation."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = set()

    def add(name):
        assert (ROOT / name).is_file(), name
        files.add(name)

    def mapping(value):
        for name, digest in value.items():
            add(name)
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

    parent_path = 'reports/joint_candidate_evaluation/joint_quality_k8_cost0p1_server25_20261027_formal/plan.json'
    parent = json.loads((ROOT / parent_path).read_text())
    add(parent_path)
    mapping(parent['source_sha256'])
    mapping(parent['input_sha256'])
    for name in (parent['protocol'], parent['training_plan']):
        add(name)
    spec = json.loads((ROOT / parent['protocol']).read_text())
    add(spec['source_review'])
    for name in ('reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json',
                 'reports/CACHED_QUERY_GAIN_EVAL_SOURCE_REVIEW_20261030.json',
                 'reports/CACHED_QUERY_GAIN_UTILITY_FOCAL_E10_EVAL_SMOKE_PLAN_20261030.json'):
        add(name)
    formal = 'cached_query_gain_parent_loss_formal_20261030'
    for name in ('config.json', 'COMPLETE.json', 'progress.json'):
        add('runs/' + formal + '/' + name)
    add('runs/pipeline_tasks/train_' + formal + '/status.json')
    add('reports/' + formal + '_plan.json')
    config = json.loads((ROOT / 'runs' / formal / 'config.json').read_text())
    mapping(config['input_sha256'])
    mapping(config['plan']['source_sha256'])
    result = json.loads((ROOT / 'runs' / formal / 'COMPLETE.json').read_text())
    for item in result['retained']:
        add(item['checkpoint'])
    template = json.loads((ROOT / 'reports/CACHED_QUERY_GAIN_UTILITY_FOCAL_E10_EVAL_SMOKE_PLAN_20261030.json').read_text())
    mapping(template['source_sha256'])
    add('runs/' + template['reference_run'] + '/config.json')
    for directory in ('scripts', 'dual_rgbt', 'dual_rgbt_global', 'dual_rgbt_evidence', 'lib', 'tools'):
        for path in (ROOT / directory).rglob('*.py'):
            add(path.relative_to(ROOT).as_posix())
    for path in (ROOT / 'experiments').rglob('*.yaml'):
        add(path.relative_to(ROOT).as_posix())
    for path in (ROOT / 'vendor/cadtrack/toolkit').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            add(path.relative_to(ROOT).as_posix())
    manifest = json.loads((ROOT / 'data/benchmarks/lasher_dev.json').read_text())
    images = {row[key][index] for row in manifest['sequences']
              for key in ('rgb', 'tir') for index in (0, -1)}
    image_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sorted(images)}
    inventory = {name: dict(sha256=hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                            bytes=(ROOT / name).stat().st_size) for name in sorted(files)}
    small = [name for name in files if inventory[name]['bytes'] < 100000000]
    report = ROOT / 'reports/CACHED_PARENT_LOSS_SERVER26_SNAPSHOT_INVENTORY_20261030.json'
    with report.open('x') as handle:
        handle.write(json.dumps(dict(origin=str(ROOT), files=inventory, origin_image_sha256=image_hashes,
                                    historical_training_records_preserved=True), indent=2) + '\n')
    archive = ROOT / 'reports/CACHED_PARENT_LOSS_SERVER26_SNAPSHOT_20261030.tar.gz'
    with tarfile.open(str(archive), 'x:gz') as handle:
        for name in sorted(small):
            handle.add(str(ROOT / name), arcname=name)
        handle.add(str(report), arcname=report.relative_to(ROOT).as_posix())
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                          external_files={name: item for name, item in inventory.items()
                                          if name not in small}), indent=2))


if __name__ == '__main__':
    main()
