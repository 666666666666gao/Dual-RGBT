"""Bind the existing full-evaluation recipe to24 server26 and8 server25 finals."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_bytes())


def sources_for(training_paths):
    sources = {}
    for name in training_paths:
        for path, digest in read(name)['source_sha256'].items():
            assert path not in sources or sources[path] == digest
            sources[path] = digest
    paths = ['scripts/joint_evaluation_contract.py', 'scripts/run_joint_candidate_evaluation.py',
             'tools/joint_candidate_results.py', 'tools/register_joint_evaluation_protocol.py',
             'tools/register_backend_evaluation_protocols.py',
             'experiments/cadtrack/cadtrack.yaml', 'lib/config/cadtrack/config.py',
             'lib/utils/box_ops.py', 'lib/utils/ce_utils.py',
             'dual_rgbt/__init__.py', 'lib/__init__.py', 'lib/test/utils/__init__.py',
             'lib/test/utils/params.py', 'lib/test/utils/hann.py',
             'lib/utils/__init__.py', 'lib/utils/tensor.py']
    # Model dependencies in the actual evaluator import graph; retired modules are unused.
    paths += ['lib/models/__init__.py', 'lib/models/cadtrack/__init__.py',
              'lib/models/cadtrack/base_backbone.py', 'lib/models/cadtrack/cadtrack.py',
              'lib/models/cadtrack/utils.py', 'lib/models/cadtrack/vit_care.py',
              'lib/models/layers/__init__.py', 'lib/models/layers/cross_layer.py',
              'lib/models/layers/frozen_bn.py', 'lib/models/layers/head.py',
              'lib/models/layers/patch_embed.py']
    for path in paths:
        sources[path] = sha(ROOT / path)
    assert all(sha(ROOT / n) == h for n, h in sources.items())
    return sources


def main():
    # No model-specific plan may exist while its shared protocol is being resealed.
    assert not (ROOT / 'reports/joint_candidate_evaluation').exists()
    archive = ROOT / '.aris/traces/experiment-audit/2026-10-27_backend_evaluation_binding/pre_change/reports'
    original8 = json.loads((archive / 'JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json').read_bytes())
    expanded = json.loads((archive / 'VRAM18_EVALUATION_PROTOCOL_20261027.json').read_bytes())
    index25 = read('reports/SERVER25_REMAINING_GRID_20261027.json')
    reassigned = {v['original_unlaunched_plan'] for v in index25['plans']}
    assert len(reassigned) == 8 and reassigned <= set(expanded['training_plan_sha256'])
    backend26 = dict(ssh_alias='2026', project='/data2/gb/Dual-RGBT',
                     python='/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python')
    backend25 = dict(ssh_alias='2025', project='/home/gaob/Dual-RGBT',
                     python='/home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python')
    protocols = []
    for path, spec in [('reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json', original8),
                       ('reports/VRAM18_EVALUATION_PROTOCOL_20261027.json', expanded)]:
        spec['provider'] = backend26
        spec['training_plan_sha256'] = {n:h for n,h in spec['training_plan_sha256'].items() if n not in reassigned}
        spec['source_sha256'] = sources_for(spec['training_plan_sha256'])
        spec['scope'] += ' Backend binding20261027: explicit26 provider; reassigned eight25 tasks excluded. No existing NN source or recipe changes.'
        protocols.append((path, spec))
    assert len(protocols[0][1]['training_plan_sha256']) == 8
    assert len(protocols[1][1]['training_plan_sha256']) == 24
    spec25 = dict(original8)
    spec25.update(provider=backend25,
                  source_review='reports/SERVER25_JOINT_EVALUATION_SOURCE_REVIEW_20261027.json',
                  training_plan_sha256={v['path']:v['sha256'] for v in index25['plans']},
                  datasets={'LasHeR': {'train_smoke':'data/benchmarks/lasher_train_smoke.json',
                                      'dev':'data/benchmarks/lasher_dev.json',
                                      'test':'data/benchmarks/lasher_test.json'},
                            'RGBT210': {'test':'data/benchmarks/rgbt210_test.json'},
                            'RGBT234': {'test':'data/benchmarks/rgbt234_test.json'}})
    spec25['source_sha256'] = sources_for(spec25['training_plan_sha256'])
    sizes, inputs = {}, {}
    inventory = read('reports/SERVER25_REJOIN_DATASET_INVENTORY_20261027.json')
    base = read(index25['plans'][0]['path'])
    for benchmark, splits in spec25['datasets'].items():
        sizes[benchmark] = {}
        for split, path in splits.items():
            manifest = read(path)
            assert (manifest['benchmark'], manifest['split']) == (benchmark, split)
            inputs[path] = sha(ROOT / path)
            if split == 'test':
                assert inputs[path] == inventory[Path(path).name]['manifest_sha256']
            else:
                assert inputs[path] == base['input_sha256'][path]
            sizes[benchmark][split] = dict(sequences=manifest['sequence_count'], frames=manifest['frame_count'])
    assert sizes == original8['split_sizes']
    for path in [base['factory_checkpoint'], base['checkpoint'], 'checkpoints/CADTrack_ep0014.pth.tar']:
        inputs[path] = base['input_sha256'][path]
    for path in ['reports/RANKSTOP_DEV_FREEZE_DECISION_20261026.json',
                 'reports/SERVER25_ENVIRONMENT_ACCEPTANCE_20261027.json']:
        inputs[path] = sha(ROOT / path)
    spec25['input_sha256'], spec25['split_sizes'] = inputs, sizes
    spec25['scope'] = original8['scope'] + ' Explicit25 provider and original25 manifests. Eight reassigned configurations only;26 tasks excluded. Same full-coverage, native0, tensor, DEV audit, selection and test gates.'
    path25 = ROOT / 'reports/SERVER25_JOINT_EVALUATION_PROTOCOL_20261027.json'
    assert not path25.exists()
    protocols.append((path25.relative_to(ROOT).as_posix(), spec25))
    for name, spec in protocols:
        assert all(sha(ROOT / n) == h for n,h in spec['training_plan_sha256'].items())
        (ROOT / name).write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps(dict(status='BACKEND_PROTOCOLS_REGISTERED_SOURCE_REVIEW_PENDING',
                         models={'2026':24,'2025':8},
                         protocols={n:sha(ROOT/n) for n,_ in protocols})), flush=True)


if __name__ == '__main__':
    main()
