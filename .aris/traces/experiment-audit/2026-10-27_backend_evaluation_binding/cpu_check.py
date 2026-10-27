"""Synthetic CPU guard checks only; no model, dataset scoring or GPU execution."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import joint_evaluation_contract as contract

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
protocols = [
    'reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json',
    'reports/VRAM18_EVALUATION_PROTOCOL_20261027.json',
    'reports/SERVER25_JOINT_EVALUATION_PROTOCOL_20261027.json',
]
specs = [json.loads((ROOT / n).read_bytes()) for n in protocols]
checks = []
assert [len(s['training_plan_sha256']) for s in specs] == [8, 24, 8]
assert set(specs[0]['training_plan_sha256']) <= set(specs[1]['training_plan_sha256'])
assert not set(specs[1]['training_plan_sha256']) & set(specs[2]['training_plan_sha256'])
checks.append('exact_disjoint_provider_case_counts_24_plus_8')
for spec in specs:
    contract.check_files(ROOT, spec['source_sha256'])
    contract.check_files(ROOT, spec['training_plan_sha256'])
    cfg = dict(plan='synthetic-plan.json', benchmark='LasHeR', split='dev', task='synthetic-task',
               run='synthetic-run', cost_weight=0.05, shard=0, shards=1, smoke_frames=0)
    command = contract.evaluation_command(cfg, spec)
    assert command[:2] == [spec['provider']['python'], spec['provider']['project'] + '/scripts/run_joint_candidate_evaluation.py']
    assert '--selection-sha256' not in command
    cfg.update(split='test', selection_sha256='synthetic-frozen-selection-sha')
    test_command = contract.evaluation_command(cfg, spec)
    assert test_command[-2:] == ['--selection-sha256', cfg['selection_sha256']]
    checks.append('explicit_%s_native_paths_and_frozen_test_arg' % spec['provider']['ssh_alias'])
    # A RUNNING receipt must be rejected before any COMPLETE/config/weight access.
    training_path = next(iter(spec['training_plan_sha256']))
    training = json.loads((ROOT / training_path).read_bytes())
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / training_path
        path.parent.mkdir(parents=True)
        path.write_bytes((ROOT / training_path).read_bytes())
        native = root / 'runs/pipeline_tasks' / training['tasks']['formal'] / 'status.json'
        native.parent.mkdir(parents=True)
        native.write_text(json.dumps(dict(state='RUNNING', exit_code=None)))
        rejected = False
        try:
            contract.training_metadata(root, spec, training_path)
        except AssertionError:
            rejected = True
        assert rejected
        assert not (root / 'runs' / training['runs']['formal']).exists()
    checks.append('running_training_rejected_before_checkpoint_access_%s' % spec['provider']['ssh_alias'])
protected_path = ROOT / '.aris/traces/experiment-audit/2026-10-27_backend_evaluation_binding/protected_training_files.json'
protected = json.loads(protected_path.read_bytes())
assert all(sha(ROOT / name) == digest for name, digest in protected.items())
checks.append('all_%d_protected_training_files_unchanged' % len(protected))
result = dict(status='CPU_GUARD_CHECKS_PASS', scope=__doc__, checks=checks,
              protocol_sha256={n: sha(ROOT / n) for n in protocols},
              source_sha256=sha(Path(__file__)), neural_network_runs=0, weights_read=0)
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
