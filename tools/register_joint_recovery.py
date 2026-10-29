"""Register immutable recovery attestations and LasHeR-dev-only plans."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DRIFT_FILES = [
    'scripts/run_tracking.py',
    'dual_rgbt_evidence/tracker.py',
    'dual_rgbt_global/candidate_tracker.py',
    'scripts/run_benchmark_tracking.py',
]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def protocol(args):
    base = json.loads((ROOT / args.base_protocol).read_bytes())
    source = {name: sha(ROOT / name) for name in base['source_sha256']}
    protocol_path = ROOT / args.output
    review_path = ROOT / args.source_review
    plans = [p.replace('\\', '/') for p in args.training_plans]
    training_hashes = {p: sha(ROOT / p) for p in plans}
    spec = dict(
        status='RECOVERY_READY',
        recovery_protocol_id='joint_candidate_recovery_lasher_dev_20261029',
        source_review=args.source_review,
        source_sha256=source,
        input_sha256=base['input_sha256'],
        training_plan_sha256=training_hashes,
        datasets={'LasHeR': {'dev': base['datasets']['LasHeR']['dev']}},
        split_sizes={'LasHeR': {'dev': base['split_sizes']['LasHeR']['dev']}},
        formal_counts=base['formal_counts'],
        seed=base['seed'],
        total_parameters=base['total_parameters'],
        dev_cost_candidates=base['dev_cost_candidates'],
        reference_decision=base['reference_decision'],
        initialization=base['initialization'],
        selection_criterion=base['selection_criterion'],
        provider=base['provider'],
        evidence_schema=base['evidence_schema'],
        scope=dict(benchmark='LasHeR', split='dev', test_eligible=False),
        test_eligible=False,
        training_plans=plans,
        source_drift_files=DRIFT_FILES,
        baseline_revision='0abf763^',
    )
    write_json(protocol_path, spec)
    review = dict(
        status='RECOVERY_SOURCE_PASS',
        blocking_issues=0,
        protocol=args.output,
        protocol_sha256=sha(protocol_path),
        source_sha256=source,
        scope=spec['scope'],
        test_eligible=False,
        review_independence='local-static-review',
        acceptance_status='provisional',
        notes='Separate recovery gate for native FAILED training receipts; LasHeR-dev only.',
    )
    write_json(review_path, review)
    print(json.dumps(dict(protocol=args.output, source_review=args.source_review,
                          training_plans=plans), indent=2))


def tensor_witness(training, candidate):
    state = torch.load(ROOT / 'runs' / candidate / 'latest.pth', map_location='cpu')
    parent = torch.load(ROOT / training['checkpoint'], map_location='cpu')
    assert state['stage'] == 'candidate_c' and state['step'] == 182182
    assert parent['stage'] == 'candidate_b' and parent['step'] == 6250
    assert tuple(state['net']) == tuple(parent['net']) and len(state['net']) == 265
    assert all(value.shape == parent['net'][name].shape and value.dtype == parent['net'][name].dtype
               for name, value in state['net'].items())
    assert all(torch.isfinite(value).all() for value in state['net'].values())
    changed = sorted(name for name in state['net'] if not torch.equal(state['net'][name], parent['net'][name]))
    assert len(changed) == 14
    return dict(state_tensors=265, changed_state_tensors=changed, frozen_state_tensors=251,
                all_state_finite=True, parent_checkpoint_sha256=sha(ROOT / training['checkpoint']))


def attestation(args, protocol_spec, training_path):
    training_path = training_path.replace('\\', '/')
    training = json.loads((ROOT / training_path).read_bytes())
    candidate = training['runs']['formal']
    task = training['tasks']['formal']
    run_dir = ROOT / 'runs' / candidate
    task_dir = ROOT / 'runs/pipeline_tasks' / task
    native_path = task_dir / 'status.json'
    native = json.loads(native_path.read_bytes())
    config_path = run_dir / 'config.json'
    checkpoint_path = run_dir / 'latest.pth'
    trajectory_path = run_dir / 'trajectory_steps.csv'
    labels_path = run_dir / 'query_labels.csv'
    log_path = task_dir / 'run.log'
    assert native['state'] == 'FAILED' and native['exit_code'] == 1
    assert not (run_dir / 'COMPLETE.json').exists()
    config = json.loads(config_path.read_bytes())
    assert config['run'] == candidate and config['plan'] == training_path
    assert config['source_sha256'] == training['source_sha256']
    assert config['input_sha256'] == training['input_sha256']
    assert config['stage'] == 'candidate_c' and not config['sanity']
    assert all(sha(ROOT / name) == digest for name, digest in training['input_sha256'].items())
    with trajectory_path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 182182
    assert [int(row['step']) for row in rows] == list(range(1, 182183))
    assert all(float(row['loss']) == float(row['loss']) for row in rows)
    drift = [name for name in DRIFT_FILES if sha(ROOT / name) != training['source_sha256'][name]]
    assert drift == DRIFT_FILES
    assert all(os.path.getmtime(ROOT / name) > native['started'] for name in drift)
    witness = tensor_witness(training, candidate)
    attestation_path = ROOT / args.attestation_dir / ('recovery_attestation_' + candidate + '.json')
    att = dict(
        schema='dual_rgbt.recovery_attestation.v1',
        status='RECOVERY_ATTESTED_DEV_ONLY',
        scope=protocol_spec['scope'],
        candidate_id=candidate,
        training_plan=training_path,
        training_plan_sha256=sha(ROOT / training_path),
        native_status_path=native_path.relative_to(ROOT).as_posix(),
        native_status_sha256=sha(native_path),
        native_state=native['state'], native_exit_code=native['exit_code'], native_task=native['task'],
        native_physical_gpu=native['physical_gpu'], native_command=native['command'],
        native_started=native['started'], native_completed=native['completed'],
        complete_json_present=False,
        config_path=config_path.relative_to(ROOT).as_posix(), config_sha256=sha(config_path),
        checkpoint_path=checkpoint_path.relative_to(ROOT).as_posix(), checkpoint_sha256=sha(checkpoint_path),
        trajectory_path=trajectory_path.relative_to(ROOT).as_posix(), trajectory_sha256=sha(trajectory_path),
        labels_path=labels_path.relative_to(ROOT).as_posix(), labels_sha256=sha(labels_path),
        task_log_path=log_path.relative_to(ROOT).as_posix(), task_log_sha256=sha(log_path),
        training_source_sha256=training['source_sha256'], training_input_sha256=training['input_sha256'],
        input_verification=dict(status='PASS', files=training['input_sha256']),
        training_counts=dict(updates=182182, tracked_frames=917106, initializes=1764,
                             trajectory_rows=len(rows), first_step=1, last_step=182182,
                             round_updates=[91091, 91091]),
        tensor_witness=witness,
        source_drift=dict(post_start_only=True, files=drift, baseline_revision='0abf763^'),
        baseline_revision='0abf763^', test_eligible=False,
    )
    write_json(attestation_path, att)
    plan_path = ROOT / args.plan_dir / ('recovery_evaluation_plan_' + candidate + '.json')
    plan = dict(
        status='RECOVERY_READY', test_eligible=False, scope=protocol_spec['scope'], candidate_id=candidate,
        protocol=args.protocol, protocol_sha256=sha(ROOT / args.protocol), training_plan=training_path,
        training_plan_sha256=sha(ROOT / training_path), recovery_attestation=attestation_path.relative_to(ROOT).as_posix(),
        recovery_attestation_sha256=sha(attestation_path), checkpoint=att['checkpoint_path'],
        checkpoint_sha256=att['checkpoint_sha256'], training_config_sha256=att['config_sha256'],
        factory_checkpoint=training['factory_checkpoint'], max_candidates=training['max_candidates'],
        source_sha256=protocol_spec['source_sha256'], input_sha256=protocol_spec['input_sha256'],
    )
    write_json(plan_path, plan)
    print(json.dumps(dict(candidate=candidate, plan=plan_path.relative_to(ROOT).as_posix(),
                          checkpoint_sha256=att['checkpoint_sha256']), indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='operation', required=True)
    p = sub.add_parser('protocol')
    p.add_argument('--base-protocol', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source-review', required=True)
    p.add_argument('--training-plans', nargs='+', required=True)
    p = sub.add_parser('attestations')
    p.add_argument('--protocol', required=True)
    p.add_argument('--source-review', required=True)
    p.add_argument('--attestation-dir', required=True)
    p.add_argument('--plan-dir', required=True)
    p.add_argument('--training-plans', nargs='+', required=True)
    args = parser.parse_args()
    if args.operation == 'protocol':
        protocol(args)
    else:
        spec = json.loads((ROOT / args.protocol).read_bytes())
        assert spec['status'] == 'RECOVERY_READY'
        assert sha(ROOT / args.source_review) == sha(ROOT / spec['source_review'])
        for training_path in args.training_plans:
            attestation(args, spec, training_path)


if __name__ == '__main__':
    main()
