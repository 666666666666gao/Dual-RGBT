"""Register the matched100 visited-state comparison without changing its budget."""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
from joint_evaluation_contract import check_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'formal'), required=True)
    args = parser.parse_args()
    control_path = ROOT / 'reports/cached_query_gain_parent_loss_formal_20261030_plan.json'
    control = json.loads(control_path.read_text())
    assert control['mode'] == 'formal' and control['variant'] == 'parent_loss'
    review_path = ROOT / 'reports/REFRESHED_GAIN_SOURCE_REVIEW_20261030.json'
    review = json.loads(review_path.read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    check_files(ROOT, review['source_sha256'])
    manifest_path = ROOT / 'inputs/server25/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    names = sorted(row['name'] for row in manifest['train'])
    random.Random(2027).shuffle(names)
    selected = names[:100]
    held = set(names[16:20] + names[20:100][4::5])
    fit = [name for name in selected if name not in held]
    caches, seen = [], set()
    for shard in range(8):
        run = 'policy_refresh_e6_matched100_s%d_20261030_full' % shard
        audit_path = ROOT / 'reports' / (run + '_audit.json')
        audit = json.loads(audit_path.read_text())
        assert audit['status'] == 'VERIFIED_CACHE_DIAGNOSTIC' and audit['complete_planned_coverage']
        assert audit['smoke'] is False
        config = json.loads((ROOT / 'runs' / run / 'config.json').read_text())
        expected = config['plan']['fit_sequences'] + config['plan']['holdout_sequences']
        assert config['selected_sequences'] == expected and not seen & set(expected)
        assert set(expected) <= set(selected)
        assert set(config['plan']['fit_sequences']) == set(expected) - held
        assert set(config['plan']['holdout_sequences']) == set(expected) & held
        seen.update(expected)
        caches.append(dict(run=run, task='collect_' + run,
                           audit=audit_path.relative_to(ROOT).as_posix(), audit_sha256=sha(audit_path)))
    assert seen == set(selected) and len(fit) == 80 and len(held) == 20
    training = 'cached_query_gain_utility_focal_formal_20261030'
    overlay = 'runs/' + training + '/epoch_006.pth'
    native = 'runs/pipeline_tasks/train_' + training + '/status.json'
    result = 'runs/' + training + '/COMPLETE.json'
    inputs = {name: sha(ROOT / name) for name in (overlay, native, result)}
    assert inputs[overlay] == 'c31c54a88f5478298db1c52f44d686582ae43f053af72d40354d8a02ee703eba'
    plan = {key: control[key] for key in (
        'parent_evaluation_plan', 'parent_checkpoint_sha256', 'variant', 'seed', 'batch_states',
        'lr', 'min_epochs', 'max_epochs', 'patience', 'min_delta',
        'minimum_recovery_states', 'minimum_recovery_sequences')}
    plan.update(mode=args.mode, launch_allowed=True, formal_launch_allowed=args.mode == 'formal',
                test_eligible=False, expected_sequences=100, caches=caches, fit_sequences=fit,
                holdout_sequences=[name for name in selected if name in held],
                source_sha256={'scripts/train_refreshed_query_gain.py': sha(ROOT / 'scripts/train_refreshed_query_gain.py')},
                source_review=review_path.relative_to(ROOT).as_posix(), source_review_sha256=sha(review_path),
                training_manifest=manifest_path.relative_to(ROOT).as_posix(),
                training_manifest_sha256=sha(manifest_path),
                cache_policy_overlay=overlay, cache_checkpoint_sha256=inputs[overlay],
                cache_policy_native=native, cache_policy_result=result,
                cache_policy_inputs_sha256=inputs, matched_control_plan=control_path.relative_to(ROOT).as_posix(),
                matched_control_plan_sha256=sha(control_path),
                comparison='Same80fit/20holdout sequences, original parent initialization/loss/batch/LR/seed/stopping; only frozen-policy visited-state cache changes. Realized states and updates may differ.',
                retention='Preserve latestfive weights and evaluate each on full97dev; dev only chooses settings/weights.')
    if args.mode == 'smoke':
        plan.update(min_epochs=2, max_epochs=2)
    else:
        smoke_run = 'refreshed_query_gain_parent_loss_smoke_20261030'
        smoke_path = ROOT / 'runs' / smoke_run / 'COMPLETE.json'
        smoke_native = ROOT / 'runs/pipeline_tasks' / ('train_' + smoke_run) / 'status.json'
        smoke = json.loads(smoke_path.read_text())
        terminal = json.loads(smoke_native.read_text())
        assert smoke['status'] == 'SANITY_PASS' and smoke['strict_reload_bit_equal']
        assert terminal['state'] == 'COMPLETE' and terminal['exit_code'] == 0
        for part in ('fit', 'holdout'):
            assert smoke['baseline'][part]['sampled_recoverable_states'] >= plan['minimum_recovery_states'][part]
            assert smoke['baseline'][part]['recovery_sequences'] >= plan['minimum_recovery_sequences'][part]
        plan.update(smoke_result=smoke_path.relative_to(ROOT).as_posix(), smoke_result_sha256=sha(smoke_path),
                    smoke_native_status=smoke_native.relative_to(ROOT).as_posix(), smoke_native_sha256=sha(smoke_native))
    path = ROOT / ('reports/REFRESHED_GAIN_PARENT_LOSS_%s_PLAN_20261030.json' % args.mode.upper())
    with path.open('x') as handle:
        handle.write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), mode=args.mode), indent=2))


if __name__ == '__main__':
    main()
