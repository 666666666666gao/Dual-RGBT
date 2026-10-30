"""Bind all sixteen audited E6 caches to the preregistered all882 recipe."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
from joint_evaluation_contract import check_files

PREFIX = 'REFRESHED_GAIN_ALL882_PARENT_LOSS'
RUN_PREFIX = 'refreshed_query_gain_all882_parent_loss'
REVIEW = 'reports/REFRESHED_GAIN_ALL882_SOURCE_REVIEW_20261030.json'
PROTOCOL = 'docs/REFRESHED_GAIN_ALL882_PROTOCOL_20261030.md'
RECIPE = dict(variant='parent_loss', seed=2027, batch_states=2866, lr=.00112,
              min_epochs=8, max_epochs=30, patience=4, min_delta=.0001,
              minimum_recovery_states=dict(fit=100, holdout=15),
              minimum_recovery_sequences=dict(fit=10, holdout=3))


def partition(manifest):
    names = sorted(row['name'] for row in manifest['train'])
    assert len(names) == len(set(names)) == 882
    random.Random(2027).shuffle(names)
    held = set(names[16:20] + names[20:100][4::5] + names[100:][4::5])
    fit = [name for name in names if name not in held]
    assert len(fit) == 706 and len(held) == 176
    return names, fit, [name for name in names if name in held]


def collect_entries(root, names, held):
    caches, seen = [], set()
    for phase in ('matched100', 'remaining782'):
        phase_names = set(names[:100] if phase == 'matched100' else names[100:])
        for shard in range(8):
            run = 'policy_refresh_e6_%s_s%d_20261030_full' % (phase, shard)
            audit_path = root / 'reports' / (run + '_audit.json')
            audit = json.loads(audit_path.read_text())
            assert audit['status'] == 'VERIFIED_CACHE_DIAGNOSTIC'
            assert audit['complete_planned_coverage'] and not audit['smoke']
            native_path = root / 'runs/pipeline_tasks' / ('collect_' + run) / 'status.json'
            receipt_path = root / 'runs' / run / 'COMPLETE.json'
            assert sha(native_path) == audit['native_sha256']
            assert sha(receipt_path) == audit['receipt_sha256']
            native, receipt = [json.loads(path.read_text()) for path in (native_path, receipt_path)]
            config = json.loads((receipt_path.parent / 'config.json').read_text())
            assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
            assert native['command'] == config['native_command']
            assert config['plan']['phase'] == phase and config['plan']['shard'] == shard
            expected = config['plan']['fit_sequences'] + config['plan']['holdout_sequences']
            selected = set(expected)
            assert len(expected) == len(selected) and selected <= phase_names
            assert config['selected_sequences'] == expected and not seen & selected
            assert set(config['plan']['fit_sequences']) == selected - held
            assert set(config['plan']['holdout_sequences']) == selected & held
            assert receipt['status'] == 'CACHE_COMPLETE'
            assert receipt['sequences'] == len(expected)
            assert receipt['frames'] == config['plan']['expected_frames']
            seen.update(selected)
            caches.append(dict(run=run, task='collect_' + run,
                               audit=audit_path.relative_to(root).as_posix(),
                               audit_sha256=sha(audit_path)))
    assert seen == set(names)
    return caches


def bind_smoke(root, plan):
    directory = root / 'runs' / (RUN_PREFIX + '_smoke_20261030')
    result_path, config_path = directory / 'COMPLETE.json', directory / 'config.json'
    result, config = [json.loads(path.read_text()) for path in (result_path, config_path)]
    native_path = root / 'runs/pipeline_tasks' / config['task'] / 'status.json'
    native = json.loads(native_path.read_text())
    assert result['status'] == 'SANITY_PASS' and result['strict_reload_bit_equal']
    assert result['epochs'] == 2 and result['updates'] > 0
    assert result['changed_state_tensors'] == [
        'query_gain.0.weight', 'query_gain.0.bias', 'query_gain.2.weight', 'query_gain.2.bias']
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert native['command'] == config['native_command']
    smoke_plan_path = root / ('reports/' + PREFIX + '_SMOKE_PLAN_20261030.json')
    assert sha(smoke_plan_path) == result['plan_sha256'] == config['plan_sha256']
    assert json.loads(smoke_plan_path.read_text()) == config['plan']
    expected = copy.deepcopy(plan)
    expected.update(mode='smoke', formal_launch_allowed=False, min_epochs=2, max_epochs=2)
    assert config['plan'] == expected
    for part in ('fit', 'holdout'):
        assert result['baseline'][part]['sampled_recoverable_states'] >= plan['minimum_recovery_states'][part]
        assert result['baseline'][part]['recovery_sequences'] >= plan['minimum_recovery_sequences'][part]
        assert config['cache_policy_prediction_max_abs_error'][part] < 1e-5
    plan.update(smoke_result=result_path.relative_to(root).as_posix(),
                smoke_result_sha256=sha(result_path),
                smoke_native_status=native_path.relative_to(root).as_posix(),
                smoke_native_sha256=sha(native_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'formal'), required=True)
    args = parser.parse_args()
    review_path = ROOT / REVIEW
    review = json.loads(review_path.read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    check_files(ROOT, review['source_sha256'])
    assert review['protocol_sha256'] == sha(ROOT / PROTOCOL)
    original_path = ROOT / 'reports/cached_query_gain_parent_loss_formal_20261030_plan.json'
    original = json.loads(original_path.read_text())
    assert original['mode'] == 'formal'
    assert {key: original[key] for key in RECIPE} == RECIPE
    base_path = ROOT / 'reports/REFRESHED_GAIN_PARENT_LOSS_SMOKE_PLAN_20261030.json'
    plan = json.loads(base_path.read_text())
    assert plan['mode'] == 'smoke' and plan['expected_sequences'] == 100
    manifest_path = ROOT / plan['training_manifest']
    assert sha(manifest_path) == plan['training_manifest_sha256']
    names, fit, held = partition(json.loads(manifest_path.read_text()))
    caches = collect_entries(ROOT, names, set(held))
    plan.update(RECIPE)
    plan.update(mode=args.mode, formal_launch_allowed=args.mode == 'formal',
                expected_sequences=882, caches=caches, fit_sequences=fit, holdout_sequences=held,
                source_review=REVIEW, source_review_sha256=sha(review_path),
                source_sha256={name: review['source_sha256'][name] for name in (
                    'scripts/train_refreshed_query_gain.py', 'tools/prepare_refreshed_gain_all882.py')},
                matched_control_plan=original_path.relative_to(ROOT).as_posix(),
                matched_control_plan_sha256=sha(original_path),
                protocol=PROTOCOL, protocol_sha256=sha(ROOT / PROTOCOL),
                comparison='Expanded882 TRAIN-state cache: same parent initialization, loss and recipe; differs from matched100 in coverage, visited states and realized updates. Not a matched100 single-factor comparison.')
    assert plan['launch_allowed'] and plan['test_eligible'] is False
    check_files(ROOT, plan['cache_policy_inputs_sha256'])
    if args.mode == 'smoke':
        plan.update(min_epochs=2, max_epochs=2)
    else:
        bind_smoke(ROOT, plan)
    path = ROOT / ('reports/' + PREFIX + '_' + args.mode.upper() + '_PLAN_20261030.json')
    with path.open('x') as handle:
        handle.write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                          mode=args.mode, sequences=882, caches=len(caches)), indent=2))


if __name__ == '__main__':
    main()
