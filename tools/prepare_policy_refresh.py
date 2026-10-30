"""Register all882 TRAIN sequences, prioritizing the matched first100."""
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
from joint_evaluation_contract import evaluation_plan


def main():
    server = {
        Path('/home/gaob/Dual-RGBT'): '25',
        Path('/data2/gb/Dual-RGBT/snapshots/cached_parent_loss_20261030'): '26',
    }[ROOT]
    manifest_path = ROOT / {
        '25': 'inputs/server25/manifest.json',
        '26': 'inputs/policy_refresh_server26/manifest.json',
    }[server]
    original_path = ROOT / {
        '25': 'inputs/server25/manifest.json',
        '26': 'inputs/policy_refresh_server26/original_server25_manifest.json',
    }[server]
    manifest, original = [json.loads(path.read_text()) for path in (manifest_path, original_path)]
    assert sha(original_path) == '389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc'
    assert sha(manifest_path) == {
        '25': '389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc',
        '26': 'dfc64432210b053fc9db517bae0771fdeb750a5d373b732949ef43e8b65b1a7a',
    }[server]
    for split, count in (('train', 882), ('dev', 97)):
        assert len(manifest[split]) == len(original[split]) == count
        for row, before in zip(manifest[split], original[split]):
            for key in ('name', 'boxes', 'valid_frames'):
                assert row[key] == before[key], (split, key)
            for key in ('rgb', 'tir'):
                assert len(row[key]) == len(before[key])
                assert all(Path(a).name == Path(b).name for a, b in zip(row[key], before[key]))
    assert manifest['test_sequences'] == original['test_sequences']
    parent_path = 'reports/joint_candidate_evaluation/joint_quality_k8_cost0p1_server25_20261027_formal/plan.json'
    parent, _, _ = evaluation_plan(ROOT, parent_path)
    by_name = {row['name']: row for row in manifest['train']}
    names = sorted(by_name)
    random.Random(2027).shuffle(names)
    holdout = set(names[16:20] + names[20:100][4::5] + names[100:][4::5])
    sources = {name: sha(ROOT / name) for name in (
        'scripts/collect_policy_query_features.py', 'scripts/collect_frozen_query_features.py',
        'tools/summarize_frozen_query_cache.py', 'tools/audit_policy_refresh.py')}
    training = 'cached_query_gain_utility_focal_formal_20261030'
    overlay = 'runs/' + training + '/epoch_006.pth'
    native = 'runs/pipeline_tasks/train_' + training + '/status.json'
    result = 'runs/' + training + '/COMPLETE.json'
    training_plan = 'reports/CACHED_QUERY_GAIN_UTILITY_FOCAL_FORMAL_PLAN_20261030.json'
    inputs = {name: sha(ROOT / name) for name in (overlay, native, result, training_plan,
                                                manifest_path.relative_to(ROOT).as_posix(),
                                                original_path.relative_to(ROOT).as_posix())}
    assert inputs[overlay] == 'c31c54a88f5478298db1c52f44d686582ae43f053af72d40354d8a02ee703eba'
    plans = []
    for phase, selected in (('matched100', names[:100]), ('remaining782', names[100:])):
        buckets, loads = [[] for _ in range(8)], [0] * 8
        for name in sorted(selected, key=lambda name: (-len(by_name[name]['rgb']), names.index(name))):
            index = min(range(8), key=lambda index: (loads[index], index))
            buckets[index].append(name)
            loads[index] += len(by_name[name]['rgb'])
        for index, bucket in enumerate(buckets):
            if index % 2 != {'25': 1, '26': 0}[server]:
                continue
            bucket.sort(key=names.index)
            fit, held = [name for name in bucket if name not in holdout], [name for name in bucket if name in holdout]
            # A native smoke must contain a sampled valid state.
            first = next(name for name in fit + held if len(by_name[name]['rgb']) >= 31 and
                         any(frame in by_name[name]['valid_frames'] for frame in (10, 20, 30)))
            fit.sort(key=lambda name: name != first)
            held.sort(key=lambda name: name != first)
            plan = dict(mode='frozen_policy_train_refresh', parent_evaluation_plan=parent_path,
                parent_checkpoint_sha256=parent['checkpoint_sha256'], overlay=overlay, overlay_sha256=inputs[overlay],
                checkpoint_sha256=inputs[overlay],
                training_native_status=native, training_result=result, source_sha256=sources,
                input_sha256=inputs, manifest=manifest_path.relative_to(ROOT).as_posix(),
                manifest_sha256=sha(manifest_path), seed=2027, cost_weight=.15, label_stride=10,
                smoke_frames=31, fit_sequences=fit, holdout_sequences=held, full_collection_allowed=False,
                training_allowed=False, test_eligible=False, phase=phase, shard=index, expected_frames=loads[index],
                runtime_provider=dict(ssh_alias='20' + server, project=str(ROOT)),
                policy_selection='Frozen focalE6: highest completed full-dev SR among retainedE6-E10, still below parent. Diagnostic collection only.',
                sampling='Identical four-family uniform sampled requests plus deployed extra. GT does not control online decisions or which frames/queries are collected.',
                scope='TRAIN only; same first100 sequence set/partition as prior parent cache, remaining782 queued. Component holdout was seen by parent training, not unseen-system generalization.')
            name = 'policy_refresh_e6_%s_s%d_20261030' % (phase, index)
            path = ROOT / 'reports' / (name + '_smoke_plan.json')
            with path.open('x') as handle:
                handle.write(json.dumps(plan, indent=2) + '\n')
            plans.append(dict(name=name, plan=path.relative_to(ROOT).as_posix(), plan_sha256=sha(path),
                              sequences=len(bucket), frames=loads[index], phase=phase, shard=index))
    report = dict(status='REGISTERED_SMOKE_REQUIRED', total_sequences=882,
                  server=server, assigned_sequences=sum(item['sequences'] for item in plans),
                  assigned_frames=sum(item['frames'] for item in plans), jobs=plans, holdout_sequences=len(holdout),
                  ownership='Server25 owns odd shards; server26 owns even shards in each phase. Disjoint TRAIN sequences.',
                  source_sha256=sources, manifest_sha256=sha(manifest_path), test_eligible=False,
                  train_only=True, optimizer_updates=0)
    with (ROOT / 'reports/POLICY_REFRESH_E6_ALL882_PLAN_20261030.json').open('x') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
