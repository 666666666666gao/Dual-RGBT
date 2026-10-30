"""Register the next fixed80train sequences without selecting by outcomes."""
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = json.loads((ROOT / 'reports/FROZEN_QUERY_CACHE_PLAN_20261030.json').read_text())
    manifest_path = ROOT / original['manifest']
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == original['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    by_name = {s['name']: s for s in manifest['train']}
    assert len(by_name) == 882
    names = sorted(by_name)
    random.Random(2027).shuffle(names)
    assert names[:20] == original['fit_sequences'] + original['holdout_sequences']
    selected = names[20:100]
    holdout = set(selected[4::5])
    buckets, loads = [[] for _ in range(4)], [0] * 4
    for name in sorted(selected, key=lambda n: (-len(by_name[n]['rgb']), selected.index(n))):
        shard = min(range(4), key=lambda i: (loads[i], i))
        buckets[shard].append(name)
        loads[shard] += len(by_name[name]['rgb'])
    plans = []
    for shard, bucket in enumerate(buckets):
        bucket.sort(key=selected.index)
        plan = {k: original[k] for k in ('mode', 'parent_evaluation_plan', 'checkpoint_sha256',
            'manifest', 'manifest_sha256', 'source_sha256', 'seed', 'cost_weight',
            'label_stride', 'smoke_frames', 'test_eligible', 'training_allowed')}
        plan.update(fit_sequences=[n for n in bucket if n not in holdout],
            holdout_sequences=[n for n in bucket if n in holdout], full_collection_allowed=False,
            selection='Seed2027 shuffled882train names, indices20:100. Every fifth NEW sequence is holdout. Greedy frame-count balancing across4shards; no outcome-based selection.',
            shard=shard, expected_frames=loads[shard], estimated_minutes=[20, 40])
        assert plan['fit_sequences'] and plan['holdout_sequences']
        assert len(by_name[plan['fit_sequences'][0]]['rgb']) >= plan['smoke_frames']
        rel = 'reports/FROZEN_QUERY_EXPANSION_S%d_SMOKE_PLAN_20261030.json' % shard
        path = ROOT / rel
        assert not path.exists()
        path.write_text(json.dumps(plan, indent=2) + '\n')
        plans.append(dict(shard=shard, plan=rel, plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fit_sequences=len(plan['fit_sequences']), holdout_sequences=len(plan['holdout_sequences']),
            frames=loads[shard]))
    report = dict(status='PLANNED_NOT_LAUNCHED', training_allowed=False, new_sequences=80,
        new_fit_sequences=64, new_holdout_sequences=16, total_frames=sum(loads), shards=plans,
        previous_sequences_recollected=0,
        scope='Additional frozen-parent train-state coverage. Holdout seen by parent training; not benchmark test data. Fresh31frame smoke per shard required before full admission.')
    path = ROOT / 'reports/FROZEN_QUERY_EXPANSION_PLAN_20261030.json'
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
