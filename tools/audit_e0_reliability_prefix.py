"""Audit real E0 prefix input coverage without fitting the reliability head."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, run, draft_path):
    directory = root / 'runs' / run
    complete_path, config_path = directory / 'COMPLETE.json', directory / 'config.json'
    complete, config = [json.loads(path.read_bytes()) for path in (complete_path, config_path)]
    assert complete['status'] == 'TRAIN_OBSERVER_PARITY_PASS'
    assert complete['split'] == config['split'] == 'train'
    assert complete['feature_width'] == 914 and complete['feature_states'] == 90
    assert complete['optimizer_updates'] == complete['backward_calls'] == 0
    assert config['plan']['training_allowed'] is False
    assert complete['checkpoint_sha256'] == config['plan']['checkpoint_sha256']
    assert complete['checkpoint_sha256'] == 'ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c'
    assert sha(root / config['plan']['checkpoint']) == complete['checkpoint_sha256']
    plan_path = root / 'reports/E0_OBSERVER_TRAIN_PARITY_PLAN_20261031.json'
    assert sha(plan_path) == complete['plan_sha256'] == config['plan_sha256']
    assert json.loads(plan_path.read_bytes()) == config['plan']
    native_path = root / 'runs/pipeline_tasks' / config['task'] / 'status.json'
    native = json.loads(native_path.read_bytes())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert native['command'] == config['native_command']
    assert native['task'] == complete['task'] == config['task']
    used = {path.relative_to(root).as_posix(): sha(path)
            for path in (complete_path, native_path, plan_path, draft_path)}
    for relative, digest in complete['artifacts_sha256'].items():
        path = directory / relative
        assert sha(path) == digest
        used[path.relative_to(root).as_posix()] = digest
    for field in ('source_sha256', 'input_sha256'):
        for relative, digest in config['plan'][field].items():
            assert sha(root / relative) == digest, relative
    draft = json.loads(draft_path.read_bytes())
    assert draft['full_collection_allowed'] is False
    assert draft['checkpoint_sha256'] == complete['checkpoint_sha256']
    fit, held = set(draft['fit_sequences']), set(draft['holdout_sequences'])
    assert len(fit) == 706 and len(held) == 176 and fit.isdisjoint(held)
    with (directory / 'labels.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    features = np.load(directory / 'features.npy', allow_pickle=False)
    assert features.shape == (len(rows), 914) == (90, 914)
    assert features.dtype == np.float32 and np.isfinite(features).all()
    assert len({(r['sequence'], int(r['frame'])) for r in rows}) == 90
    assert {r['sequence'] for r in rows} == set(config['plan']['sequence_names'])
    counts = collections.Counter()
    partitions = {p: set() for p in ('fit', 'holdout')}
    for row in rows:
        iou = float(row['final_iou'])
        assert 0 <= iou <= 1 and 1 <= int(row['frame']) <= 30
        state = 'reliable' if iou >= .5 else 'lost' if iou < .1 else 'uncertain'
        counts[state] += 1
        assert row['sequence'] in fit | held
        partitions['fit' if row['sequence'] in fit else 'holdout'].add(row['sequence'])
    return dict(status='VERIFIED_E0_PREFIX_RELIABILITY_COVERAGE', split='train',
        run=run, checkpoint_sha256=complete['checkpoint_sha256'], states=len(rows),
        feature_width=914, class_counts={s: counts[s] for s in ('reliable', 'uncertain', 'lost')},
        sequences_by_partition={p: sorted(names) for p, names in partitions.items()},
        action_counts=dict(collections.Counter(int(r['action']) for r in rows)),
        source_files_reverified=len(config['plan']['source_sha256']),
        input_files_reverified=len(config['plan']['input_sha256']),
        used_artifacts_sha256=used, audit_source_sha256=sha(Path(__file__)),
        m1_training_admitted=False, draft_full_collection_allowed=draft['full_collection_allowed'],
        optimizer_updates=0, backward_calls=0, test_eligible=False,
        scientific_result_audit_accepted=False,
        scope='Existing three TRAIN prefixes only. Input/artifact hashes and finite 914-wide '
              'features checked without fitting or running the new head. These are localization '
              'overlap categories, not target-identity, modality-conflict or cloned recovery labels. '
              'Zero lost examples cannot validate loss detection. Neither full TRAIN coverage '
              'nor independent held-out generalization is established. Collector review remains required.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--draft', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists()
    result = audit(ROOT, args.run, ROOT / args.draft)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
