"""CPU reconstruction of the completed first A/B sampling scope, not a rerun.

Uses the current manifest and the sampler's first RNG draws. No image loading,
Torch, checkpoint loading, training, or evaluation is performed.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTION_NAMES = ('STOP', 'PRIMARY', 'SECONDARY', 'PREVIEW', 'EXPAND')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(run_name, filename, sequences, valid_frames, routes):
    run = ROOT / 'runs' / run_name
    complete = json.loads((run / 'COMPLETE.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    with (run / filename).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    epochs = sorted({int(r['epoch']) for r in rows})
    batch = config['batch_size']
    assert config['samples'] % batch == 0
    batches = config['samples'] // batch
    assert len(epochs) == config['epochs'] == complete['epochs']
    assert epochs == list(range(epochs[0], epochs[0] + config['epochs']))
    assert len(rows) == len(epochs) * batches
    assert [int(r['step']) for r in rows] == list(range(int(rows[0]['step']), complete['steps']))
    assert [int(r['epoch']) for r in rows] == [e for e in epochs for _ in range(batches)]
    if routes is not None:
        assert all(int(r['modality']) == int(r['step']) % 2 and
                   int(r['action']) == (int(r['step']) // 2) % 5 for r in rows)
    counts, targets = Counter(), set()
    for epoch_index, epoch in enumerate(epochs):
        dataset_seed = config['seed'] + (epoch - 1) * config['samples']
        for index in range(config['samples']):
            rng = np.random.RandomState(dataset_seed + index)
            sequence_id = rng.randint(len(sequences))
            valid = valid_frames[sequence_id]
            frame = int(valid[rng.randint(2, len(valid))])
            name = sequences[sequence_id]['name']
            counts[name] += 1
            targets.add((name, frame))
            if routes is not None:
                row = rows[epoch_index * batches + index // batch]
                key = (int(row['modality']), int(row['action']))
                routes[key]['counts'][name] += 1
                routes[key]['targets'].add((name, frame))
    return dict(run=run_name, epochs=epochs, sampled_pairs=sum(counts.values()),
                unique_sequences=len(counts), unique_search_targets=len(targets),
                config_sha256=digest(run / 'config.json'),
                complete_sha256=digest(run / 'COMPLETE.json'), csv_sha256=digest(run / filename)), counts, targets


def main():
    manifest_path = ROOT / 'data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sequences = manifest['train']
    names = [s['name'] for s in sequences]
    assert len(names) == len(set(names)) == 882
    valid = [np.array(s['valid_frames']) for s in sequences]
    assert all(len(v) >= 3 for v in valid)
    routes = {(m, a): dict(counts=Counter(), targets=set()) for m in range(2) for a in range(5)}
    short, short_counts, short_targets = replay('stage_a_spatial_20261013', 'steps.csv', sequences, valid, routes)
    full, full_counts, full_targets = replay('stage_a_spatial_full_20261013', 'steps.csv', sequences, valid, routes)
    policy, policy_counts, policy_targets = replay('stage_b_refreshed_20261013', 'policy_steps.csv', sequences, valid, None)
    assert short['sampled_pairs'] == 4000 and full['sampled_pairs'] == policy['sampled_pairs'] == 50000
    assert short['epochs'] == [1, 2] and full['epochs'] == [3, 4, 5, 6, 7] and policy['epochs'] == [1, 2, 3, 4, 5]
    executor_counts = short_counts + full_counts
    executor_targets = short_targets | full_targets
    per_route = []
    for (modality, action), group in routes.items():
        counts = group['counts']
        per_route.append(dict(modality=['RGB', 'TIR'][modality], action=ACTION_NAMES[action],
                              sampled_pairs=sum(counts.values()), unique_sequences=len(counts),
                              unique_search_targets=len(group['targets']),
                              sequences_without_sample=[name for name in names if counts[name] == 0]))
    c_run = ROOT / 'runs/stage_c_complete_20261013'
    c_config = json.loads((c_run / 'config.json').read_text())
    c_complete = json.loads((c_run / 'COMPLETE.json').read_text())
    assert c_complete['tracked_frames'] == 17708 and c_complete['steps'] == 3504
    c_names = set(c_config['sequence_names'])
    assert len(c_names) == 20 and c_names <= set(names)
    c_total = sum(len(s['boxes']) - 1 for s in sequences if s['name'] in c_names)
    assert 2 * c_total == c_complete['tracked_frames']
    total = sum(len(s['boxes']) - 1 for s in sequences)
    eligible = sum(len(v) - 2 for v in valid)
    result = dict(scope='Current-manifest deterministic reconstruction of completed A/B sampled search frames and C coverage.',
                  manifest_sha256=digest(manifest_path), numpy_version=np.__version__,
                  source_sha256={p: digest(ROOT / p) for p in ('dual_rgbt/data.py', 'scripts/train_executor.py',
                                                              'scripts/train_policy.py', 'scripts/audit_sampling_scope.py')},
                  train_sequences=882, train_noninitialization_frames=total,
                  sampler_eligible_search_targets=eligible, runs=[short, full, policy],
                  stage_a=dict(sampled_pairs=sum(executor_counts.values()), unique_sequences=len(executor_counts),
                               unique_search_targets=len(executor_targets),
                               fraction_of_eligible_search_targets=len(executor_targets) / eligible,
                               per_route=per_route),
                  stage_b=dict(sampled_pairs=sum(policy_counts.values()), unique_sequences=len(policy_counts),
                               unique_search_targets=len(policy_targets),
                               fraction_of_eligible_search_targets=len(policy_targets) / eligible),
                  stage_c=dict(sequences=20, rounds=2, tracked_frames=17708, supervised_updates=3504,
                               distinct_noninitialization_frames=c_total,
                               fraction_of_train_noninitialization_frames=c_total / total,
                               hypothetical_all_train_two_round_track_calls=2 * total),
                  per_sequence=[dict(sequence=name, stage_a_pairs=executor_counts[name],
                                      stage_b_pairs=policy_counts[name], stage_c_selected=name in c_names) for name in names],
                  neural_execution=False,
                  limitations=[
                      'Reconstruction uses the current manifest; it is not a historical per-item image-load trace.',
                      'The manifest SHA binds this audit input, not an independently attested launch-time snapshot.',
                      'Unique search targets count sequence/frame pairs, not templates, crop coordinates or pixels.',
                      'Stage A route membership follows sequential drop_last batches and saved step routes.',
                      'Stage B label generation evaluates both modalities/actions; no per-batch route partition is applied here.',
                      'A/B sequence participation is not complete frame traversal; C visits only its selected 20 sequences.',
                      'The hypothetical whole-train C call count is a budget calculation, not a launched or completed run.',
                      'No image equality, checkpoint reload, accuracy, speed or module-effectiveness claim.'
                  ])
    output = ROOT / 'reports/sampling_scope_audit.json'
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'per_sequence'}, indent=2))


if __name__ == '__main__':
    main()
