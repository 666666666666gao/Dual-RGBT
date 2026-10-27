"""CPU integrity audit after the fixed all-882 Stage-C task exits zero."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / args.plan).read_text())
    run = root / 'runs/stage_c_all882_20261013'
    status = json.loads((root / 'runs/pipeline_tasks/train_c_all882_20261013/status.json').read_text())
    assert status['state'] == 'COMPLETE' and status['exit_code'] == 0, status
    manifest_path = root / 'data/manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == plan['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['train']) == plan['sequences'] == 882
    order = np.random.RandomState(plan['seed']).permutation(882)
    selected = [manifest['train'][int(i)] for i in order]
    complete = json.loads((run / 'COMPLETE.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    assert config['sequence_names'] == plan['sequence_names'] == [s['name'] for s in selected]
    for key in ('policy', 'sequences', 'rounds', 'label_stride', 'cost_weight', 'seed'):
        value = plan['input_checkpoint'] if key == 'policy' else plan[key]
        assert config[key] == value, key
    assert config['sanity'] is False
    expected = []
    for iteration in (1, 2):
        for sequence in selected:
            valid = set(sequence['valid_frames'])
            expected.extend((iteration, sequence['name'], frame)
                            for frame in range(5, len(sequence['boxes']), 5) if frame in valid)
    with (run / 'trajectory_steps.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [(int(r['round']), r['sequence'], int(r['frame'])) for r in rows] == expected
    assert [int(r['step']) for r in rows] == list(range(len(expected)))
    assert complete['rounds'] == 2 and complete['sequences_per_round'] == 882
    assert complete['steps'] == len(expected) == plan['expected_updates']
    assert complete['tracked_frames'] == 2 * sum(len(s['boxes']) - 1 for s in selected) == plan['expected_tracked_frames']
    fields = ('loss', 'quality_loss', 'gain_loss', 'negative_fraction')
    assert all(math.isfinite(float(r[k])) for r in rows for k in fields)
    assert all(abs(float(r['loss']) - float(r['quality_loss']) - float(r['gain_loss'])) < 1e-6 for r in rows)
    policy = root / plan['input_checkpoint']
    assert hashlib.sha256(policy.read_bytes()).hexdigest() == plan['input_checkpoint_sha256']
    before = torch.load(policy, map_location='cpu')
    after = torch.load(run / 'latest.pth', map_location='cpu')
    assert before['step'] == 6250 and after['round'] == 2 and after['step'] == len(expected)
    assert after['config'] == {k: v for k, v in config.items() if k != 'sequence_names'}
    assert before['net'].keys() == after['net'].keys()
    frozen = [k for k in before['net'] if not k.startswith(('quality.', 'gain.'))]
    assert len(frozen) == 257 and all(torch.equal(before['net'][k], after['net'][k]) for k in frozen)
    changes = {prefix: sum(not torch.equal(before['net'][k], after['net'][k])
                           for k in before['net'] if k.startswith(prefix))
               for prefix in ('quality.', 'gain.')}
    assert all(count > 0 for count in changes.values())
    optimizer_steps = [int(state['step'].item()) for state in after['optimizer']['state'].values()]
    assert optimizer_steps == [len(expected)] * 8
    rounds = []
    for iteration in (1, 2):
        group = [r for r in rows if int(r['round']) == iteration]
        rounds.append(dict(round=iteration, supervised_updates=len(group),
                           target_centre_outside=sum(r['target_centre_inside'] == 'False' for r in group),
                           action_counts=[sum(int(r['chosen_action']) == a for r in group) for a in range(5)],
                           modality_counts=[sum(int(r['chosen_modality']) == m for r in group) for m in range(2)],
                           mean_losses={k: sum(float(r[k]) for r in group) / len(group) for k in fields}))
    report = dict(status='PASS', actual_wrapper_exit_code=0, sequences_per_round=882,
                  steps=len(expected), tracked_frames=complete['tracked_frames'], rounds=rounds,
                  exact_supervision_keys=True, all_losses_finite=True,
                  frozen_state_tensors_equal=len(frozen), changed_state_tensors=changes,
                  optimizer_steps=optimizer_steps, neural_execution=False,
                  checkpoint_sha256=hashlib.sha256((run / 'latest.pth').read_bytes()).hexdigest(),
                  input_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (root / args.plan, policy, manifest_path, run / 'config.json',
                                          run / 'COMPLETE.json', run / 'trajectory_steps.csv')},
                  scope='Completed 882-sequence/two-round training integrity. Route/outside statistics '
                        'cover supervised frames only, not every track call; no NN reload, accuracy or speed claim.')
    output = root / 'reports/all_train_integrity.json'
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
