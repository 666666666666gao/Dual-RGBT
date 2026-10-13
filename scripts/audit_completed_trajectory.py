"""CPU-only integrity audit of the completed, fixed first Stage-C protocol."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
run = root / 'runs/stage_c_complete_20261013'
status = json.loads((root / 'runs/pipeline_tasks/train_c_complete/status.json').read_text())
assert status['state'] == 'COMPLETE' and status['exit_code'] == 0
complete = json.loads((run / 'COMPLETE.json').read_text())
config = json.loads((run / 'config.json').read_text())
assert config['sequences'] == 20 and config['rounds'] == 2
assert config['seed'] == 2027 and config['label_stride'] == 5
assert config['policy'] == 'runs/stage_b_refreshed_20261013/latest.pth'
manifest = json.loads((root / 'data/manifest.json').read_text())
order = np.random.RandomState(config['seed']).permutation(len(manifest['train']))[:20]
selected = [manifest['train'][int(i)] for i in order]
assert config['sequence_names'] == [s['name'] for s in selected]
expected = [(iteration, sequence['name'], frame)
            for iteration in (1, 2) for sequence in selected
            for frame in range(1, len(sequence['boxes']))
            if frame % 5 == 0 and frame in set(sequence['valid_frames'])]
tracked_frames = 2 * sum(len(s['boxes']) - 1 for s in selected)
rows = list(csv.DictReader((run / 'trajectory_steps.csv').open(newline='')))
assert [(int(r['round']), r['sequence'], int(r['frame'])) for r in rows] == expected
assert [int(r['step']) for r in rows] == list(range(len(expected)))
assert complete['rounds'] == 2 and complete['sequences_per_round'] == 20
assert complete['steps'] == len(expected) == 3504
assert complete['tracked_frames'] == tracked_frames == 17708
loss_fields = ('loss', 'quality_loss', 'gain_loss', 'negative_fraction')
assert all(math.isfinite(float(r[k])) for r in rows for k in loss_fields)
assert max(abs(float(r['loss']) - float(r['quality_loss']) - float(r['gain_loss']))
           for r in rows) < 1e-6
before = torch.load(root / config['policy'], map_location='cpu')
after = torch.load(run / 'latest.pth', map_location='cpu')
assert before['step'] == 6250 and after['round'] == 2 and after['step'] == 3504
assert after['config'] == {k: v for k, v in config.items() if k != 'sequence_names'}
assert before['net'].keys() == after['net'].keys()
frozen = [k for k in before['net'] if not k.startswith(('quality.', 'gain.'))]
assert len(frozen) == 257
assert all(torch.equal(before['net'][k], after['net'][k]) for k in frozen)
changes = {prefix: sum(not torch.equal(before['net'][k], after['net'][k])
                       for k in before['net'] if k.startswith(prefix))
           for prefix in ('quality.', 'gain.')}
assert all(count > 0 for count in changes.values())
optimizer_steps = [int(state['step'].item()) for state in after['optimizer']['state'].values()]
assert len(optimizer_steps) == 8 and optimizer_steps == [3504] * 8
rounds = []
for iteration in (1, 2):
    group = [r for r in rows if int(r['round']) == iteration]
    rounds.append(dict(round=iteration, supervised_updates=len(group),
                       target_centre_outside=sum(r['target_centre_inside'] == 'False' for r in group),
                       action_counts=[sum(int(r['chosen_action']) == a for r in group) for a in range(5)],
                       modality_counts=[sum(int(r['chosen_modality']) == m for r in group) for m in range(2)],
                       mean_losses={k: sum(float(r[k]) for r in group) / len(group) for k in loss_fields}))
report = dict(status='PASS', actual_wrapper_exit_code=0, steps=3504, tracked_frames=17708,
              sequences_per_round=20, rounds=rounds, csv_keys_match_fixed_train_protocol=True,
              all_losses_finite=True, frozen_state_tensors_equal=len(frozen),
              changed_state_tensors=changes, saved_optimizer_parameter_steps=optimizer_steps,
              checkpoint_sha256=hashlib.sha256((run / 'latest.pth').read_bytes()).hexdigest(),
              neural_execution=False,
              scope='Completed fixed 20-sequence/two-round training integrity only. '
                    'Route/outside statistics cover 3,504 supervised frames, not all 17,708 track calls. '
                    'No full-882-sequence traversal, native reload forward, accuracy, or speed claim.')
(root / 'reports/completed_trajectory_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
