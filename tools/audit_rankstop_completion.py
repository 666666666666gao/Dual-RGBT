"""Check the saved C882 repair on CPU; never infer a missing native exit code."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    plan_path = ROOT / 'reports/global_candidate_rankstop_trajectory_plan_20261018.json'
    plan = json.loads(plan_path.read_bytes())
    run = ROOT / 'runs' / plan['runs']['formal']
    complete = json.loads((run / 'COMPLETE.json').read_bytes())
    config = json.loads((run / 'config.json').read_bytes())
    assert complete['status'] == 'COMPLETE' and not complete['sanity']
    assert config['plan_sha256'] == sha(plan_path) and not config['sanity']
    assert config['source_sha256'] == plan['source_sha256']
    assert config['input_sha256'] == plan['input_sha256']
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha(ROOT / name) == digest, name
    assert sha(Path('/home/gaob/CADTrack_ep0014.pth.tar')) == plan['protected_checkpoint_sha256']
    assert sha(run / 'latest.pth') == complete['checkpoint_sha256']
    assert sha(run / 'query_labels.csv') == complete['private_labels_sha256']
    manifest = json.loads((ROOT / 'data/manifest.json').read_bytes())
    by_name = {row['name']: row for row in manifest['train']}
    assert len(plan['sequence_names']) == len(set(plan['sequence_names'])) == len(by_name) == 882
    assert set(plan['sequence_names']) == set(by_name) == set(complete['seen_train_sequences'])
    sequences = [by_name[name] for name in plan['sequence_names']]
    expected = [(iteration, row['name'], frame)
                for iteration in range(1, plan['rounds'] + 1) for row in sequences
                for frame in sorted(set(row['valid_frames']))
                if frame > 0 and frame % plan['label_stride'] == 0]
    counts = dict(updates=len(expected), initializes=plan['rounds'] * len(sequences),
                  tracked_frames=plan['rounds'] * sum(len(row['boxes']) - 1 for row in sequences))
    assert counts == plan['expected_formal_counts']
    assert {key: complete[key] for key in counts} == counts
    with (run / 'trajectory_steps.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [(int(row['round']), row['sequence'], int(row['frame'])) for row in rows] == expected
    assert [int(row['step']) for row in rows] == list(range(1, len(rows) + 1))
    fields = ('loss', 'quality_loss', 'gain_loss')
    assert all(math.isfinite(float(row[key])) for row in rows for key in fields)
    assert max(abs(float(row['loss']) - float(row['quality_loss']) - float(row['gain_loss']))
               for row in rows) < 1e-6
    before = torch.load(ROOT / plan['checkpoint'], map_location='cpu')
    after = torch.load(run / 'latest.pth', map_location='cpu')
    assert before['stage'] == 'candidate_b' and before['step'] == 6250
    assert after['stage'] == 'candidate_c' and after['step'] == counts['updates']
    assert after['config'] == config and before['net'].keys() == after['net'].keys()
    assert all(torch.isfinite(value).all() for value in after['net'].values())
    changed = [key for key in before['net'] if not torch.equal(before['net'][key], after['net'][key])]
    prefixes = ('executor.quality.', 'query_gain.')
    frozen = [key for key in before['net'] if not key.startswith(prefixes)]
    assert len(frozen) == 257 and len(changed) == 8
    assert all(key.startswith(prefixes) for key in changed)
    assert complete['frozen_state_tensors'] == len(frozen)
    assert complete['changed_state_tensors'] == changed
    steps = [int(state['step'].item()) for state in after['optimizer']['state'].values()]
    assert steps == [counts['updates']] * 8
    warnings = ['No native terminal status.json exists for this direct launch; actual process exit code is unknown.',
                'CPU audit does not reproduce training-time teacher/replay checks or measure tracking accuracy.']
    assert not (ROOT / 'runs/pipeline_tasks/train_global_candidate_rankstop_trajectory_formal_20261018/status.json').exists()
    report = dict(status='ARTIFACT_CHECKS_PASS_WITH_PROVENANCE_WARNING',
                  counts=counts, sequences_per_round=882, rounds=plan['rounds'],
                  checkpoint_sha256=complete['checkpoint_sha256'],
                  plan_sha256=sha(plan_path), frozen_tensors_bit_equal=len(frozen),
                  changed_state_tensors=changed, all_saved_model_tensors_finite=True,
                  exact_supervision_order=True, all_logged_losses_finite=True,
                  optimizer_steps=steps, actual_process_exit_code=None,
                  current_native_status_available=False, warnings=warnings,
                  evaluation_status='PENDING', neural_execution_during_audit=False,
                  artifacts_sha256={str(path.relative_to(ROOT)).replace('\\', '/'): sha(path)
                                    for path in (run / 'COMPLETE.json', run / 'config.json',
                                                 run / 'trajectory_steps.csv')})
    output = ROOT / 'reports/RANKSTOP_C882_COMPLETION_AUDIT_20261026.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
