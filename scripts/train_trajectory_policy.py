"""Stage C: refresh policy supervision on its own causal training trajectories.

GT initializes each sequence and measures counterfactual losses only. Predictions
control subsequent search crops and template updates. No test/dev sequences are
used here. Executor and preview stay frozen; quality/gain learn real history.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from dual_rgbt.data import read_pair, crop_pair
from dual_rgbt.model import DualRGBT
from dual_rgbt.losses import localization_loss
from dual_rgbt.tracker import Tracker
from train_policy import labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--sequences', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--label-stride', type=int, default=5)
    parser.add_argument('--cost-weight', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--sanity', action='store_true',
                        help='Check two real trajectory updates, frozen state and saved-state reload only.')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads((root / 'data/manifest.json').read_text())
    order = np.random.RandomState(args.seed).permutation(len(manifest['train']))[:args.sequences]
    sequences = [manifest['train'][int(i)] for i in order]
    model = DualRGBT().cuda().eval()
    model.load_state_dict(torch.load(args.policy, map_location='cpu')['net'], strict=True)
    for module in (model.backbone, model.decode, model.box_head, model.preview):
        module.requires_grad_(False)
    parameters = list(model.quality.parameters()) + list(model.gain.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    if args.sanity:
        before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    run = root / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    (run / 'config.json').write_text(json.dumps(
        dict(vars(args), sequence_names=[s['name'] for s in sequences]), indent=2))
    tracker = Tracker(model, args.cost_weight)
    fields = ['round', 'sequence', 'frame', 'step', 'loss', 'quality_loss', 'gain_loss',
              'negative_fraction', 'target_centre_inside', 'chosen_modality', 'chosen_action']
    step, frames = 0, 0
    with (run / 'trajectory_steps.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for iteration in range(1, args.rounds + 1):
            for sequence in sequences:
                boxes = np.asarray(sequence['boxes'], dtype=np.float32)
                valid = set(sequence['valid_frames'])
                assert 0 in valid, sequence['name']
                tracker.initialize(read_pair(sequence, 0), boxes[0])
                for frame in range(1, len(boxes)):
                    image = read_pair(sequence, frame)
                    # Capture the actual pre-update state before track() changes it.
                    previous = tracker.state.copy()
                    templates = tracker.templates.clone()
                    history = tracker.history.clone()
                    _, route = tracker.track(image)
                    frames += 1
                    if frame % args.label_stride or frame not in valid:
                        continue
                    search, _, geometry = crop_pair(image, previous, 4., 256)
                    search = search[None].cuda()
                    left, top, side = geometry
                    target = search.new_tensor(np.concatenate(
                        [(boxes[frame, :2] - [left, top]) / side, boxes[frame, 2:] / side]))[None]
                    with torch.no_grad():
                        _, primaries, candidates = labels(model, templates, search, history,
                                                          tracker.output_window)
                        quality_targets, gain_targets = [], []
                        for modality in range(2):
                            main_loss, _ = localization_loss(primaries[modality]['box'], target)
                            quality_targets.append(-main_loss)
                            indices, candidate_boxes = candidates[modality]
                            actual = []
                            for action in range(1, 5):
                                refined = model.correct(
                                    templates, search, primaries[modality], modality, action,
                                    indices[:, action - 1], candidate_boxes[:, action - 1],
                                    tracker.output_window)
                                action_loss, _ = localization_loss(refined['box'], target)
                                actual.append(main_loss - action_loss)
                            gain_targets.append(torch.stack(actual, 1))
                    preview = model.preview_state(templates, search, history)
                    quality_loss = F.smooth_l1_loss(preview['quality'], torch.stack(quality_targets, 1))
                    gain_loss = search.new_zeros(())
                    for modality in range(2):
                        indices, candidate_boxes = candidates[modality]
                        predictions = model.gain_predictions(
                            primaries[modality], preview, modality, indices, candidate_boxes)[:, 1:]
                        gain_loss = gain_loss + F.smooth_l1_loss(predictions, gain_targets[modality]) / 2
                    loss = quality_loss + gain_loss
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.sanity:
                        for module in (model.quality, model.gain):
                            assert any(p.grad is not None and p.grad.abs().sum() > 0
                                       for p in module.parameters())
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                    assert torch.isfinite(norm)
                    optimizer.step()
                    centre = target[:, :2] + target[:, 2:] / 2
                    row = dict(round=iteration, sequence=sequence['name'], frame=frame, step=step,
                               loss=loss.item(), quality_loss=quality_loss.item(), gain_loss=gain_loss.item(),
                               negative_fraction=torch.cat(gain_targets, 1).lt(0).float().mean().item(),
                               target_centre_inside=bool(((centre >= 0) & (centre <= 1)).all().item()),
                               chosen_modality=route['modality'], chosen_action=route['action'])
                    writer.writerow(row)
                    if step % 25 == 0:
                        handle.flush()
                        print(json.dumps(row), flush=True)
                    step += 1
                    if args.sanity and step == 2:
                        frozen = [key for key in before if not key.startswith(('quality.', 'gain.'))]
                        after = model.state_dict()
                        assert all(torch.equal(after[key].cpu(), before[key]) for key in frozen)
                        for prefix in ('quality.', 'gain.'):
                            assert any(not torch.equal(after[key].cpu(), before[key])
                                       for key in before if key.startswith(prefix))
                        expected = model.predict(templates, search, history, args.cost_weight,
                                                 tracker.output_window)
                        torch.save({'net': after, 'step': step, 'config': vars(args)}, run / 'latest.pth')
                        clone = DualRGBT().cuda().eval()
                        clone.load_state_dict(torch.load(run / 'latest.pth', map_location='cpu')['net'], strict=True)
                        reloaded = clone.predict(templates, search, history, args.cost_weight,
                                                 tracker.output_window)
                        assert expected['modality'] == reloaded['modality']
                        assert expected['action'] == reloaded['action']
                        assert torch.equal(expected['box'], reloaded['box'])
                        assert torch.equal(expected['gains'], reloaded['gains'])
                        report = dict(status='PASS', steps=step, tracked_frames=frames,
                                      frozen_state_tensors=len(frozen), saved_state_reload_bit_equal=True,
                                      physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                                      scope='Two real train-trajectory updates only; not complete training or metrics.')
                        (run / 'SANITY.json').write_text(json.dumps(report, indent=2))
                        print(json.dumps(report), flush=True)
                        return
                handle.flush()
                print('SEQUENCE_COMPLETE', iteration, sequence['name'], step, flush=True)
            torch.save({'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'round': iteration, 'step': step, 'config': vars(args)}, run / 'latest.pth')
        (run / 'COMPLETE.json').write_text(json.dumps(
            {'rounds': args.rounds, 'sequences_per_round': len(sequences), 'steps': step,
             'tracked_frames': frames, 'scope': 'Training trajectories only; no formal test accuracy.'}))


if __name__ == '__main__':
    main()
