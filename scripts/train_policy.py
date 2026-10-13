"""Stage B: learn signed localization utility with a frozen stage-A executor.

Labels are recomputed from the current preview-selected actions on training
images. No test annotations or accuracy evaluation are involved.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from dual_rgbt.data import LasHeRPairs
from dual_rgbt.model import DualRGBT
from dual_rgbt.losses import localization_loss


def labels(model, templates, search):
    history = search.new_zeros(search.shape[0], 8)
    preview = model.preview_state(templates, search, history)
    primaries, candidates = [], []
    for modality in range(2):
        main = model.primary(templates, search, modality)
        indices, boxes = model.candidates(main, preview, modality)
        primaries.append(main)
        candidates.append((indices, boxes))
    return preview, primaries, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executor', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2027)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = DualRGBT().cuda()
    model.load_state_dict(torch.load(args.executor, map_location='cpu')['net'], strict=True)
    for module in (model.backbone, model.decode, model.box_head):
        module.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    run = root / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    (run / 'config.json').write_text(json.dumps(vars(args), indent=2))
    with (run / 'policy_steps.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'step', 'loss', 'quality_loss',
                                                   'gain_loss', 'preview_loss', 'negative_fraction'])
        writer.writeheader()
        step = 0
        for epoch in range(1, args.epochs + 1):
            dataset = LasHeRPairs(root / 'data/manifest.json', 'train', args.samples,
                                 args.seed + (epoch - 1) * args.samples)
            for batch in DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers):
                model.eval()  # Frozen execution/math, preview contains no dropout or BN.
                templates, search, target = [batch[k].cuda() for k in ('templates', 'search', 'target')]
                with torch.no_grad():
                    _, main_predictions, candidates = labels(model, templates, search)
                    quality_targets, gain_targets = [], []
                    for modality in range(2):
                        main_loss, _ = localization_loss(main_predictions[modality]['box'], target)
                        quality_targets.append(-main_loss)
                        actual = []
                        indices, boxes = candidates[modality]
                        for action in range(1, 5):
                            refined = model.correct(templates, search, main_predictions[modality],
                                                    modality, action, indices[:, action - 1],
                                                    boxes[:, action - 1])
                            action_loss, _ = localization_loss(refined['box'], target)
                            actual.append(main_loss - action_loss)
                        gain_targets.append(torch.stack(actual, 1))
                preview = model.preview_state(templates, search, search.new_zeros(search.shape[0], 8))
                quality_loss = F.smooth_l1_loss(preview['quality'], torch.stack(quality_targets, 1))
                gain_loss = search.new_zeros(())
                for modality in range(2):
                    indices, boxes = candidates[modality]
                    predictions = model.gain_predictions(main_predictions[modality], preview,
                                                         modality, indices, boxes)[:, 1:]
                    gain_loss = gain_loss + F.smooth_l1_loss(predictions, gain_targets[modality]) / 2
                centre = target[:, :2] + target[:, 2:] / 2
                # Current bounded training jitter keeps the target centre inside.
                spatial_target = (centre[:, 1] * 8).long() * 8 + (centre[:, 0] * 8).long()
                preview_loss = search.new_zeros(())
                for feature in preview['maps']:
                    preview_loss = preview_loss + F.cross_entropy(
                        10 * feature.flatten(1), spatial_target) / 2
                loss = quality_loss + gain_loss + .1 * preview_loss
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                assert torch.isfinite(norm)
                optimizer.step()
                row = {'epoch': epoch, 'step': step, 'loss': loss.item(),
                       'quality_loss': quality_loss.item(), 'gain_loss': gain_loss.item(),
                       'preview_loss': preview_loss.item(),
                       'negative_fraction': torch.cat(gain_targets, 1).lt(0).float().mean().item()}
                writer.writerow(row)
                if step % 50 == 0:
                    handle.flush()
                    print(json.dumps(row), flush=True)
                step += 1
            torch.save({'net': model.state_dict(), 'epoch': epoch, 'config': vars(args)}, run / 'latest.pth')
        (run / 'COMPLETE.json').write_text(json.dumps({'epochs': args.epochs, 'steps': step}))


if __name__ == '__main__':
    main()
