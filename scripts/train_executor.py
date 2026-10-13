import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from dual_rgbt.data import LasHeRPairs
from dual_rgbt.model import DualRGBT
from dual_rgbt.losses import executor_loss, localization_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--samples', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint', default='/home/gaob/CADTrack_ep0014.pth.tar')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    run = root / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    (run / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    model = DualRGBT(args.checkpoint).cuda()
    # Stage A trains executors, not the two policy heads. Frozen preview selects
    # a candidate deterministically for coverage training; stage B learns it.
    for module in (model.preview, model.quality, model.gain):
        module.requires_grad_(False)
    groups = [{'params': model.backbone.parameters(), 'lr': args.lr * .1},
              {'params': list(model.decode.parameters()) + list(model.box_head.parameters()), 'lr': args.lr}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    fieldnames = ['epoch', 'step', 'modality', 'action', 'loss', 'iou', 'seconds',
                  'primary_tokens', 'auxiliary_tokens']
    with (run / 'steps.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        step = 0
        best_loss = float('inf')
        for epoch in range(1, args.epochs + 1):
            dataset = LasHeRPairs(root / 'data/manifest.json', 'train', args.samples,
                                 args.seed + (epoch - 1) * args.samples)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True, drop_last=True)
            model.train()
            model.preview.eval()
            # Match the original CADTrack FIX_BN training convention.
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
            for batch in loader:
                start = time.perf_counter()
                templates, search, target = [batch[key].cuda(non_blocking=True)
                                              for key in ('templates', 'search', 'target')]
                modality, action = step % 2, (step // 2) % 5
                output = model.forward_action(templates, search, modality, action)
                loss = executor_loss(output, target)
                assert torch.isfinite(loss), 'Non-finite loss'
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], .1)
                assert torch.isfinite(norm), 'Non-finite gradient'
                optimizer.step()
                _, iou = localization_loss(output['box'], target)
                row = dict(epoch=epoch, step=step, modality=modality, action=action,
                           loss=float(loss.item()), iou=float(iou.mean().item()),
                           seconds=time.perf_counter() - start,
                           primary_tokens=output['primary']['primary_tokens'],
                           auxiliary_tokens=output['auxiliary_tokens'])
                writer.writerow(row)
                if step % 50 == 0:
                    handle.flush()
                    print(json.dumps(row), flush=True)
                step += 1
            model.eval()
            dev = LasHeRPairs(root / 'data/manifest.json', 'dev', 100, args.seed)
            summaries = []
            sample_rows = []
            with torch.no_grad():
                for modality in range(2):
                    for action in range(5):
                        losses, ious = [], []
                        for batch in DataLoader(dev, batch_size=args.batch_size, num_workers=args.workers):
                            templates, search, target = [batch[key].cuda() for key in
                                                          ('templates', 'search', 'target')]
                            output = model.forward_action(templates, search, modality, action)
                            values, iou = localization_loss(output['box'], target)
                            offset = len(losses)
                            coverage = [None] * len(values)
                            if action:
                                region = output['region_indices']
                                query_lt = torch.stack([region[:, 0] % 16, region[:, 0] // 16], 1) / 16
                                query_rb = torch.stack([region[:, -1] % 16 + 1,
                                                        region[:, -1] // 16 + 1], 1) / 16
                                intersection = (torch.minimum(query_rb, target[:, :2] + target[:, 2:]) -
                                                torch.maximum(query_lt, target[:, :2])).clamp(min=0).prod(1)
                                coverage = (intersection / target[:, 2:].prod(1)).cpu().tolist()
                            for index, (value, overlap) in enumerate(zip(values.cpu().tolist(), iou.cpu().tolist())):
                                sample_rows.append({'sample_id': offset + index, 'modality': modality,
                                                    'action': action, 'sequence': batch['sequence'][index],
                                                    'frame': int(batch['frame'][index]),
                                                    'localization_loss': value, 'sample_iou': overlap,
                                                    'gt_coverage': coverage[index]})
                            losses.extend(values.cpu().tolist())
                            ious.extend(iou.cpu().tolist())
                        summaries.append({'modality': modality, 'action': action,
                                          'localization_loss': float(np.mean(losses)),
                                          'sample_iou': float(np.mean(ious)), 'samples': len(losses)})
            (run / 'dev_epoch{:02d}.json'.format(epoch)).write_text(json.dumps(summaries, indent=2))
            with (run / 'dev_samples_epoch{:02d}.csv'.format(epoch)).open('w', newline='') as output_file:
                sample_writer = csv.DictWriter(output_file, fieldnames=list(sample_rows[0]))
                sample_writer.writeheader()
                sample_writer.writerows(sample_rows)
            utility = []
            for modality in range(2):
                initial = [r for r in sample_rows if r['modality'] == modality and r['action'] == 0]
                for action in range(1, 5):
                    added = [r for r in sample_rows if r['modality'] == modality and r['action'] == action]
                    gains = np.array([p['localization_loss'] - a['localization_loss']
                                      for p, a in zip(initial, added)])
                    utility.append({'modality': modality, 'action': action, 'samples': len(gains),
                                    'mean_gain': float(gains.mean()),
                                    'gain_quantiles': np.quantile(gains, [0, .1, .5, .9, 1]).tolist(),
                                    'negative_fraction': float((gains < 0).mean()),
                                    'mean_gt_coverage': float(np.mean([a['gt_coverage'] for a in added])),
                                    'corrected_iou50': sum(p['sample_iou'] < .5 <= a['sample_iou']
                                                           for p, a in zip(initial, added)),
                                    'harmed_iou50': sum(a['sample_iou'] < .5 <= p['sample_iou']
                                                        for p, a in zip(initial, added))})
            (run / 'utility_epoch{:02d}.json'.format(epoch)).write_text(json.dumps(utility, indent=2))
            torch.save({'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'step': step, 'config': vars(args)}, run / 'latest.pth')
            dev_loss = float(np.mean([r['localization_loss'] for r in summaries]))
            if dev_loss < best_loss:
                best_loss = dev_loss
                torch.save({'net': model.state_dict(), 'epoch': epoch, 'step': step,
                            'config': vars(args)}, run / 'best.pth')
                (run / 'best.json').write_text(json.dumps(
                    {'epoch': epoch, 'step': step, 'dev_loss': dev_loss,
                     'criterion': 'Mean localization loss across the same 10 dev-crop paths.'}, indent=2))
            print('EPOCH_COMPLETE', epoch, json.dumps(summaries), flush=True)
        (run / 'COMPLETE.json').write_text(json.dumps({'epochs': args.epochs, 'steps': step}))


if __name__ == '__main__':
    main()
