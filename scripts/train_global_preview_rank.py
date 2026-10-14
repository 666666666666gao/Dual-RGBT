"""Fit the existing global preview ranking; frozen executor/controllers stay unchanged."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt.data import crop_pair, read_pair
from dual_rgbt_evidence.regions import crop_region
from dual_rgbt_global.policy import build_policy, global_geometry
from dual_rgbt_global.training import EvidencePairs, collate_pairs, heatmap_target
from train_global_evidence_policy import (cpu_state, cuda_batch, probe_inputs,
                                          probe_prediction, step_optimizer)
from lib.test.utils.hann import hann2d


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_cells(geometry, sizes):
    cells = torch.arange(1024, device=geometry.device)
    normalized = (torch.stack([cells % 32, cells // 32], 1).to(geometry.dtype) + .5) / 32
    coordinates = normalized[None] * geometry[:, None, 2:3] + geometry[:, None, :2]
    valid = ((coordinates >= 0).all(2) &
             (coordinates[:, :, 0] < sizes[:, None, 1]) &
             (coordinates[:, :, 1] < sizes[:, None, 0]))
    return coordinates, valid


def rank_loss(maps, target, geometry, sizes):
    """Unit-mass Gaussian supervision of relative location, excluding padded cells."""
    _, valid = valid_cells(geometry, sizes)
    truth, visible = heatmap_target(target, geometry, 32)
    centre = target[:, :2] + target[:, 2:] / 2
    visible = visible & (centre >= 0).all(1) & (centre[:, 0] < sizes[:, 1]) & (centre[:, 1] < sizes[:, 0])
    weights = truth.flatten(1) * valid
    weights = weights / weights.sum(1, keepdim=True).clamp(min=1e-12)
    losses = []
    for scores in maps:
        logits = (10 * scores.flatten(1)).masked_fill(~valid, torch.finfo(scores.dtype).min)
        per_sample = -(weights * F.log_softmax(logits, dim=1)).sum(1)
        losses.append((per_sample * visible).sum() / visible.sum().clamp(min=1))
    return torch.stack(losses).mean()


@torch.no_grad()
def inspect_ranking(model, plan, run, label, sanity):
    """Use exactly the old first-template/whole-image map and fixed old role/ROI budget."""
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    by_name = {row['name']: row for row in manifest['train']}
    with (ROOT / plan['diagnostic_csv']).open(newline='') as handle:
        records = [r for r in csv.DictReader(handle) if r['region'] == 'GLOBAL_WIDE' and r['content'] == 'TRUE']
    assert len(records) == 1752
    if sanity:
        records = records[:2]
    descriptors, rows = {}, []
    for row in records:
        name, frame, modality = row['sequence'], int(row['frame']), 1 - int(row['modality'])
        sequence = by_name[name]
        if name not in descriptors:
            template = crop_pair(read_pair(sequence, 0), np.asarray(sequence['boxes'][0], dtype=np.float32), 2., 128)[0][None].cuda()
            descriptors[name] = [model.executor.preview(template[:, m * 3:(m + 1) * 3])[:, :, 2:6, 2:6].mean((2, 3)) for m in range(2)]
        image = read_pair(sequence, frame)
        geometry = global_geometry(image)
        pixels = crop_region(image, geometry, 256, image.mean((0, 1)))[None].cuda()
        feature = model.executor.preview.net(pixels[:, modality * 3:(modality + 1) * 3])
        score = F.cosine_similarity(feature, descriptors[name][modality][:, :, None, None], dim=1).flatten(1)
        geo = pixels.new_tensor(geometry)[None]
        sizes = pixels.new_tensor([image.shape[:2]])
        coordinates, valid = valid_cells(geo, sizes)
        scores = score.masked_fill(~valid, -torch.inf)
        index = scores.argmax(1)
        point = coordinates[0, index.item()].cpu().numpy()
        original_geo = np.asarray(json.loads(row['query_geometry']), dtype=np.float32)
        if label == 'before':
            original_point = original_geo[:2] + original_geo[2] / 2
            assert np.max(np.abs(point - original_point)) < 1e-3, (name, frame)
        target = np.asarray(sequence['boxes'][frame], dtype=np.float32)
        centre = target[:2] + target[2:] / 2
        nearest = ((coordinates[0] - coordinates.new_tensor(centre)).square().sum(1)).masked_fill(~valid[0], torch.inf).argmin()
        rank = int((scores[0] > scores[0, nearest]).sum().item())
        value = dict(sequence=name, frame=frame, state=int(row['state']), auxiliary_modality=modality,
                     native_main_outside=row['main_centre_inside'] == 'False',
                     target_nearest_cell_rank=rank, peak=point.tolist(),
                     score_margin=float((scores[0, nearest] - scores[0, index.item()]).item()))
        for region, side in (('local', original_geo[2] / 2), ('wide', original_geo[2])):
            left = point - side / 2
            value[region + '_centre_inside'] = bool(((centre >= left) & (centre < left + side)).all())
            value[region + '_coverage'] = float(np.maximum(0, np.minimum(target[:2] + target[2:], left + side) - np.maximum(target[:2], left)).prod() / target[2:].prod())
        rows.append(value)
    # Keep frame locations private; publish only aggregate summaries.
    private_path = run / ('ranking_' + label + '.json')
    private_path.write_text(json.dumps(rows))
    groups = {}
    for scope, selected in (('all', rows), ('native_main_outside', [r for r in rows if r['native_main_outside']])):
        groups[scope] = dict(states=len(selected))
        if selected:
            groups[scope].update(mean_nearest_cell_rank=float(np.mean([r['target_nearest_cell_rank'] for r in selected])),
                                median_nearest_cell_rank=float(np.median([r['target_nearest_cell_rank'] for r in selected])),
                                mean_score_margin=float(np.mean([r['score_margin'] for r in selected])))
            for region in ('local', 'wide'):
                groups[scope][region] = dict(centre_inside=sum(r[region + '_centre_inside'] for r in selected),
                                            mean_gt_coverage=float(np.mean([r[region + '_coverage'] for r in selected])))
    return dict(groups=groups, private_sha256=sha(private_path),
                before_coordinates_reproduce_original=label == 'before',
                scope='Fixed train states, original first-template/auxiliary-role/ROI-side; no new trajectory, PR/NPR/SR or localization IoU.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--sanity', action='store_true')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1' and os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    plan = json.loads((ROOT / args.plan).read_text())
    review = json.loads((ROOT / plan['source_review']).read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / name) == expected for name, expected in plan['source_sha256'].items())
    assert sha(ROOT / 'data/manifest.json') == plan['manifest_sha256']
    assert sha(ROOT / plan['input_checkpoint']) == plan['input_checkpoint_sha256']
    assert sha(ROOT / plan['factory_checkpoint']) == plan['factory_checkpoint_sha256']
    assert sha(ROOT / plan['diagnostic_csv']) == plan['diagnostic_csv_sha256']
    protected = Path('/home/gaob/CADTrack_ep0014.pth.tar')
    assert sha(protected) == plan['protected_checkpoint_sha256']
    prior = json.loads((ROOT / plan['input_completion']).read_text())
    native = json.loads((ROOT / plan['input_native_status']).read_text())
    assert prior['stage'] == 'b' and not prior['sanity'] and prior['updates'] == 6250
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    tag = 'sanity' if args.sanity else 'formal'
    assert args.run == plan['runs'][tag]
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_policy(ROOT / plan['factory_checkpoint']).cuda().eval()
    state = torch.load(ROOT / plan['input_checkpoint'], map_location='cpu')
    assert state['stage'] == 'b' and state['step'] == 6250 and not state['config']['sanity']
    model.load_state_dict(state['net'], strict=True)
    del state
    prefixes = ('executor.preview.',)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    before = cpu_state(model)
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), plan_sha256=sha(ROOT / args.plan), input_checkpoint_sha256=plan['input_checkpoint_sha256'],
                  source_sha256=plan['source_sha256'], trainable_prefixes=prefixes, scope=plan['scope'])
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    before_ranking = inspect_ranking(model, plan, run, 'before', args.sanity)
    epochs = 1 if args.sanity else plan['epochs']
    samples = 16 if args.sanity else plan['samples_per_epoch']
    step, seen = 0, set()
    with (run / 'steps.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('epoch', 'step', 'loss'))
        writer.writeheader()
        for epoch in range(epochs):
            dataset = EvidencePairs(ROOT / 'data/manifest.json', samples, plan['seed'] + epoch * plan['samples_per_epoch'])
            loader = DataLoader(dataset, batch_size=8, num_workers=4, collate_fn=collate_pairs)
            for batch in loader:
                seen.update(batch['sequences'])
                batch = cuda_batch(batch)
                history = batch['main'].new_zeros(batch['main'].shape[0], 8)
                preview = model.preview_state(batch['templates'], batch['main_preview'], batch['broad_preview'], history)
                loss = rank_loss(preview['broad_maps'], batch['target'], batch['broad_geometry'], batch['image_size'])
                step_optimizer(loss, model, optimizer, prefixes, args.sanity)
                step += 1
                writer.writerow(dict(epoch=epoch + 1, step=step, loss=loss.item()))
                if step == 1 or step % 25 == 0:
                    handle.flush()
                    print(json.dumps(dict(epoch=epoch + 1, step=step, loss=loss.item())), flush=True)
                probe = probe_inputs(batch, history)
            torch.save(dict(net=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                            stage='localize', config=config), run / 'latest.pth')
    expected_updates = 2 if args.sanity else plan['expected_updates']
    assert step == expected_updates
    after = cpu_state(model)
    frozen = [name for name in before if not name.startswith(prefixes)]
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    assert len(frozen) == 259 and all(torch.equal(before[name], after[name]) for name in frozen)
    assert len(changed) == 6 and all(name.startswith(prefixes) for name in changed)
    assert all(torch.isfinite(value).all() for value in after.values())
    assert {int(value['step'].item()) for value in optimizer.state.values()} == {step}
    if not args.sanity:
        assert len(seen) == 882
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    expected = probe_prediction(model, probe, window)
    clone = build_policy(ROOT / plan['factory_checkpoint']).cuda().eval()
    clone.load_state_dict(torch.load(run / 'latest.pth', map_location='cpu')['net'], strict=True)
    actual = probe_prediction(clone, probe, window)
    assert all(torch.equal(expected[name], actual[name]) for name in expected)
    del clone
    after_ranking = inspect_ranking(model, plan, run, 'after', args.sanity)
    result = dict(status='COMPLETE', sanity=args.sanity, updates=step, samples=epochs * samples,
                  sequences_seen=len(seen), frozen_state_tensors=len(frozen), changed_state_tensors=changed,
                  strict_reload_probe_bit_equal=True, probe_fields=list(expected),
                  before_ranking=before_ranking, after_ranking=after_ranking,
                  checkpoint_sha256=sha(run / 'latest.pth'), scope=plan['scope'])
    assert all(sha(ROOT / name) == expected for name, expected in plan['source_sha256'].items())
    assert sha(ROOT / plan['input_checkpoint']) == plan['input_checkpoint_sha256']
    assert sha(ROOT / plan['factory_checkpoint']) == plan['factory_checkpoint_sha256']
    assert sha(ROOT / plan['diagnostic_csv']) == plan['diagnostic_csv_sha256']
    assert sha(protected) == plan['protected_checkpoint_sha256']
    (run / 'COMPLETE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

