"""Frozen descriptor-scale comparison, with identical global pixels/grid/ROI budgets."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt.data import crop_pair, read_pair
from dual_rgbt_evidence.regions import crop_region
from dual_rgbt_global.policy import build_policy, global_geometry
from train_global_evidence_policy import cpu_state, sha
from train_global_preview_rank import valid_cells


def descriptions(model, template):
    values = {}
    for resolution in (64, 32):
        for modality in range(2):
            pixels = F.interpolate(template[:, modality * 3:(modality + 1) * 3],
                                   (resolution, resolution), mode='bilinear', align_corners=False)
            feature = model.executor.preview.net(pixels)
            side = resolution // 8
            values[resolution, modality] = feature[:, :, side // 4:3 * side // 4,
                                                    side // 4:3 * side // 4].mean((2, 3))
    return values


def predictions(model, template, broad, modality, geometry, sizes):
    desc = descriptions(model, template)
    feature = model.executor.preview.net(broad[:, modality * 3:(modality + 1) * 3])
    coordinates, valid = valid_cells(geometry, sizes)
    result = {}
    for resolution in (64, 32):
        scores = F.cosine_similarity(feature, desc[resolution, modality][:, :, None, None], dim=1).flatten(1)
        scores = scores.masked_fill(~valid, -torch.inf)
        result['scores_' + str(resolution)] = scores
        result['point_' + str(resolution)] = coordinates[0, scores.argmax(1).item()]
    return result


@torch.no_grad()
def diagnose(model, plan, run, sanity):
    sequences = {row['name']: row for row in json.loads((ROOT / 'data/manifest.json').read_text())['train']}
    with (ROOT / plan['content_csv']).open(newline='') as handle:
        original = [row for row in csv.DictReader(handle) if row['region'] == 'GLOBAL_WIDE' and row['content'] == 'TRUE']
    baseline = json.loads((ROOT / plan['baseline_ranking']).read_text())
    assert len(original) == len(baseline) == 1752
    if sanity:
        original, baseline = original[:2], baseline[:2]
    templates, descriptors, rows = {}, {}, []
    for old, previous in zip(original, baseline):
        name, frame, modality = old['sequence'], int(old['frame']), 1 - int(old['modality'])
        sequence = sequences[name]
        assert (previous['sequence'], previous['frame'], previous['auxiliary_modality']) == (name, frame, modality)
        if name not in templates:
            templates[name] = crop_pair(read_pair(sequence, 0), np.asarray(sequence['boxes'][0], dtype=np.float32), 2., 128)[0][None].cuda()
            descriptors[name] = descriptions(model, templates[name])
        image = read_pair(sequence, frame)
        geometry = global_geometry(image)
        broad = crop_region(image, geometry, 256, image.mean((0, 1)))[None].cuda()
        geo, sizes = broad.new_tensor(geometry)[None], broad.new_tensor([image.shape[:2]])
        coordinates, valid = valid_cells(geo, sizes)
        feature = model.executor.preview.net(broad[:, modality * 3:(modality + 1) * 3])
        target = np.asarray(sequence['boxes'][frame], dtype=np.float32)
        centre = target[:2] + target[2:] / 2
        nearest = ((coordinates[0] - coordinates.new_tensor(centre)).square().sum(1)).masked_fill(~valid[0], torch.inf).argmin()
        original_geo = np.asarray(json.loads(old['query_geometry']), dtype=np.float32)
        for resolution in (64, 32):
            scores = F.cosine_similarity(feature, descriptors[name][resolution, modality][:, :, None, None], dim=1).flatten(1).masked_fill(~valid, -torch.inf)
            peak = scores.argmax(1).item()
            point = coordinates[0, peak].cpu().numpy()
            if resolution == 64:
                assert np.max(np.abs(point - np.asarray(previous['peak'], dtype=np.float32))) < 1e-3
            row = dict(sequence=name, frame=frame, state=int(old['state']), resolution=resolution,
                       auxiliary_modality=modality, native_main_outside=previous['native_main_outside'],
                       peak=point.tolist(), target_nearest_cell_rank=int((scores[0] > scores[0, nearest]).sum().item()),
                       score_margin=float((scores[0, nearest] - scores[0, peak]).item()))
            assert row['native_main_outside'] == (old['main_centre_inside'] == 'False')
            for region, side in (('local', original_geo[2] / 2), ('wide', original_geo[2])):
                left = point - side / 2
                row[region + '_centre_inside'] = bool(((centre >= left) & (centre < left + side)).all())
                row[region + '_coverage'] = float(np.maximum(0, np.minimum(target[:2] + target[2:], left + side) - np.maximum(target[:2], left)).prod() / target[2:].prod())
            rows.append(row)
        probe = (templates[name], broad, modality, geo, sizes)
    (run / 'comparisons.json').write_text(json.dumps(rows))
    groups = {}
    for resolution in (64, 32):
        groups[str(resolution)] = {}
        for scope in ('all', 'native_main_outside'):
            selected = [row for row in rows if row['resolution'] == resolution and
                        (scope == 'all' or row['native_main_outside'])]
            group = dict(states=len(selected))
            if selected:
                group.update(mean_nearest_cell_rank=float(np.mean([row['target_nearest_cell_rank'] for row in selected])),
                             mean_score_margin=float(np.mean([row['score_margin'] for row in selected])))
                for region in ('local', 'wide'):
                    group[region] = dict(centre_inside=sum(row[region + '_centre_inside'] for row in selected),
                                         mean_gt_coverage=float(np.mean([row[region + '_coverage'] for row in selected])))
            groups[str(resolution)][scope] = group
    return groups, probe, len(rows)


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
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    for path, expected in plan['input_sha256'].items():
        assert sha(ROOT / path) == expected
    protected = Path('/home/gaob/CADTrack_ep0014.pth.tar')
    assert sha(protected) == plan['protected_checkpoint_sha256']
    native = json.loads((ROOT / plan['input_native_status']).read_text())
    complete = json.loads((ROOT / plan['input_completion']).read_text())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert complete['updates'] == 6250 and complete['sequences_seen'] == 882 and not complete['sanity']
    assert args.run == plan['runs']['sanity' if args.sanity else 'formal']
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_policy(ROOT / plan['factory_checkpoint']).cuda().eval()
    state = torch.load(ROOT / plan['checkpoint'], map_location='cpu')
    assert state['stage'] == 'localize' and state['step'] == 6250 and not state['config']['sanity']
    assert state['config']['plan_sha256'] == plan['warmup_plan_sha256']
    model.load_state_dict(state['net'], strict=True)
    del state
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = cpu_state(model)
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), plan_sha256=sha(ROOT / args.plan), checkpoint_sha256=plan['input_sha256'][plan['checkpoint']],
                  source_sha256=plan['source_sha256'], scope=plan['scope'])
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    groups, probe, comparisons = diagnose(model, plan, run, args.sanity)
    assert comparisons == (4 if args.sanity else 3504)
    assert all(torch.equal(before[name], value) for name, value in cpu_state(model).items())
    assert all(parameter.grad is None for parameter in model.parameters())
    if args.sanity:
        with torch.no_grad():
            expected = predictions(model, *probe)
            clone = build_policy(ROOT / plan['factory_checkpoint']).cuda().eval()
            clone.load_state_dict(torch.load(ROOT / plan['checkpoint'], map_location='cpu')['net'], strict=True)
            actual = predictions(clone, *probe)
        assert all(torch.equal(expected[name], actual[name]) for name in expected)
        (run / 'SANITY.json').write_text(json.dumps(dict(status='PASS', strict_reload_bit_equal=True,
                                                         probe_fields=list(expected), frozen_state_tensors=265), indent=2))
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / path) == expected for path, expected in plan['input_sha256'].items())
    assert sha(protected) == plan['protected_checkpoint_sha256']
    result = dict(status='COMPLETE', sanity=args.sanity, states=comparisons // 2, comparisons=comparisons,
                  per_resolution=groups, frozen_state_tensors=265, optimizer_updates=0, backward_calls=0,
                  baseline64_peaks_reproduced=True, private_comparisons_sha256=sha(run / 'comparisons.json'),
                  scope=plan['scope'])
    (run / 'COMPLETE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

