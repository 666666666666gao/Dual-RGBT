"""Frozen spatial candidate opportunity; target annotations never choose proposals."""
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


def separated_peaks(scores, coordinates, radius):
    remaining = scores.clone()
    indices = []
    for _ in range(8):
        peak = remaining.argmax().item()
        if not torch.isfinite(remaining[peak]).item():
            break
        indices.append(peak)
        distance = (coordinates - coordinates[peak]).square().sum(1)
        remaining = remaining.masked_fill(distance < radius ** 2, -torch.inf)
    return torch.as_tensor(indices, dtype=torch.long, device=scores.device)


def predictions(model, template, broad, modality, geometry, sizes, local_side):
    descriptor = model.executor.preview(template[:, modality * 3:(modality + 1) * 3])[:, :, 2:6, 2:6].mean((2, 3))
    feature = model.executor.preview.net(broad[:, modality * 3:(modality + 1) * 3])
    coordinates, valid = valid_cells(geometry, sizes)
    scores = F.cosine_similarity(feature, descriptor[:, :, None, None], dim=1).flatten(1)
    scores = scores.masked_fill(~valid, -torch.inf)
    indices = separated_peaks(scores[0], coordinates[0], local_side / 2)
    return dict(scores=scores, indices=indices, points=coordinates[0, indices])


@torch.no_grad()
def diagnose(model, plan, run, sanity):
    sequences = {r['name']: r for r in json.loads((ROOT / 'data/manifest.json').read_text())['train']}
    with (ROOT / plan['content_csv']).open(newline='') as handle:
        original = [r for r in csv.DictReader(handle) if r['region'] == 'GLOBAL_WIDE' and r['content'] == 'TRUE']
    baseline = json.loads((ROOT / plan['baseline_ranking']).read_text())
    assert len(original) == len(baseline) == 1752
    if sanity:
        original, baseline = original[:2], baseline[:2]
    templates, rows, score_maps = {}, [], []
    for old, previous in zip(original, baseline):
        name, frame, modality = old['sequence'], int(old['frame']), 1 - int(old['modality'])
        assert (previous['sequence'], previous['frame'], previous['auxiliary_modality']) == (name, frame, modality)
        sequence = sequences[name]
        if name not in templates:
            templates[name] = crop_pair(read_pair(sequence, 0), np.asarray(sequence['boxes'][0], dtype=np.float32), 2., 128)[0][None].cuda()
        image = read_pair(sequence, frame)
        geometry = global_geometry(image)
        broad = crop_region(image, geometry, 256, image.mean((0, 1)))[None].cuda()
        geo, sizes = broad.new_tensor(geometry)[None], broad.new_tensor([image.shape[:2]])
        old_geometry = np.asarray(json.loads(old['query_geometry']), dtype=np.float32)
        local_side = float(old_geometry[2] / 2)
        # Proposal selection uses only frozen CNN scores, image geometry and old search size.
        predicted = predictions(model, templates[name], broad, modality, geo, sizes, local_side)
        points = predicted['points'].cpu().numpy()
        assert np.max(np.abs(points[0] - np.asarray(previous['peak'], dtype=np.float32))) < 1e-3
        score_maps.append(predicted['scores'][0].cpu().numpy())
        row = dict(state=int(old['state']), sequence=name, frame=frame, auxiliary_modality=modality,
                   native_main_outside=previous['native_main_outside'], candidate_count=len(points),
                   indices=predicted['indices'].cpu().tolist(), points=points.tolist(), by_k={})
        assert row['native_main_outside'] == (old['main_centre_inside'] == 'False')
        # GT appears only after proposal collection, to diagnose the available action set.
        target = np.asarray(sequence['boxes'][frame], dtype=np.float32)
        centre = target[:2] + target[2:] / 2
        for k in (1, 2, 4, 8):
            row['by_k'][str(k)] = {}
            for region, side in (('local', local_side), ('wide', float(old_geometry[2]))):
                left = points[:k] - side / 2
                inside = ((centre >= left) & (centre < left + side)).all(1)
                coverage = np.maximum(0, np.minimum(target[:2] + target[2:], left + side) - np.maximum(target[:2], left)).prod(1) / target[2:].prod()
                row['by_k'][str(k)][region] = dict(centre_inside=bool(inside.any()),
                    best_gt_coverage=float(coverage.max()), oracle_index=int(coverage.argmax()))
        rows.append(row)
        probe = (templates[name], broad, modality, geo, sizes, local_side)
    (run / 'candidates.json').write_text(json.dumps(rows))
    np.save(str(run / 'scores.npy'), np.stack(score_maps))
    groups = {}
    for scope in ('all', 'native_main_outside'):
        selected = [r for r in rows if scope == 'all' or r['native_main_outside']]
        group = dict(states=len(selected), candidate_count_histogram={
            str(n): sum(r['candidate_count'] == n for r in selected) for n in range(1, 9)}, by_k={})
        if selected:
            for k in (1, 2, 4, 8):
                group['by_k'][str(k)] = {region: dict(
                    centre_inside=sum(r['by_k'][str(k)][region]['centre_inside'] for r in selected),
                    mean_best_gt_coverage=float(np.mean([r['by_k'][str(k)][region]['best_gt_coverage'] for r in selected])))
                    for region in ('local', 'wide')}
        groups[scope] = group
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
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
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
    groups, probe, states = diagnose(model, plan, run, args.sanity)
    assert states == (2 if args.sanity else 1752)
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
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    assert sha(protected) == plan['protected_checkpoint_sha256']
    result = dict(status='COMPLETE', sanity=args.sanity, states=states, per_scope=groups,
                  frozen_state_tensors=265, optimizer_updates=0, backward_calls=0, baseline_top1_reproduced=True,
                  private_candidates_sha256=sha(run / 'candidates.json'), private_scores_sha256=sha(run / 'scores.npy'),
                  scope=plan['scope'])
    (run / 'COMPLETE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
