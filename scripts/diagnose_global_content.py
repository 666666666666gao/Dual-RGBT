"""Fixed-coordinate auxiliary-content controls on a causal train-only rollout."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnose_global_opportunity import PrimaryOnlyTracker, region_coverage, sha
from dual_rgbt.data import read_pair
from dual_rgbt_evidence.readout import pixel_xywh
from dual_rgbt_evidence.regions import crop_region, execute_boxes, pixel_iou, search_geometry
from dual_rgbt_global.policy import ACTION_NAMES, build_policy, global_geometry
from dual_rgbt_global.training import query_pixels
from train_global_evidence_policy import cpu_state, probe_prediction


def content_variants(pixels, modality):
    """Change only auxiliary search pixels; keep coordinates/templates fixed."""
    auxiliary = slice((1 - modality) * 3, (2 - modality) * 3)
    primary = slice(modality * 3, (modality + 1) * 3)
    blank, shifted = pixels.clone(), pixels.clone()
    blank[:, auxiliary] = 0.
    shifted[:, auxiliary] = torch.roll(pixels[:, auxiliary],
                                      shifts=(pixels.shape[-2] // 2, pixels.shape[-1] // 2), dims=(-2, -1))
    assert torch.equal(blank[:, primary], pixels[:, primary])
    assert torch.equal(shifted[:, primary], pixels[:, primary])
    return (('TRUE', pixels), ('ZERO_AUX', blank), ('SHIFT_AUX', shifted))


def aggregate(rows):
    result = {}
    for region in sorted({row['region'] for row in rows}):
        group = [row for row in rows if row['region'] == region]
        contents = {}
        for content in ('TRUE', 'ZERO_AUX', 'SHIFT_AUX'):
            values = [row for row in group if row['content'] == content]
            contents[content] = dict(rows=len(values), mean_iou=float(np.mean([row['iou'] for row in values])),
                                     mean_delta_from_stop=float(np.mean([row['delta_from_stop'] for row in values])))
        paired = {}
        for control in ('ZERO_AUX', 'SHIFT_AUX'):
            for scope in ('all', 'main_centre_outside'):
                true = [row for row in group if row['content'] == 'TRUE' and
                        (scope == 'all' or not row['main_centre_inside'])]
                by_state = {row['state']: row for row in group if row['content'] == control}
                delta = np.asarray([row['iou'] - by_state[row['state']]['iou'] for row in true])
                paired[control + ':' + scope] = dict(rows=len(delta), sum_delta=float(delta.sum()),
                    positive=int((delta > 0).sum()), negative=int((delta < 0).sum()), zero=int((delta == 0).sum()))
        result[region] = dict(contents=contents, true_minus_control=paired)
    return result


@torch.no_grad()
def diagnose(model, plan, run, sanity):
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    by_name = {row['name']: row for row in manifest['train']}
    names = plan['sequence_names'][:1] if sanity else plan['sequence_names']
    tracker = PrimaryOnlyTracker(model, 0.)
    rows, states, frames = [], 0, 0
    fields = ('sequence', 'frame', 'state', 'modality', 'region', 'content', 'auxiliary_tokens',
              'main_centre_inside', 'query_centre_inside', 'query_gt_coverage', 'main_iou',
              'iou', 'delta_from_stop', 'query_geometry', 'executed_box')
    with (run / 'content_comparisons.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in names:
            sequence = by_name[name]
            boxes = np.asarray(sequence['boxes'], dtype=np.float32)
            valid = set(sequence['valid_frames'])
            tracker.initialize(read_pair(sequence, 0), boxes[0])
            stop = min(len(boxes), plan['sanity_frames'] + 1) if sanity else len(boxes)
            for frame in range(1, stop):
                image = read_pair(sequence, frame)
                templates, history, previous = tracker.templates.clone(), tracker.history.clone(), tracker.state.copy()
                actual, modality = tracker.track(image)
                frames += 1
                if frame % plan['label_stride'] or frame not in valid:
                    continue
                means = image.mean((0, 1))
                geometry, broad = search_geometry(previous), global_geometry(image)
                main = crop_region(image, geometry, 256, means)[None].cuda()
                main_preview = crop_region(image, geometry, 64, means)[None].cuda()
                broad_preview = crop_region(image, broad, 256, means)[None].cuda()
                main_geo, broad_geo = main.new_tensor(geometry)[None], main.new_tensor(broad)[None]
                sizes = main.new_tensor([[image.shape[0], image.shape[1]]])
                target = main.new_tensor(boxes[frame])[None]
                preview = model.preview_state(templates, main_preview, broad_preview, history)
                primary = model.executor.primary(templates, main, modality, tracker.window)
                factual = execute_boxes(pixel_xywh(primary['box'], main_geo), sizes)
                assert np.array_equal(factual[0].cpu().numpy(), actual)
                main_iou = float(pixel_iou(factual, target).item())
                main_inside, _ = region_coverage(boxes[frame], geometry)
                queries = model.query_geometries(primary, preview, modality, main_geo, broad_geo, sizes)
                regions = [(ACTION_NAMES[action], queries[:, action - 1], action) for action in range(1, 5)]
                centre = target[:, :2] + target[:, 2:] / 2
                for region_name, side, action in (('GT_CENTER_LOCAL', main_geo[:, 2:3] / 2, 1),
                                                   ('GT_CENTER_WIDE', main_geo[:, 2:3], 4)):
                    regions.append((region_name, torch.cat([centre - side / 2, side], dim=1), action))
                for region_name, query_geo, action in regions:
                    pixels = query_pixels([image], [means], query_geo, action)
                    inside, coverage = region_coverage(boxes[frame], query_geo[0].cpu().numpy())
                    for content, variant in content_variants(pixels, modality):
                        output = model.query(templates, variant, primary, main_geo, query_geo, modality)
                        box = execute_boxes(output['pixel_box'], sizes)
                        value = float(pixel_iou(box, target).item())
                        assert output['auxiliary_tokens'] == (385 if action == 4 else 193)
                        row = dict(sequence=name, frame=frame, state=states, modality=modality,
                            region=region_name, content=content, auxiliary_tokens=output['auxiliary_tokens'],
                            main_centre_inside=main_inside, query_centre_inside=inside, query_gt_coverage=coverage,
                            main_iou=main_iou, iou=value, delta_from_stop=value - main_iou,
                            query_geometry=json.dumps(query_geo[0].cpu().tolist()),
                            executed_box=json.dumps(box[0].cpu().tolist()))
                        assert np.isfinite([value, main_iou, coverage]).all()
                        writer.writerow(row)
                        rows.append(row)
                states += 1
                if states == 1 or states % 25 == 0:
                    handle.flush()
                    print(json.dumps(dict(tracked_frames=frames, labelled_states=states, comparisons=len(rows))), flush=True)
                probe = dict(templates=templates, main=main, main_preview=main_preview, broad_preview=broad_preview,
                    main_geometry=main_geo, broad_geometry=broad_geo, image_size=sizes, history=history,
                    images=[image], means=[means])
    assert states > 0 and len(rows) == states * 18
    return dict(sequences=len(names), tracked_frames=frames, labelled_states=states,
                comparisons=len(rows), per_region=aggregate(rows)), probe, tracker.window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--sanity', action='store_true')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1' and os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    plan = json.loads((ROOT / args.plan).read_text())
    review = json.loads((ROOT / plan['source_review']).read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan) and review['source_sha256'] == plan['source_sha256']
    training = json.loads((ROOT / plan['training_plan']).read_text())
    assert sha(ROOT / plan['training_plan']) == plan['training_plan_sha256']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert sha(ROOT / 'data/manifest.json') == training['manifest_sha256']
    factory = ROOT / training['input_checkpoint']
    assert sha(factory) == training['input_checkpoint_sha256']
    protected = Path('/home/gaob/CADTrack_ep0014.pth.tar')
    assert sha(protected) == plan['protected_checkpoint_sha256']
    assert args.run == plan['runs']['sanity' if args.sanity else 'formal']
    checkpoint = ROOT / 'runs' / training['runs']['formal_b'] / 'latest.pth'
    assert sha(checkpoint) == args.checkpoint_sha256
    native = json.loads((ROOT / 'runs/pipeline_tasks' / ('train_' + training['runs']['formal_b']) / 'status.json').read_text())
    complete = json.loads((checkpoint.parent / 'COMPLETE.json').read_text())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert complete['updates'] == 6250 and complete['samples'] == 50000 and complete['stage'] == 'b' and not complete['sanity']
    state = torch.load(checkpoint, map_location='cpu')
    assert state['step'] == 6250 and state['stage'] == 'b' and not state['config']['sanity']
    assert state['config']['plan_sha256'] == plan['training_plan_sha256']
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_policy(factory).cuda().eval()
    model.load_state_dict(state['net'], strict=True)
    del state
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = cpu_state(model)
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), source_sha256=plan['source_sha256'], plan_sha256=sha(ROOT / args.plan),
        training_plan_sha256=plan['training_plan_sha256'], checkpoint_sha256=args.checkpoint_sha256,
        zero_definition='Zero normalized auxiliary search pixels; auxiliary templates unchanged.',
        shift_definition='Half-width and half-height cyclic roll of auxiliary search pixels; geometry unchanged.',
        scope='Train-only fixed-coordinate content comparisons on a primary-only own-state rollout; GT-centred regions diagnostic only. Not PR/NPR/SR or latency.')
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    result, probe, window = diagnose(model, plan, run, args.sanity)
    after = cpu_state(model)
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(parameter.grad is None for parameter in model.parameters())
    if args.sanity:
        expected = probe_prediction(model, probe, window)
        clone = build_policy(factory).cuda().eval()
        clone.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            actual = probe_prediction(clone, probe, window)
        assert all(torch.equal(expected[name], actual[name]) for name in expected)
        (run / 'SANITY.json').write_text(json.dumps(dict(status='PASS', frozen_state_tensors=len(before),
            strict_reload_probe_bit_equal=True, probe_fields=list(expected), optimizer_updates=0, backward_calls=0), indent=2))
    else:
        assert result['tracked_frames'] == plan['expected_tracked_frames']
        assert result['labelled_states'] == plan['expected_labelled_states']
    assert sha(checkpoint) == args.checkpoint_sha256
    assert sha(factory) == training['input_checkpoint_sha256']
    assert sha(protected) == plan['protected_checkpoint_sha256']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    result.update(frozen_state_tensors=len(before), optimizer_updates=0, backward_calls=0,
        checkpoint_sha256=args.checkpoint_sha256, raw_csv_sha256=sha(run / 'content_comparisons.csv'),
        sanity=args.sanity, scope=config['scope'])
    (run / 'COMPLETE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
