"""Frozen train-state action comparisons; GT-centred controls are not deployable."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt.data import crop_pair, read_pair
from dual_rgbt_evidence.readout import pixel_xywh
from dual_rgbt_evidence.regions import crop_region, execute_boxes, pixel_iou, search_geometry
from dual_rgbt_global.policy import ACTION_NAMES, build_policy, global_geometry
from dual_rgbt_global.tracker import EvidenceTracker
from dual_rgbt_global.training import counterfactuals, query_pixels
from train_global_evidence_policy import cpu_state, probe_prediction
from lib.test.utils.hann import hann2d


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PrimaryOnlyTracker(EvidenceTracker):
    """Actual own-state rollout with the existing cheap scheduler and forced STOP."""
    @torch.no_grad()
    def track(self, image):
        previous = self.state.copy()
        means = image.mean((0, 1))
        geometry, broad = search_geometry(previous), global_geometry(image)
        main_preview = crop_region(image, geometry, 64, means)[None].cuda()
        broad_preview = crop_region(image, broad, 256, means)[None].cuda()
        preview = self.model.preview_state(self.templates, main_preview, broad_preview, self.history)
        modality = int(preview['quality'].argmax(1).item())
        main = crop_region(image, geometry, 256, means)[None].cuda()
        main_geo = main.new_tensor(geometry)[None]
        sizes = main.new_tensor([[image.shape[0], image.shape[1]]])
        primary = self.model.executor.primary(self.templates, main, modality, self.window)
        self.state = execute_boxes(pixel_xywh(primary['box'], main_geo), sizes)[0].cpu().numpy()
        confidence = primary['score'].max().item()
        self.frame += 1
        if self.frame % 5 == 0 and confidence >= .65:
            updated = crop_pair(image, self.state, 2., 128)[0].cuda()
            self.templates[:, 1, modality * 3:(modality + 1) * 3] = updated[modality * 3:(modality + 1) * 3]
        self.since_query += 1
        movement = ((self.state[:2] + self.state[2:] / 2) -
                    (previous[:2] + previous[2:] / 2)) / previous[2:]
        scale = np.log(self.state[2:] / previous[2:])
        values = [float(modality == 0), float(modality == 1), confidence] + movement.tolist() + scale.tolist() + [self.since_query / 100.]
        self.history = self.history.new_tensor(values)[None]
        return self.state.copy(), modality


def region_coverage(target, geometry):
    centre = target[:2] + target[2:] / 2
    inside = bool(((centre >= geometry[:2]) & (centre < geometry[:2] + geometry[2])).all())
    overlap = np.maximum(0, np.minimum(target[:2] + target[2:], geometry[:2] + geometry[2]) -
                         np.maximum(target[:2], geometry[:2])).prod()
    return inside, float(overlap / target[2:].prod())


def summarize(rows):
    result = {}
    for name in sorted(set(row['action'] for row in rows)):
        group = [row for row in rows if row['action'] == name]
        iou = np.asarray([row['iou'] for row in group])
        gains = np.asarray([row['delta'] for row in group])
        result[name] = dict(rows=len(group), mean_iou=float(iou.mean()), mean_delta=float(gains.mean()),
                            positive=int((gains > 0).sum()), negative=int((gains < 0).sum()),
                            zero=int((gains == 0).sum()),
                            centre_inside=int(sum(row['centre_inside'] for row in group)),
                            mean_gt_coverage=float(np.mean([row['gt_coverage'] for row in group])))
    return result


@torch.no_grad()
def diagnose(model, plan, run, sanity):
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    by_name = {row['name']: row for row in manifest['train']}
    names = plan['sequence_names'][:1] if sanity else plan['sequence_names']
    tracker = PrimaryOnlyTracker(model, 0.)
    rows, states, frames = [], 0, 0
    rng = np.random.RandomState(plan['seed'])
    fields = ('sequence', 'frame', 'state', 'reference_modality', 'modality', 'action', 'auxiliary_tokens',
              'iou', 'main_iou', 'delta', 'centre_inside', 'gt_coverage', 'predicted_delta', 'executed_box')
    with (run / 'action_comparisons.csv').open('w', newline='') as handle:
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
                actual, selected = tracker.track(image)
                frames += 1
                if frame % plan['label_stride'] or frame not in valid:
                    continue
                means = image.mean((0, 1))
                geometry, broad = search_geometry(previous), global_geometry(image)
                main = crop_region(image, geometry, 256, means)[None].cuda()
                main_preview = crop_region(image, geometry, 64, means)[None].cuda()
                broad_preview = crop_region(image, broad, 256, means)[None].cuda()
                main_geo, broad_geo = main.new_tensor(geometry)[None], main.new_tensor(broad)[None]
                sizes, target = main.new_tensor([[image.shape[0], image.shape[1]]]), main.new_tensor(boxes[frame])[None]
                targets = counterfactuals(model, templates, main, main_preview, broad_preview, history,
                                         main_geo, broad_geo, target, sizes, [image], [means], tracker.window)
                assert np.array_equal(targets['executed'][selected][0, 0].cpu().numpy(), actual)
                preview = model.preview_state(templates, main_preview, broad_preview, history)
                random_centre = rng.uniform([0., 0.], [image.shape[1], image.shape[0]])
                gt_centre = boxes[frame, :2] + boxes[frame, 2:] / 2
                local_side = float(geometry[2]) / 2
                controls = [('DENSE_SAME_SEARCH', geometry, 4),
                            ('RANDOM_LOCAL', np.r_[random_centre - local_side / 2, local_side].astype(np.float32), 1),
                            ('GT_CENTER_LOCAL', np.r_[gt_centre - local_side / 2, local_side].astype(np.float32), 1),
                            ('GT_CENTER_WIDE', np.r_[gt_centre - geometry[2] / 2, geometry[2]].astype(np.float32), 4)]
                for modality in range(2):
                    primary = targets['primaries'][modality]
                    predicted = model.gain_predictions(primary, preview, modality, main_geo, broad_geo,
                                                       targets['queries'][modality])[0]
                    values = []
                    for action, action_name in enumerate(ACTION_NAMES):
                        region = geometry if action == 0 else targets['queries'][modality][0, action - 1].cpu().numpy()
                        box = targets['executed'][modality][:, action]
                        tokens = 0 if action == 0 else (385 if action == 4 else 193)
                        values.append((action_name, region, box, tokens, float(predicted[action].item())))
                    if modality == selected:
                        for action_name, region, input_action in controls:
                            geo = main.new_tensor(region)[None]
                            pixels = query_pixels([image], [means], geo, input_action)
                            output = model.query(templates, pixels, primary, main_geo, geo, modality)
                            box = execute_boxes(output['pixel_box'], sizes)
                            values.append((action_name, region, box, output['auxiliary_tokens'], ''))
                    for action_name, region, box, tokens, predicted_delta in values:
                        value = float(pixel_iou(box, target).item())
                        main_iou = float(targets['quality'][0, modality].item())
                        inside, coverage = region_coverage(boxes[frame], region)
                        row = dict(sequence=name, frame=frame, state=states, reference_modality=selected,
                                   modality=modality, action=action_name, auxiliary_tokens=tokens,
                                   iou=value, main_iou=main_iou, delta=value - main_iou,
                                   centre_inside=inside, gt_coverage=coverage, predicted_delta=predicted_delta,
                                   executed_box=json.dumps(box[0].cpu().tolist()))
                        assert np.isfinite([value, main_iou, row['delta'], coverage]).all()
                        writer.writerow(row)
                        rows.append(row)
                states += 1
                if states == 1 or states % 25 == 0:
                    handle.flush()
                    print(json.dumps(dict(tracked_frames=frames, labelled_states=states, comparisons=len(rows))), flush=True)
                probe = dict(templates=templates, main=main, main_preview=main_preview, broad_preview=broad_preview,
                             main_geometry=main_geo, broad_geometry=broad_geo, image_size=sizes, history=history,
                             images=[image], means=[means])
    assert states > 0 and len(rows) == states * 14
    return dict(sequences=len(names), tracked_frames=frames, labelled_states=states,
                comparisons=len(rows), per_action=summarize(rows)), probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stage', choices=('a', 'b'), required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--sanity', action='store_true')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1' and os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    plan = json.loads((ROOT / args.plan).read_text())
    training = json.loads((ROOT / plan['training_plan']).read_text())
    assert sha(ROOT / plan['training_plan']) == plan['training_plan_sha256']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert sha(ROOT / 'data/manifest.json') == training['manifest_sha256']
    assert sha(ROOT / training['input_checkpoint']) == training['input_checkpoint_sha256']
    protected = Path('/home/gaob/CADTrack_ep0014.pth.tar')
    protected_sha = sha(protected)
    assert protected_sha == plan['protected_checkpoint_sha256']
    key = ('sanity_' if args.sanity else 'formal_') + args.stage
    assert args.run == plan['runs'][key]
    checkpoint = ROOT / 'runs' / training['runs']['formal_' + args.stage] / 'latest.pth'
    assert sha(checkpoint) == args.checkpoint_sha256
    task = json.loads((ROOT / 'runs/pipeline_tasks' / ('train_' + training['runs']['formal_' + args.stage]) / 'status.json').read_text())
    assert task['state'] == 'COMPLETE' and task['exit_code'] == 0
    complete = json.loads((checkpoint.parent / 'COMPLETE.json').read_text())
    assert complete['updates'] == 6250 and complete['samples'] == 50000 and not complete['sanity']
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    state = torch.load(checkpoint, map_location='cpu')
    assert state['step'] == 6250 and state['stage'] == args.stage and not state['config']['sanity']
    assert state['config']['plan_sha256'] == plan['training_plan_sha256']
    model = build_policy(ROOT / training['input_checkpoint']).cuda().eval()
    model.load_state_dict(state['net'], strict=True)
    del state
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = cpu_state(model)
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), source_sha256=plan['source_sha256'], plan_sha256=sha(ROOT / args.plan),
                  initial_policy=str(checkpoint), input_checkpoint_sha256=args.checkpoint_sha256,
                  reference='Cheap scheduler with forced STOP; own state and selected-template updates.',
                  scope='Frozen train20 same-state comparisons; GT-centred controls diagnostic only; not toolkit metrics or trajectory upper bound.')
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    result, probe = diagnose(model, plan, run, args.sanity)
    after = cpu_state(model)
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(parameter.grad is None for parameter in model.parameters())
    if args.sanity:
        window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
        expected = probe_prediction(model, probe, window)
        clone = build_policy(ROOT / training['input_checkpoint']).cuda().eval()
        clone.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
        actual = probe_prediction(clone, probe, window)
        assert all(torch.equal(expected[name], actual[name]) for name in expected)
        (run / 'SANITY.json').write_text(json.dumps(dict(status='PASS', frozen_state_tensors=len(before),
            strict_reload_probe_bit_equal=True, probe_fields=list(expected), optimizer_updates=0, backward_calls=0), indent=2))
    else:
        assert result['tracked_frames'] == plan['expected_tracked_frames']
        assert result['labelled_states'] == plan['expected_labelled_states']
    assert sha(checkpoint) == args.checkpoint_sha256 and sha(protected) == protected_sha
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    result.update(state='COMPLETE_DIAGNOSTIC', frozen_state_tensors=len(before), optimizer_updates=0,
                  backward_calls=0, checkpoint_sha256=args.checkpoint_sha256,
                  raw_csv_sha256=sha(run / 'action_comparisons.csv'), sanity=args.sanity,
                  scope=config['scope'])
    (run / 'COMPLETE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
