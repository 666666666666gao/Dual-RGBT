"""Frozen train-trajectory action diagnostics; no optimization or test labels.

Compare the current clipped proxy, uncensored raw-proposal error, and error of
the box actually executed by Tracker. The admitted crop API is used unchanged.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import dual_rgbt.data as data
from profile_crop_api import six_channel_crop
data.crop_pair = six_channel_crop
from dual_rgbt.model import DualRGBT
from dual_rgbt.tracker import Tracker
from dual_rgbt.losses import localization_loss
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, clip_box
from train_policy import labels


def uncensored_loss(prediction, target):
    predicted = box_cxcywh_to_xyxy(prediction)
    truth = box_xywh_to_xyxy(target)
    intersection_lt = torch.maximum(predicted[:, :2], truth[:, :2])
    intersection_rb = torch.minimum(predicted[:, 2:], truth[:, 2:])
    intersection = (intersection_rb - intersection_lt).clamp(min=0).prod(1)
    pred_area = (predicted[:, 2:] - predicted[:, :2]).clamp(min=0).prod(1)
    true_area = (truth[:, 2:] - truth[:, :2]).prod(1)
    union = pred_area + true_area - intersection
    iou = intersection / union.clamp(min=1e-7)
    enclosing = (torch.maximum(predicted[:, 2:], truth[:, 2:]) -
                 torch.minimum(predicted[:, :2], truth[:, :2])).prod(1)
    giou = iou - (enclosing - union) / enclosing.clamp(min=1e-7)
    return 2 * (1 - giou) + 5 * (predicted - truth).abs().mean(1), iou


def execute_box(box, geometry, image_shape):
    left, top, side = geometry
    xywh = np.concatenate([box[:2] * side + [left, top] - box[2:] * side / 2,
                           box[2:] * side])
    return np.asarray(clip_box(xywh.tolist(), image_shape[0], image_shape[1], margin=10), dtype=np.float32)


def full_image_iou(prediction, target):
    prediction, target = prediction.astype(np.float64), target.astype(np.float64)
    intersection = np.maximum(0, np.minimum(prediction[:2] + prediction[2:], target[:2] + target[2:]) -
                              np.maximum(prediction[:2], target[:2])).prod()
    return float(intersection / (prediction[2:].prod() + target[2:].prod() - intersection))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--sanity', action='store_true')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    plan = json.loads((root / args.plan).read_text(encoding='utf-8'))
    manifest_path = root / 'data/manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == plan['manifest_sha256']
    assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
               for name, expected in plan['source_sha256'].items())
    checkpoint = root / plan['input_checkpoint']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == plan['input_checkpoint_sha256']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    order = np.random.RandomState(plan['seed']).permutation(len(manifest['train']))[:plan['sequences']]
    sequences = [manifest['train'][int(index)] for index in order]
    assert [sequence['name'] for sequence in sequences] == plan['sequence_names']
    assert not set(plan['sequence_names']) & ({s['name'] for s in manifest['dev']} | set(manifest['test_sequences']))
    expected_frames = sum(len(s['boxes']) - 1 for s in sequences)
    expected_labels = sum(sum(frame > 0 and frame % plan['label_stride'] == 0 for frame in s['valid_frames'])
                          for s in sequences)
    assert expected_frames == plan['expected_tracked_frames'] and expected_labels == plan['expected_labeled_states']
    run = root / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    (run / 'config.json').write_text(json.dumps(dict(plan=plan, sanity=args.sanity), indent=2))
    torch.manual_seed(plan['seed'])
    torch.cuda.manual_seed_all(plan['seed'])
    model = DualRGBT().cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
    model.requires_grad_(False)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    tracker = Tracker(model, plan['cost_weight'])
    fields = ['sequence', 'frame', 'modality', 'action', 'target_centre_inside', 'target_corners_inside',
              'chosen_modality', 'chosen_action', 'predicted_gain', 'clipped_main_loss', 'clipped_action_loss',
              'clipped_gain', 'uncensored_main_loss', 'uncensored_action_loss', 'uncensored_gain',
              'executed_main_loss', 'executed_action_loss', 'executed_gain', 'main_full_iou',
              'action_full_iou', 'full_iou_gain', 'raw_cx', 'raw_cy', 'raw_w', 'raw_h',
              'executed_x', 'executed_y', 'executed_w', 'executed_h']
    frames, states, pairs, null_pairs, execution_checks = 0, 0, 0, 0, 0
    flips, gain_errors, outside_states = [], [], 0
    with (run / 'action_rewards.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sequence in sequences:
            boxes = np.asarray(sequence['boxes'], dtype=np.float32)
            valid = set(sequence['valid_frames'])
            assert 0 in valid
            tracker.initialize(data.read_pair(sequence, 0), boxes[0])
            for frame in range(1, len(boxes)):
                image = data.read_pair(sequence, frame)
                previous = tracker.state.copy()
                templates, history = tracker.templates.clone(), tracker.history.clone()
                actual_box, route = tracker.track(image)
                frames += 1
                if frame % plan['label_stride'] or frame not in valid:
                    continue
                search, _, geometry = data.crop_pair(image, previous, 4., 256)
                search = search[None].cuda()
                left, top, side = geometry
                target = search.new_tensor(np.concatenate(
                    [(boxes[frame, :2] - [left, top]) / side, boxes[frame, 2:] / side]))[None]
                centre = target[:, :2] + target[:, 2:] / 2
                centre_inside = bool(((centre >= 0) & (centre <= 1)).all().item())
                corners = box_xywh_to_xyxy(target)
                corners_inside = bool(((corners >= 0) & (corners <= 1)).all().item())
                outside_states += int(not centre_inside)
                preview, primaries, candidates = labels(model, templates, search, history, tracker.output_window)
                executed = []
                for modality in range(2):
                    primary = primaries[modality]
                    indices, candidate_boxes = candidates[modality]
                    predicted = model.gain_predictions(primary, preview, modality, indices, candidate_boxes)[0]
                    outputs = [primary]
                    for action in range(1, 5):
                        outputs.append(model.correct(templates, search, primary, modality, action,
                                                     indices[:, action - 1], candidate_boxes[:, action - 1],
                                                     tracker.output_window))
                    metrics, pixel_boxes = [], []
                    for output in outputs:
                        raw = output['box']
                        pixels = execute_box(raw[0].cpu().numpy(), geometry, image.shape)
                        relative = np.concatenate([(pixels[:2] - [left, top]) / side, pixels[2:] / side])
                        executed_cxcywh = np.concatenate([relative[:2] + relative[2:] / 2, relative[2:]])
                        clipped, _ = localization_loss(raw, target)
                        uncensored, _ = uncensored_loss(raw, target)
                        applied, _ = uncensored_loss(raw.new_tensor(executed_cxcywh)[None], target)
                        if corners_inside:
                            assert torch.equal(clipped, uncensored)
                            null_pairs += 1
                        pixel_boxes.append(pixels)
                        metrics.append((clipped.item(), uncensored.item(), applied.item(), full_image_iou(pixels, boxes[frame])))
                    executed.append(pixel_boxes)
                    main = metrics[0]
                    for action in range(1, 5):
                        query = metrics[action]
                        clipped_gain, uncensored_gain, applied_gain = [main[index] - query[index] for index in range(3)]
                        row = dict(sequence=sequence['name'], frame=frame, modality=modality, action=action,
                                   target_centre_inside=centre_inside, target_corners_inside=corners_inside,
                                   chosen_modality=route['modality'], chosen_action=route['action'],
                                   predicted_gain=predicted[action].item(), clipped_main_loss=main[0],
                                   clipped_action_loss=query[0], clipped_gain=clipped_gain,
                                   uncensored_main_loss=main[1], uncensored_action_loss=query[1],
                                   uncensored_gain=uncensored_gain, executed_main_loss=main[2],
                                   executed_action_loss=query[2], executed_gain=applied_gain,
                                   main_full_iou=main[3], action_full_iou=query[3], full_iou_gain=query[3] - main[3])
                        raw_values = outputs[action]['box'][0].cpu().tolist()
                        row.update(dict(zip(('raw_cx', 'raw_cy', 'raw_w', 'raw_h'), raw_values)))
                        row.update(dict(zip(('executed_x', 'executed_y', 'executed_w', 'executed_h'), pixel_boxes[action].tolist())))
                        assert all(np.isfinite(row[name]) for name in fields if name not in ('sequence',))
                        writer.writerow(row)
                        if clipped_gain * applied_gain < 0:
                            flips.append(min(abs(clipped_gain), abs(applied_gain)))
                        gain_errors.append(abs(clipped_gain - applied_gain))
                        pairs += 1
                assert np.array_equal(executed[route['modality']][route['action']], actual_box)
                execution_checks += 1
                states += 1
                if states % 25 == 0:
                    handle.flush()
                    print(json.dumps(dict(labeled_states=states, tracked_frames=frames, action_pairs=pairs)), flush=True)
                if args.sanity and states == 2:
                    break
            handle.flush()
            if args.sanity:
                break
    assert all(torch.equal(value.cpu(), before[name]) for name, value in model.state_dict().items())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == plan['input_checkpoint_sha256']
    report = dict(status='PASS', physical_gpu='1', checkpoint_sha256=plan['input_checkpoint_sha256'],
                  sanity=args.sanity, sequences=1 if args.sanity else len(sequences), tracked_frames=frames,
                  labeled_states=states, action_pairs=pairs, outside_centre_states=outside_states,
                  actual_chosen_execution_bit_equal_checks=execution_checks, uncensored_null_control_outputs=null_pairs,
                  frozen_state_tensors=len(before), optimizer_updates=0, input_checkpoint_unchanged=True,
                  raw_sign_flip_pairs=len(flips),
                  flipped_min_abs_gain_quantiles=np.quantile(flips, [.0, .5, 1.]).tolist() if flips else None,
                  mean_abs_clipped_vs_executed_gain_difference=float(np.mean(gain_errors)),
                  raw_sign_flip_definition='clipped_gain * executed_gain < 0; near-zero roundoff not filtered.',
                  scope='Frozen B rollout on training sequences only. Paired same-state action diagnosis, '
                        'not model training, complete LasHeR metrics, causal trajectory improvement or a deployable oracle.')
    if args.sanity:
        assert frames == 10 and states == 2 and pairs == 16
        clone = DualRGBT().cuda().eval()
        clone.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
        expected = model.predict(templates, search, history, plan['cost_weight'], tracker.output_window)
        restored = clone.predict(templates, search, history, plan['cost_weight'], tracker.output_window)
        assert expected['modality'] == restored['modality'] and expected['action'] == restored['action']
        assert torch.equal(expected['box'], restored['box']) and torch.equal(expected['gains'], restored['gains'])
        report['input_checkpoint_second_load_prediction_bit_equal'] = True
        output = run / 'SANITY.json'
    else:
        assert frames == expected_frames and states == expected_labels and pairs == expected_labels * 8
        output = run / 'COMPLETE.json'
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
