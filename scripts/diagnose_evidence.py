"""Frozen-executor diagnostics on the exact stage-A dev samples.

Candidate switches, corrections, pooling and representation interventions are
recorded separately. Oracle foreground pooling is diagnostic only; it is never
used by the online tracker or training policy.
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from dual_rgbt.data import LasHeRPairs
from dual_rgbt.model import DualRGBT
from dual_rgbt.losses import localization_loss


def boxes_from_input(model, features):
    correction = model.corrector(features)
    box = features[:, 1536:1540]
    return torch.cat([box[:, :2] + .25 * correction[:, :2].tanh(),
                      box[:, 2:] * (.5 * correction[:, 2:4].tanh()).exp()], 1)


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    model = DualRGBT().cuda().eval()
    checkpoint = root / 'runs/stage_a_diagnostic_20261013/latest.pth'
    state = torch.load(checkpoint, map_location='cpu')
    assert state['epoch'] == 2 and state['step'] == 1000
    model.load_state_dict(state['net'], strict=True)
    model.requires_grad_(False)
    captures = {}
    norm_hook = model.backbone.norm.register_forward_hook(
        lambda module, args, output: captures.update(encoded=output.detach()))
    input_hook = model.corrector.register_forward_pre_hook(
        lambda module, args: captures.update(features=args[0].detach()))
    dataset = LasHeRPairs(root / 'data/manifest.json', 'dev', 100, 2027)
    # Cache the same 100 CPU crops once. Actions retain identical search state.
    batches = list(DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False))
    rows = []
    with torch.no_grad():
        for modality in range(2):
            sample_id = 0
            for batch in batches:
                templates, search, target = [batch[k].cuda() for k in ('templates', 'search', 'target')]
                preview = model.preview_state(templates, search, search.new_zeros(len(search), 8))
                primary = model.primary(templates, search, modality)
                candidate_ids, candidate_boxes = model.candidates(primary, preview, modality)
                primary_loss, primary_iou = localization_loss(primary['box'], target)
                for action in range(1, 5):
                    candidate_box = candidate_boxes[:, action - 1]
                    result = model.correct(templates, search, primary, modality, action,
                                           candidate_ids[:, action - 1], candidate_box)
                    features = captures['features'].clone()
                    indices = result['region_indices']
                    evidence = captures['encoded'][:, -indices.shape[1]:]
                    query = features[:, :768]
                    weight = F.cosine_similarity(evidence, query[:, None], dim=2).softmax(1)
                    summary = (evidence * weight[:, :, None]).sum(1)
                    assert torch.equal(summary, features[:, 768:1536])
                    entropy = -(weight * weight.log()).sum(1) / math.log(indices.shape[1])
                    patch_lt = torch.stack([indices % 16, indices // 16], 2) / 16
                    patch_rb = patch_lt + 1. / 16
                    intersections = (torch.minimum(patch_rb, (target[:, :2] + target[:, 2:])[:, None]) -
                                     torch.maximum(patch_lt, target[:, None, :2])).clamp(min=0).prod(2)
                    foreground = intersections * 256  # Fraction of each raw patch covered by GT.
                    weighted_fg = (weight * foreground).sum(1)
                    uniform_fg = foreground.mean(1)
                    base_loss, base_iou = localization_loss(candidate_box, target)
                    actual_loss, actual_iou = localization_loss(result['box'], target)
                    null_features = features.clone()
                    null_features[:, 768:1536] = 0
                    null_box = boxes_from_input(model, null_features)
                    null_loss, null_iou = localization_loss(null_box, target)
                    shuffled_features = features.clone()
                    shuffled_features[:, 768:1536] = features[:, 768:1536].roll(1, 0)
                    shuffled_box = boxes_from_input(model, shuffled_features)
                    shuffled_loss, _ = localization_loss(shuffled_box, target)
                    eligible = foreground.sum(1) > 0
                    oracle_losses = [None] * len(search)
                    if eligible.any():
                        oracle_features = features[eligible].clone()
                        oracle_weight = foreground[eligible] / foreground[eligible].sum(1, keepdim=True)
                        oracle_features[:, 768:1536] = (
                            evidence[eligible] * oracle_weight[:, :, None]).sum(1)
                        oracle_box = boxes_from_input(model, oracle_features)
                        oracle_loss, _ = localization_loss(oracle_box, target[eligible])
                        for index, value in zip(eligible.nonzero(as_tuple=True)[0].tolist(), oracle_loss.tolist()):
                            oracle_losses[index] = value
                    for i in range(len(search)):
                        rows.append({'sample_id': sample_id + i, 'modality': modality, 'action': action,
                                     'sequence': batch['sequence'][i], 'frame': int(batch['frame'][i]),
                                     'primary_loss': primary_loss[i].item(), 'candidate_loss': base_loss[i].item(),
                                     'refined_loss': actual_loss[i].item(), 'null_summary_loss': null_loss[i].item(),
                                     'shuffled_summary_loss': shuffled_loss[i].item(),
                                     'oracle_foreground_loss': oracle_losses[i],
                                     'primary_iou': primary_iou[i].item(), 'candidate_iou': base_iou[i].item(),
                                     'refined_iou': actual_iou[i].item(), 'null_summary_iou': null_iou[i].item(),
                                     'candidate_switch_gain': (primary_loss[i] - base_loss[i]).item(),
                                     'correction_gain': (base_loss[i] - actual_loss[i]).item(),
                                     'total_gain': (primary_loss[i] - actual_loss[i]).item(),
                                     'normalized_attention_entropy': entropy[i].item(),
                                     'max_patch_weight': weight[i].max().item(),
                                     'weighted_foreground_mass': weighted_fg[i].item(),
                                     'uniform_foreground_mass': uniform_fg[i].item(),
                                     'real_vs_null_box_l1': (result['box'][i] - null_box[i]).abs().mean().item(),
                                     'real_vs_shuffle_box_l1': (result['box'][i] - shuffled_box[i]).abs().mean().item()})
                sample_id += len(search)
    norm_hook.remove()
    input_hook.remove()
    directory = root / 'reports/evidence_diagnosis'
    directory.mkdir(exist_ok=False)
    with (directory / 'samples.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for modality in range(2):
        for action in range(1, 5):
            selected = [r for r in rows if r['modality'] == modality and r['action'] == action]
            numeric = [key for key in selected[0] if key not in
                       ('sample_id', 'modality', 'action', 'sequence', 'frame', 'oracle_foreground_loss')]
            summary = {'modality': modality, 'action': action, 'samples': len(selected)}
            summary.update({key: float(np.mean([r[key] for r in selected])) for key in numeric})
            valid_oracle = [r for r in selected if r['oracle_foreground_loss'] is not None]
            summary['oracle_eligible_samples'] = len(valid_oracle)
            summary['oracle_vs_real_gain_eligible'] = float(np.mean(
                [r['refined_loss'] - r['oracle_foreground_loss'] for r in valid_oracle]))
            summaries.append(summary)
    report = {'scope': 'Frozen dev-crop diagnostic only; oracle is non-deployable, '
                        'zero/shuffle summaries are representation interventions, not physical missing modality.',
              'checkpoint_epoch': state['epoch'], 'optimizer_updates': state['step'],
              'seed': 2027, 'summary': summaries}
    (directory / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
