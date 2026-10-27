"""Verify actual B updates, frozen executor, reload and causal online execution."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from dual_rgbt.data import read_pair, crop_pair
from dual_rgbt.model import DualRGBT
from dual_rgbt.tracker import Tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executor', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1')
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    executor = torch.load(args.executor, map_location='cpu')['net']
    policy = torch.load(args.policy, map_location='cpu')
    weights = policy['net']
    assert policy['step'] > 0
    frozen = [key for key in weights if key.startswith(('backbone.', 'decode.', 'box_head.'))]
    assert all(torch.equal(weights[key], executor[key]) for key in frozen)
    for prefix in ('preview.', 'quality.', 'gain.'):
        assert any(not torch.equal(weights[key], executor[key]) for key in weights if key.startswith(prefix))
    model = DualRGBT().cuda().eval()
    model.load_state_dict(weights, strict=True)
    clone = DualRGBT().cuda().eval()
    clone.load_state_dict(weights, strict=True)
    sequence = json.loads((root / 'data/manifest.json').read_text())['train'][0]
    assert 0 in sequence['valid_frames']
    image = read_pair(sequence, 0)
    box = np.asarray(sequence['boxes'][0], dtype=np.float32)
    template = crop_pair(image, box, 2., 128)[0].cuda()
    templates = torch.stack([template, template])[None]
    search = crop_pair(image, box, 4., 256)[0][None].cuda()
    history = search.new_zeros(1, 8)
    first = model.predict(templates, search, history, .05)
    reloaded = clone.predict(templates, search, history, .05)
    assert first['modality'] == reloaded['modality'] and first['action'] == reloaded['action']
    assert torch.equal(first['box'], reloaded['box']) and torch.equal(first['gains'], reloaded['gains'])
    tracker = Tracker(model, .05)
    tracker.initialize(image, box)
    trace = []
    hook = model.backbone.blocks[0].attn.register_forward_pre_hook(
        lambda module, inputs: trace.append(inputs[0].shape[1]))
    rows = []
    for frame in range(1, 33):
        trace.clear()
        prediction, route = tracker.track(read_pair(sequence, frame))
        expected = [385]
        if route['action']:
            expected.append(385 if route['action'] == 4 else 193)
        assert trace == expected, (frame, trace, route)
        assert np.isfinite(prediction).all() and (prediction[2:] >= 10).all()
        assert torch.isfinite(tracker.history).all()
        rows.append(dict(frame=frame, box=prediction.tolist(), heavy_token_trace=list(trace), **route))
    hook.remove()
    report = dict(status='PASS', physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                  actual_policy_updates=policy['step'], frozen_executor_state_tensors=len(frozen),
                  strict_load_instances_bit_equal=True, sequence=sequence['name'], frames=rows,
                  scope='B checkpoint and 32 real closed-loop training frames; not accuracy or speed evidence.')
    (root / args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
