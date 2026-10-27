"""Real-train-data admission check for the new auxiliary query executor only."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn import functional as F

from dual_rgbt.data import crop_pair, read_pair
from dual_rgbt.model import DualRGBT
from dual_rgbt_evidence.readout import LocalEvidenceReader, pixel_xywh, sample_main_features
from lib.utils.box_ops import clip_box


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    run = root / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    plan = json.loads((root / args.plan).read_text())
    assert args.policy == plan['input_checkpoint']
    assert sha(root / args.policy) == plan['input_checkpoint_sha256']
    assert sha(root / 'data/manifest.json') == plan['manifest_sha256']
    assert all(sha(root / name) == expected for name, expected in plan['source_sha256'].items())
    manifest = json.loads((root / 'data/manifest.json').read_text())
    sequence = next(row for row in manifest['train'] if row['name'] == plan['sequence_names'][0])
    assert len(sequence['boxes']) >= 11 and 0 in sequence['valid_frames']
    protected = Path('/home/gaob/CADTrack_ep0014.pth.tar')
    protected_before = sha(protected)
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    model = DualRGBT(str(protected)).cuda().eval()
    model.load_state_dict(torch.load(root / args.policy, map_location='cpu')['net'], strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('decode.', 'box_head.')))
    reader = LocalEvidenceReader(model).eval()
    assert reader.model is model
    assert {id(p) for p in reader.parameters()} == {id(p) for p in model.parameters()}
    before = cpu_state(model)
    frozen = {name: value for name, value in before.items()
              if not name.startswith(('decode.', 'box_head.'))}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    # These are tensor/geometry controls, not synthetic tracking accuracy.
    features = torch.arange(256 * 768, device='cuda', dtype=torch.float32).reshape(1, 256, 768)
    geometry = features.new_tensor([[0., 0., 256.]])
    identity = sample_main_features(features, geometry, geometry, 16)
    assert torch.equal(identity.flatten(2).transpose(1, 2), features)
    for size, left in ((8, 244.), (16, 252.)):
        outside = sample_main_features(torch.ones_like(features), geometry,
                                       geometry.new_tensor([[left, 0., 256.]]), size)
        assert torch.equal(outside, torch.zeros_like(outside))
    mapped = pixel_xywh(features.new_tensor([[.5, .5, .25, .25]]),
                       geometry.new_tensor([[300., 0., 128.]]))
    assert torch.equal(mapped, mapped.new_tensor([[348., 48., 32., 32.]]))

    image = read_pair(sequence, 0)
    previous = np.asarray(sequence['boxes'][0], dtype=np.float32)
    template = crop_pair(image, previous, 2., 128)[0].cuda()
    templates = torch.stack([template, template])[None]
    cases, updates, gradient_checks, full_null_controls = [], 0, 0, 0
    for frame in range(1, 11):
        image = read_pair(sequence, frame)
        modality = (frame - 1) % 2
        local = (frame - 1) % 4 < 2
        main_image, _, main_geometry = crop_pair(image, previous, 4., 256)
        with torch.no_grad():
            primary = model.primary(templates, main_image[None].cuda(), modality)
        main_geo = primary['features'].new_tensor([main_geometry])
        if frame <= 2:
            with torch.no_grad():
                old = model.correct(templates, main_image[None].cuda(), primary, modality, 4,
                                    torch.zeros(1, device='cuda', dtype=torch.long), primary['box'])
                same = reader(templates, main_image[None].cuda(), primary['features'], main_geo,
                              main_geo, modality)
                for field in ('box', 'score', 'size', 'offset'):
                    assert torch.equal(old[field], same[field]), field
            full_null_controls += 1
        if local:
            centre = primary['box'][0, :2].cpu().numpy() * main_geometry[2] + np.asarray(main_geometry[:2])
            side = main_geometry[2] / 2
            anchor = np.asarray([centre[0] - side / 2, centre[1] - side / 2, side, side], dtype=np.float32)
            query, _, query_geometry = crop_pair(image, anchor, 1., 128)
        else:
            query, _, query_geometry = crop_pair(image, previous, 8., 256)
        query_image = query[None].cuda()
        query_geo = primary['features'].new_tensor([query_geometry])
        output = reader(templates, query_image, primary['features'], main_geo, query_geo, modality)
        assert output['auxiliary_tokens'] == (193 if local else 385)
        assert output['query_grid_size'] == (8 if local else 16)
        assert torch.isfinite(output['pixel_box']).all() and (output['box'][:, 2:] > 0).all()
        loss_value = None
        if frame <= 4:
            target = output['box'].new_tensor(sequence['boxes'][frame])[None]
            assert (target[:, 2:] > 0).all()
            centre = (target[:, :2] + target[:, 2:] / 2 - query_geo[:, :2]) / query_geo[:, 2:3]
            target_box = torch.cat([centre, target[:, 2:] / query_geo[:, 2:3]], dim=1)
            grid = torch.arange(output['query_grid_size'], device='cuda', dtype=output['box'].dtype)
            distance = ((grid[None, None, :] - centre[:, :1, None] * output['query_grid_size']) ** 2 +
                        (grid[None, :, None] - centre[:, 1:, None] * output['query_grid_size']) ** 2)
            heatmap = torch.exp(-distance / 2)[:, None]
            loss = F.smooth_l1_loss(output['box'], target_box) + F.binary_cross_entropy(output['score'], heatmap)
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            for prefix in ('decode.', 'box_head.'):
                gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                assert sum(g.abs().sum().item() for g in gradients) > 0
            gradient_checks += 1
            optimizer.step()
            updates += 1
            loss_value = loss.item()
        # Own primary predictions maintain this diagnostic input trajectory.
        # Current-frame GT above supervises only the four training losses.
        raw_primary = pixel_xywh(primary['box'], main_geo)[0].detach().cpu().tolist()
        previous = np.asarray(clip_box(raw_primary, image.shape[0], image.shape[1], margin=10), dtype=np.float32)
        cases.append(dict(frame=frame, modality=modality, input_side=128 if local else 256,
                          auxiliary_tokens=output['auxiliary_tokens'], grid_size=output['query_grid_size'],
                          main_geometry=main_geometry.tolist(), query_geometry=query_geometry.tolist(), loss=loss_value))
        probe = (templates, query_image, primary['features'], main_geo, query_geo, modality)
    after = cpu_state(model)
    assert all(torch.equal(after[name], value) for name, value in frozen.items())
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    assert changed and all(name.startswith(('decode.', 'box_head.')) for name in changed)
    assert updates == gradient_checks == 4 and full_null_controls == 2
    assert {(row['modality'], row['input_side']) for row in cases} == {(m, s) for m in (0, 1) for s in (128, 256)}
    assert {int(state['step'].item()) for state in optimizer.state.values()} == {4}
    torch.save({'net': model.state_dict()}, run / 'sanity.pth')
    reloaded = DualRGBT(str(protected)).cuda().eval()
    reloaded.load_state_dict(torch.load(run / 'sanity.pth', map_location='cpu')['net'], strict=True)
    with torch.no_grad():
        expected = reader(*probe)
        actual = LocalEvidenceReader(reloaded).eval()(*probe)
        assert all(torch.equal(expected[key], actual[key]) for key in ('box', 'pixel_box', 'score', 'size', 'offset'))
    assert sha(root / args.policy) == plan['input_checkpoint_sha256']
    assert sha(protected) == protected_before
    assert all(sha(root / name) == expected for name, expected in plan['source_sha256'].items())
    report = dict(status='PASS', physical_gpu='1', sequence=sequence['name'], frames=10, updates=updates,
                  real_path_gradient_checks=gradient_checks, shared_parameter_identity=True,
                  frozen_state_tensors=len(frozen), changed_state_tensors=changed, full16_null_controls=full_null_controls,
                  outside_geometry_controls=2, strict_reload_prediction_bit_equal=True,
                  input_checkpoint_unchanged=True, protected_checkpoint_unchanged=True, source17_unchanged=True,
                  cases=cases, elapsed_seconds=time.perf_counter() - start,
                  scope='Real train-image executor admission only; forced balanced paths, fixed initial templates, '
                        'own primary-state trajectory. No learned query policy, final accuracy or latency evidence. '
                        'sanity.pth is not a formal training initialization.')
    (run / 'SANITY.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
