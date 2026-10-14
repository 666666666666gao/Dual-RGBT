"""Independent-region executor A, benefit policy B, and causal trajectory refresh C."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt.data import read_pair
from dual_rgbt_global.policy import build_policy, global_geometry
from dual_rgbt_evidence.regions import crop_region, search_geometry
from dual_rgbt_global.tracker import EvidenceTracker
from dual_rgbt_global.training import (EvidencePairs, collate_pairs, counterfactuals,
                                         path_loss, policy_loss, query_pixels)
from lib.test.utils.hann import hann2d


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def cuda_batch(batch):
    for key in ('templates', 'main', 'main_preview', 'broad_preview', 'main_geometry',
                'broad_geometry', 'target', 'image_size'):
        batch[key] = batch[key].cuda()
    return batch


def step_optimizer(loss, model, optimizer, prefixes, sanity):
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if sanity:
        for prefix in prefixes:
            gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
            assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert sum(g.abs().sum().item() for g in gradients) > 0, prefix
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    assert torch.isfinite(norm)
    optimizer.step()


def probe_inputs(batch, history):
    keys = ('templates', 'main', 'main_preview', 'broad_preview', 'main_geometry', 'broad_geometry', 'image_size')
    result = {key: batch[key][:1] for key in keys}
    result.update(history=history[:1], images=batch['images'][:1], means=batch['means'][:1])
    return result


@torch.no_grad()
def probe_prediction(model, batch, window):
    preview = model.preview_state(batch['templates'], batch['main_preview'], batch['broad_preview'], batch['history'])
    modality = int(preview['quality'].argmax(1).item())
    primary = model.executor.primary(batch['templates'], batch['main'], modality, window)
    queries = model.query_geometries(primary, preview, modality, batch['main_geometry'], batch['broad_geometry'], batch['image_size'])
    gains = model.gain_predictions(primary, preview, modality, batch['main_geometry'], batch['broad_geometry'], queries)
    pixels = query_pixels(batch['images'], batch['means'], queries[:, 2], 3)
    output = model.query(batch['templates'], pixels, primary, batch['main_geometry'], queries[:, 2], modality)
    return dict(quality=preview['quality'], gains=gains, queries=queries, primary_box=primary['box'],
                box=output['box'], pixel_box=output['pixel_box'], score=output['score'])


def save_and_check(model, optimizer, run, config, step, before, prefixes, probe, window):
    torch.save(dict(net=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                    step=step, stage=config['stage']), run / 'latest.pth')
    if not config['sanity']:
        return
    after = cpu_state(model)
    frozen = [name for name in before if not name.startswith(prefixes)]
    assert all(torch.equal(before[name], after[name]) for name in frozen)
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    assert changed and all(name.startswith(prefixes) for name in changed)
    for prefix in prefixes:
        assert any(name.startswith(prefix) for name in changed), prefix
    assert {int(state['step'].item()) for state in optimizer.state.values()} == {step}
    expected = probe_prediction(model, probe, window)
    clone = build_policy(config['executor']).cuda().eval()
    clone.load_state_dict(torch.load(run / 'latest.pth', map_location='cpu')['net'], strict=True)
    actual = probe_prediction(clone, probe, window)
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    report = dict(status='PASS', stage=config['stage'], updates=step, frozen_state_tensors=len(frozen),
                  changed_state_tensors=changed, strict_reload_prediction_bit_equal=True,
                  probe_fields=list(expected), physical_gpu='1',
                  scope='Real-data stage admission only; these sanity weights cannot initialize formal training.')
    (run / 'SANITY.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def pair_stages(args, model, optimizer, prefixes, run, config, before, window):
    fields = ('epoch', 'step', 'modality', 'action', 'loss', 'quality_loss', 'gain_loss',
              'preview_loss', 'visible_fraction', 'teacher_query')
    step, routes, seen_sequences = 0, {}, set()
    with (run / 'steps.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            dataset = EvidencePairs(ROOT / 'data/manifest.json', args.samples, args.seed + (epoch - 1) * args.samples)
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                collate_fn=collate_pairs)
            for batch in loader:
                seen_sequences.update(batch['sequences'])
                batch = cuda_batch(batch)
                history = batch['main'].new_zeros(batch['main'].shape[0], 8)
                qloss = gloss = spatial = batch['main'].new_zeros(())
                visible = teacher = None
                if args.stage == 'a':
                    modality, action = step % 2, (step // 2) % 5
                    teacher = False
                    if action == 0:
                        output = model.executor.primary(batch['templates'], batch['main'], modality)
                        geometry = batch['main_geometry']
                    else:
                        with torch.no_grad():
                            preview = model.preview_state(batch['templates'], batch['main_preview'], batch['broad_preview'], history)
                            primary = model.executor.primary(batch['templates'], batch['main'], modality)
                            queries = model.query_geometries(primary, preview, modality, batch['main_geometry'], batch['broad_geometry'], batch['image_size'])
                            geometry = queries[:, action - 1].clone()
                            # Half the external-query batches teach useful off-main evidence;
                            # the other half use the deployed cheap proposal and include misses.
                            teacher = action in (3, 4) and (step // 10) % 2 == 0
                            if teacher:
                                centre = batch['target'][:, :2] + batch['target'][:, 2:] / 2
                                jitter = (torch.rand_like(centre) - .5) * geometry[:, 2:3] / 2
                                geometry[:, :2] = centre + jitter - geometry[:, 2:3] / 2
                        pixels = query_pixels(batch['images'], batch['means'], geometry, action)
                        output = model.query(batch['templates'], pixels, primary, batch['main_geometry'], geometry, modality)
                    loss, mask = path_loss(output, batch['target'], geometry)
                    visible = mask.float().mean().item()
                    key = '%d:%d' % (modality, action)
                    routes[key] = routes.get(key, 0) + batch['main'].shape[0]
                else:
                    modality = action = None
                    targets = counterfactuals(model, batch['templates'], batch['main'], batch['main_preview'],
                                             batch['broad_preview'], history, batch['main_geometry'],
                                             batch['broad_geometry'], batch['target'], batch['image_size'],
                                             batch['images'], batch['means'], window)
                    loss, qloss, gloss, spatial = policy_loss(
                        model, targets, batch['templates'], batch['main_preview'], batch['broad_preview'], history,
                        batch['main_geometry'], batch['broad_geometry'], batch['target'], True)
                step_optimizer(loss, model, optimizer, prefixes, args.sanity)
                step += 1
                row = dict(epoch=epoch, step=step, modality=modality, action=action, loss=loss.item(),
                           quality_loss=qloss.item(), gain_loss=gloss.item(), preview_loss=spatial.item(),
                           visible_fraction=visible, teacher_query=teacher)
                writer.writerow(row)
                if step == 1 or step % 25 == 0:
                    handle.flush()
                    print(json.dumps(row), flush=True)
                probe = probe_inputs(batch, history)
            save_and_check(model, optimizer, run, config, step, before, prefixes, probe, window)
    return dict(epochs=args.epochs, updates=step, samples=args.epochs * args.samples,
                route_samples=routes, seen_train_sequences=sorted(seen_sequences))


def trajectory_stage(args, model, optimizer, prefixes, run, config, before, window, plan):
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    by_name = {row['name']: row for row in manifest['train']}
    names = plan['sequence_names'][:1] if args.sanity else plan['sequence_names']
    sequences = [by_name[name] for name in names]
    tracker = EvidenceTracker(model, args.cost_weight)
    fields = ('round', 'sequence', 'frame', 'step', 'loss', 'quality_loss', 'gain_loss',
              'chosen_modality', 'chosen_action', 'quality_targets', 'gain_targets', 'executed_action_boxes')
    step = frames = initializes = 0
    with (run / 'trajectory_steps.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for iteration in range(1, args.rounds + 1):
            for sequence in sequences:
                valid = set(sequence['valid_frames'])
                assert 0 in valid
                tracker.initialize(read_pair(sequence, 0), sequence['boxes'][0])
                initializes += 1
                for frame in range(1, len(sequence['boxes'])):
                    image = read_pair(sequence, frame)
                    previous = tracker.state.copy()
                    templates, history = tracker.templates.clone(), tracker.history.clone()
                    executed, route = tracker.track(image)
                    frames += 1
                    if frame % args.label_stride or frame not in valid:
                        continue
                    means = image.mean((0, 1))
                    geometry = search_geometry(previous)
                    broad = global_geometry(image)
                    batch = dict(templates=templates, main=crop_region(image, geometry, 256, means)[None].cuda(),
                                 main_preview=crop_region(image, geometry, 64, means)[None].cuda(),
                                 broad_preview=crop_region(image, broad, 256, means)[None].cuda(),
                                 main_geometry=templates.new_tensor(geometry)[None],
                                 broad_geometry=templates.new_tensor(broad)[None],
                                 target=templates.new_tensor(sequence['boxes'][frame])[None],
                                 image_size=templates.new_tensor([image.shape[:2]]), images=[image], means=[means])
                    targets = counterfactuals(model, templates, batch['main'], batch['main_preview'],
                                             batch['broad_preview'], history, batch['main_geometry'],
                                             batch['broad_geometry'], batch['target'], batch['image_size'],
                                             batch['images'], batch['means'], window)
                    chosen = targets['executed'][route['modality']][0, route['action']].cpu().numpy()
                    assert np.array_equal(chosen, executed), 'Selected counterfactual must reproduce actual execution.'
                    loss, qloss, gloss, _ = policy_loss(
                        model, targets, templates, batch['main_preview'], batch['broad_preview'], history,
                        batch['main_geometry'], batch['broad_geometry'], batch['target'], False)
                    step_optimizer(loss, model, optimizer, prefixes, args.sanity)
                    step += 1
                    row = dict(round=iteration, sequence=sequence['name'], frame=frame, step=step,
                               loss=loss.item(), quality_loss=qloss.item(), gain_loss=gloss.item(),
                               chosen_modality=route['modality'], chosen_action=route['action'],
                               quality_targets=json.dumps(targets['quality'].cpu().tolist()),
                               gain_targets=json.dumps([value.cpu().tolist() for value in targets['gains']]),
                               executed_action_boxes=json.dumps([value.cpu().tolist() for value in targets['executed']]))
                    writer.writerow(row)
                    if step == 1 or step % 25 == 0:
                        handle.flush()
                        print(json.dumps({key: row[key] for key in fields[:9]}), flush=True)
                    probe = probe_inputs(batch, history)
                    if args.sanity and step == 2:
                        save_and_check(model, optimizer, run, config, step, before, prefixes, probe, window)
                        return dict(rounds=iteration, sequences=len(sequences), tracked_frames=frames,
                                    updates=step, initializes=initializes)
                torch.save(dict(net=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                                step=step, stage='c'), run / 'latest.pth')
    return dict(rounds=args.rounds, sequences=len(sequences), tracked_frames=frames,
                updates=step, initializes=initializes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('a', 'b', 'c'), required=True)
    parser.add_argument('--executor', required=True)
    parser.add_argument('--policy')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--label-stride', type=int, default=5)
    parser.add_argument('--cost-weight', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--sanity', action='store_true')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    plan = json.loads((ROOT / args.plan).read_text())
    assert args.seed == plan['seed']
    assert args.executor == plan['input_checkpoint'] and sha(ROOT / args.executor) == plan['input_checkpoint_sha256']
    assert sha(ROOT / 'data/manifest.json') == plan['manifest_sha256']
    assert sha(ROOT / plan['initial_policy']) == plan['initial_policy_sha256']
    assert all(sha(ROOT / name) == expected for name, expected in plan['source_sha256'].items())
    tag = ('sanity_' if args.sanity else 'formal_') + args.stage
    assert args.run == plan['runs'][tag]
    assert all(getattr(args, key) == value for key, value in plan['settings'][tag].items())
    if args.stage == 'a':
        assert args.policy is None
    else:
        prior = ('sanity_' if args.sanity else 'formal_') + ('a' if args.stage == 'b' else 'b')
        assert args.policy == 'runs/' + plan['runs'][prior] + '/latest.pth'
        input_state = torch.load(ROOT / args.policy, map_location='cpu')
        assert input_state['config']['sanity'] == args.sanity
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_policy(ROOT / args.executor).cuda().eval()
    if args.stage == 'a':
        model.load_state_dict(torch.load(ROOT / plan['initial_policy'], map_location='cpu')['net'], strict=True)
    if args.stage != 'a':
        model.load_state_dict(input_state['net'], strict=True)
        del input_state
    prefixes = dict(a=('executor.decode.', 'executor.box_head.'),
                    b=('executor.preview.', 'executor.quality.', 'query_gain.'),
                    c=('executor.quality.', 'query_gain.'))[args.stage]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5 if args.stage == 'a' else 1e-4, weight_decay=1e-4)
    before = cpu_state(model) if args.sanity else None
    protected = Path('/home/gaob/CADTrack_ep0014.pth.tar')
    protected_sha = sha(protected)
    config = dict(vars(args), plan_sha256=sha(ROOT / args.plan), protected_checkpoint_sha256=protected_sha,
                  input_policy_sha256=sha(ROOT / args.policy) if args.policy else plan['initial_policy_sha256'],
                  initial_policy=plan['initial_policy'], initial_policy_sha256=plan['initial_policy_sha256'],
                  source_sha256=plan['source_sha256'], gain_target='Executed original-image IoU delta; uncensored GT.')
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    if args.stage == 'c':
        result = trajectory_stage(args, model, optimizer, prefixes, run, config, before, window, plan)
    else:
        result = pair_stages(args, model, optimizer, prefixes, run, config, before, window)
    assert sha(ROOT / args.executor) == plan['input_checkpoint_sha256']
    assert sha(ROOT / plan['initial_policy']) == plan['initial_policy_sha256']
    assert sha(protected) == protected_sha
    assert all(sha(ROOT / name) == expected for name, expected in plan['source_sha256'].items())
    (run / 'COMPLETE.json').write_text(json.dumps(dict(result, stage=args.stage, sanity=args.sanity), indent=2))
    print(json.dumps(dict(result, stage=args.stage, sanity=args.sanity, state='COMPLETE')), flush=True)


if __name__ == '__main__':
    main()
