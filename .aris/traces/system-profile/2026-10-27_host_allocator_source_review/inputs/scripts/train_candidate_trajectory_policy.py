"""Refresh candidate benefits on causal train-only trajectories; executor stays frozen."""
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
from dual_rgbt.data import read_pair
from dual_rgbt_evidence.readout import pixel_xywh
from dual_rgbt_evidence.regions import crop_region, execute_boxes, pixel_iou, search_geometry
from dual_rgbt_global.candidate_policy import build_candidate_policy
from dual_rgbt_global.candidate_tracker import CandidateEvidenceTracker
from dual_rgbt_global.policy import global_geometry
from dual_rgbt_global.training import query_pixels
from train_candidate_evidence_policy import PREFIXES, PROBE_FIELDS, probe_prediction, sampled_targets, sha
from train_global_evidence_policy import cpu_state, probe_inputs, step_optimizer


def trajectory_counts(sequences, rounds, stride):
    return dict(tracked_frames=rounds * sum(len(row['boxes']) - 1 for row in sequences),
        updates=rounds * sum(sum(frame > 0 and frame % stride == 0 for frame in row['valid_frames'])
                            for row in sequences), initializes=rounds * len(sequences))


def causal_batch(image, previous, templates, target):
    means = image.mean((0, 1))
    geometry, broad = search_geometry(previous), global_geometry(image)
    return dict(templates=templates, main=crop_region(image, geometry, 256, means)[None].cuda(),
        main_preview=crop_region(image, geometry, 64, means)[None].cuda(),
        broad_preview=crop_region(image, broad, 256, means)[None].cuda(),
        main_geometry=templates.new_tensor(geometry)[None], broad_geometry=templates.new_tensor(broad)[None],
        image_size=templates.new_tensor([image.shape[:2]]), target=templates.new_tensor(target)[None],
        images=[image], means=[means])


@torch.no_grad()
def selected_target(model, batch, preview, targets, kind, route, executed, cost_weight, max_candidates):
    """Replay the already executed decision, reusing a matching teacher query when available."""
    modality = int(preview['quality'].argmax(1).item())
    assert modality == route['modality']
    primary = targets['primaries'][modality]
    pool = model.candidate_queries(primary, preview, modality, batch['main_geometry'],
                                  batch['broad_geometry'], batch['image_size'], max_candidates)
    gains = model.candidate_gains(primary, preview, modality, batch['main_geometry'],
                                 batch['broad_geometry'], pool['geometries'], pool['kinds'])
    selected = int((gains - cost_weight * model.candidate_costs(pool['kinds'])).argmax(1).item())
    assert selected - 1 == route['request_index']
    assert int(pool['candidate_indices'].numel()) == route['candidate_count']
    main_raw = pixel_xywh(primary['box'], batch['main_geometry'])
    result = dict(kind=-1, rank=-1, geometry=None, raw_box=main_raw,
                  box=targets['boxes'][modality][0], gain=None, extra_query=False)
    if selected:
        index = selected - 1
        selected_kind = int(pool['kinds'][0, index].item())
        rank = int(pool['ranks'][0, index].item())
        geometry = pool['geometries'][:, index]
        assert selected_kind + 1 == route['action'] and rank == route['candidate_rank']
        matches = selected_kind == kind and rank == targets['ranks'][modality][0]
        if matches:
            assert torch.equal(geometry, targets['queries'][modality][:, 0])
            raw_box, box = targets['raw_boxes'][modality][1], targets['boxes'][modality][1]
            gain = targets['gains'][modality]
        else:
            pixels = query_pixels(batch['images'], batch['means'], geometry, selected_kind + 1)
            output = model.query(batch['templates'], pixels, primary, batch['main_geometry'], geometry, modality)
            assert output['auxiliary_tokens'] == route['auxiliary_tokens']
            raw_box = output['pixel_box']
            box = execute_boxes(raw_box, batch['image_size'])
            gain = pixel_iou(box, batch['target']) - targets['quality'][:, modality]
        assert route['auxiliary_tokens'] == (385 if selected_kind == 3 else 193)
        result.update(kind=selected_kind, rank=rank, geometry=geometry, raw_box=raw_box,
                      box=box, gain=gain, extra_query=not matches)
    else:
        assert route['action'] == route['auxiliary_tokens'] == 0 and route['candidate_rank'] == -1
    assert np.array_equal(executed, result['box'][0].cpu().numpy())
    return result


def controller_loss(model, batch, preview, targets, kind, selected, modality):
    quality_loss = F.smooth_l1_loss(preview['quality'], targets['quality'])
    gain_losses = []
    kinds = torch.full((1, 1), kind, dtype=torch.long, device=batch['main'].device)
    for role in range(2):
        prediction = model.candidate_gains(targets['primaries'][role], preview, role,
            batch['main_geometry'], batch['broad_geometry'], targets['queries'][role], kinds)[:, 1]
        gain_losses.append(F.smooth_l1_loss(prediction, targets['gains'][role]))
    if selected['extra_query']:
        selected_kind = kinds.new_full((1, 1), selected['kind'])
        prediction = model.candidate_gains(targets['primaries'][modality], preview, modality,
            batch['main_geometry'], batch['broad_geometry'], selected['geometry'][:, None], selected_kind)[:, 1]
        gain_losses.append(F.smooth_l1_loss(prediction, selected['gain']))
    gain_loss = torch.stack(gain_losses).mean()
    return quality_loss + gain_loss, quality_loss, gain_loss


def save_checkpoint(model, optimizer, run, config, step, before, probe, window):
    after = cpu_state(model)
    frozen = [name for name in before if not name.startswith(PREFIXES)]
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    assert len(frozen) == 257 and all(torch.equal(before[name], after[name]) for name in frozen)
    assert len(changed) == 8 and all(name.startswith(PREFIXES) for name in changed)
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if not name.startswith(PREFIXES))
    assert {int(state['step'].item()) for state in optimizer.state.values()} == {step}
    torch.save(dict(net=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                    step=step, stage='candidate_c'), run / 'latest.pth')
    if config['sanity']:
        expected = probe_prediction(model, probe, window, config['max_candidates'])
        clone = build_candidate_policy(ROOT / config['factory_checkpoint']).cuda().eval()
        clone.load_state_dict(torch.load(run / 'latest.pth', map_location='cpu')['net'], strict=True)
        actual = probe_prediction(clone, probe, window, config['max_candidates'])
        assert tuple(expected) == PROBE_FIELDS and all(torch.equal(expected[key], actual[key]) for key in PROBE_FIELDS)
        (run / 'SANITY.json').write_text(json.dumps(dict(status='PASS', optimizer_updates=step,
            frozen_state_tensors=len(frozen), changed_state_tensors=changed,
            strict_reload_prediction_bit_equal=True, probe_fields=list(PROBE_FIELDS),
            four_request_legacy_bit_equal=True,
            scope='Real causal controller admission; sanity weights cannot initialize formal training.'), indent=2))
    return frozen, changed


LABEL_FIELDS = ('round', 'step', 'sequence', 'frame', 'purpose', 'modality', 'kind', 'candidate_rank',
    'candidate_count', 'main_geometry', 'query_geometry', 'main_raw_box', 'query_raw_box', 'main_box',
    'query_box', 'main_iou', 'query_iou', 'delta_from_stop')


def write_label(writer, iteration, step, sequence, frame, purpose, role, kind, rank, count,
                batch, main_raw, query_raw, main_box, query_box, geometry, main_iou, gain):
    writer.writerow(dict(round=iteration, step=step, sequence=sequence, frame=frame, purpose=purpose,
        modality=role, kind=kind, candidate_rank=rank, candidate_count=count,
        main_geometry=json.dumps(batch['main_geometry'][0].cpu().tolist()),
        query_geometry=json.dumps(geometry[0].cpu().tolist()), main_raw_box=json.dumps(main_raw[0].cpu().tolist()),
        query_raw_box=json.dumps(query_raw[0].cpu().tolist()), main_box=json.dumps(main_box[0].cpu().tolist()),
        query_box=json.dumps(query_box[0].cpu().tolist()), main_iou=float(main_iou.item()),
        query_iou=float(pixel_iou(query_box, batch['target']).item()), delta_from_stop=float(gain.item())))


def trajectory_stage(model, optimizer, plan, config, run, before, sequences, sanity):
    tracker = CandidateEvidenceTracker(model, plan['cost_weight'], plan['max_candidates'])
    rng = np.random.RandomState(plan['seed'])
    steps = frames = initializes = extra_queries = tracking_queries = 0
    seen, families, routes = set(), {}, {}
    fields = ('round', 'sequence', 'frame', 'step', 'loss', 'quality_loss', 'gain_loss',
              'chosen_modality', 'chosen_action', 'chosen_rank', 'sampled_kind', 'extra_selected_query')
    with (run / 'trajectory_steps.csv').open('w', newline='') as step_file, (run / 'query_labels.csv').open('w', newline='') as labels:
        step_writer, label_writer = csv.DictWriter(step_file, fields), csv.DictWriter(labels, LABEL_FIELDS)
        step_writer.writeheader()
        label_writer.writeheader()
        for iteration in range(1, (1 if sanity else plan['rounds']) + 1):
            for sequence in sequences:
                valid = set(sequence['valid_frames'])
                assert 0 in valid
                tracker.initialize(read_pair(sequence, 0), sequence['boxes'][0])
                initializes += 1
                seen.add(sequence['name'])
                for frame in range(1, len(sequence['boxes'])):
                    image = read_pair(sequence, frame)
                    previous, templates, history = tracker.state.copy(), tracker.templates.clone(), tracker.history.clone()
                    executed, route = tracker.track(image)
                    frames += 1
                    tracking_queries += int(route['action'] != 0)
                    key = '%d:%d' % (route['modality'], route['action'])
                    routes[key] = routes.get(key, 0) + 1
                    if frame % plan['label_stride'] or frame not in valid:
                        continue
                    # Current GT is first read after the deployment-equivalent track has advanced state.
                    batch = causal_batch(image, previous, templates, sequence['boxes'][frame])
                    state_after, templates_after, history_after = tracker.state.copy(), tracker.templates.clone(), tracker.history.clone()
                    kind = steps % 4
                    targets = sampled_targets(model, batch, history, kind, plan['max_candidates'], rng, tracker.window)
                    preview = model.preview_state(templates, batch['main_preview'], batch['broad_preview'], history)
                    selected = selected_target(model, batch, preview, targets, kind, route, executed,
                                               plan['cost_weight'], plan['max_candidates'])
                    assert np.array_equal(tracker.state, state_after)
                    assert torch.equal(tracker.templates, templates_after) and torch.equal(tracker.history, history_after)
                    loss, qloss, gloss = controller_loss(model, batch, preview, targets, kind, selected, route['modality'])
                    step_optimizer(loss, model, optimizer, PREFIXES, sanity)
                    steps += 1
                    extra_queries += int(selected['extra_query'])
                    row = dict(round=iteration, sequence=sequence['name'], frame=frame, step=steps,
                        loss=loss.item(), quality_loss=qloss.item(), gain_loss=gloss.item(),
                        chosen_modality=route['modality'], chosen_action=route['action'], chosen_rank=route['candidate_rank'],
                        sampled_kind=kind, extra_selected_query=selected['extra_query'])
                    step_writer.writerow(row)
                    for role in range(2):
                        family = '%d:%d' % (role, kind)
                        families[family] = families.get(family, 0) + 1
                        write_label(label_writer, iteration, steps, sequence['name'], frame, 'sampled', role,
                            kind, targets['ranks'][role][0], targets['counts'][role][0], batch,
                            targets['raw_boxes'][role][0], targets['raw_boxes'][role][1],
                            targets['boxes'][role][0], targets['boxes'][role][1], targets['queries'][role][:, 0],
                            targets['quality'][:, role], targets['gains'][role])
                    if selected['extra_query']:
                        role = route['modality']
                        write_label(label_writer, iteration, steps, sequence['name'], frame, 'selected_extra', role,
                            selected['kind'], selected['rank'], route['candidate_count'], batch,
                            targets['raw_boxes'][role][0], selected['raw_box'], targets['boxes'][role][0],
                            selected['box'], selected['geometry'], targets['quality'][:, role], selected['gain'])
                    if steps == 1 or steps % 25 == 0:
                        step_file.flush()
                        labels.flush()
                        print(json.dumps(row), flush=True)
                    probe = probe_inputs(batch, history)
                    if sanity and steps == plan['sanity_updates']:
                        frozen, changed = save_checkpoint(model, optimizer, run, config, steps, before, probe, tracker.window)
                        return dict(updates=steps, tracked_frames=frames, initializes=initializes,
                            tracking_auxiliary_queries=tracking_queries, sampled_query_labels=2 * steps,
                            extra_selected_query_labels=extra_queries, labelled_queries=2 * steps + extra_queries,
                            seen_train_sequences=sorted(seen), route_frames=routes, sampled_family_counts=families,
                            frozen_state_tensors=len(frozen), changed_state_tensors=changed,
                            selected_execution_bit_equal=True, teacher_preserves_tracker_state=True)
            frozen, changed = save_checkpoint(model, optimizer, run, config, steps, before, probe, tracker.window)
    assert dict(updates=steps, tracked_frames=frames, initializes=initializes) == plan['expected_formal_counts']
    assert len(seen) == 882 and len(families) == 8
    return dict(updates=steps, tracked_frames=frames, initializes=initializes,
        tracking_auxiliary_queries=tracking_queries, sampled_query_labels=2 * steps,
        extra_selected_query_labels=extra_queries, labelled_queries=2 * steps + extra_queries,
        seen_train_sequences=sorted(seen), route_frames=routes, sampled_family_counts=families,
        frozen_state_tensors=len(frozen), changed_state_tensors=changed,
        selected_execution_bit_equal=True, teacher_preserves_tracker_state=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--sanity-complete-sha256')
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text())
    assert plan['status'] == 'READY'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(plan['physical_gpu'])
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    review = json.loads((ROOT / plan['source_review']).read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    assert sha(ROOT / 'checkpoints/CADTrack_ep0014.pth.tar') == plan['protected_checkpoint_sha256']
    assert args.run == plan['runs']['sanity' if args.sanity else 'formal']
    assert all(name in plan['input_sha256'] for name in
               (plan['checkpoint'], plan['b_complete'], plan['b_native_status']))
    b_complete = json.loads((ROOT / plan['b_complete']).read_text())
    b_native = json.loads((ROOT / plan['b_native_status']).read_text())
    assert b_native['state'] == 'COMPLETE' and b_native['exit_code'] == 0 and b_native['physical_gpu'] == '1'
    assert b_complete['status'] == 'COMPLETE' and not b_complete['sanity']
    assert b_complete['updates'] == 6250 and b_complete['samples'] == 50000 and b_complete['labelled_queries'] == 100000
    assert len(b_complete['seen_train_sequences']) == 882 and b_complete['frozen_state_tensors'] == 257
    assert len(b_complete['changed_state_tensors']) == 8 and all(name.startswith(PREFIXES) for name in b_complete['changed_state_tensors'])
    assert b_complete['route_samples'] == {'%d:%d' % (role, kind): 8 * ((6250 + 3 - kind) // 4)
                                         for role in range(2) for kind in range(4)}
    assert b_complete['checkpoint_sha256'] == sha(ROOT / plan['checkpoint'])
    if args.sanity:
        assert args.sanity_complete_sha256 is None
    else:
        sanity_run = ROOT / 'runs' / plan['runs']['sanity']
        assert sha(sanity_run / 'COMPLETE.json') == args.sanity_complete_sha256
        native = json.loads((ROOT / plan['sanity_native_status']).read_text())
        complete = json.loads((sanity_run / 'COMPLETE.json').read_text())
        sanity = json.loads((sanity_run / 'SANITY.json').read_text())
        sanity_config = json.loads((sanity_run / 'config.json').read_text())
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
        assert native['physical_gpu'] == str(plan['physical_gpu'])
        assert complete['sanity'] and complete['updates'] == plan['sanity_updates']
        assert complete['selected_execution_bit_equal'] and complete['teacher_preserves_tracker_state']
        assert sanity_config['plan_sha256'] == sha(ROOT / args.plan)
        assert sanity_config['source_sha256'] == plan['source_sha256'] and sanity_config['input_sha256'] == plan['input_sha256']
        assert sanity['status'] == 'PASS' and sanity['strict_reload_prediction_bit_equal']
        assert tuple(sanity['probe_fields']) == PROBE_FIELDS and sanity['four_request_legacy_bit_equal']
        assert sanity['frozen_state_tensors'] == 257 and len(sanity['changed_state_tensors']) == 8
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    by_name = {row['name']: row for row in manifest['train']}
    assert len(by_name) == len(plan['sequence_names']) == 882 and set(by_name) == set(plan['sequence_names'])
    assert sorted(b_complete['seen_train_sequences']) == sorted(by_name)
    sequences = [by_name[name] for name in plan['sequence_names']]
    assert trajectory_counts(sequences, plan['rounds'], plan['label_stride']) == plan['expected_formal_counts']
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_candidate_policy(ROOT / plan['factory_checkpoint']).cuda().eval()
    checkpoint = torch.load(ROOT / plan['checkpoint'], map_location='cpu')
    assert checkpoint['stage'] == 'candidate_b' and checkpoint['step'] == 6250 and not checkpoint['config']['sanity']
    assert checkpoint['config']['plan_sha256'] == plan['b_plan_sha256']
    b_plan = json.loads((ROOT / 'reports/global_candidate_policy_plan.json').read_text())
    assert checkpoint['config']['source_sha256'] == b_plan['source_sha256']
    assert checkpoint['config']['input_sha256'] == b_plan['input_sha256']
    model.load_state_dict(checkpoint['net'], strict=True)
    del checkpoint
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(PREFIXES))
    before = cpu_state(model)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), stage='candidate_c', plan_sha256=sha(ROOT / args.plan),
        source_sha256=plan['source_sha256'], input_sha256=plan['input_sha256'],
        factory_checkpoint=plan['factory_checkpoint'], max_candidates=plan['max_candidates'], scope=plan['scope'])
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    result = trajectory_stage(model, optimizer, plan, config, run, before, sequences[:1] if args.sanity else sequences, args.sanity)
    assert len(result['sampled_family_counts']) == 8
    assert result['updates'] == (plan['sanity_updates'] if args.sanity else plan['expected_formal_counts']['updates'])
    if args.sanity:
        assert {key: result[key] for key in plan['sanity_expected_counts']} == plan['sanity_expected_counts']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    assert sha(ROOT / 'checkpoints/CADTrack_ep0014.pth.tar') == plan['protected_checkpoint_sha256']
    result.update(status='COMPLETE', sanity=args.sanity, checkpoint_sha256=sha(run / 'latest.pth'),
                  private_labels_sha256=sha(run / 'query_labels.csv'), scope=plan['scope'])
    (run / 'COMPLETE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
