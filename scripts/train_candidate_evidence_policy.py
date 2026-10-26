"""Fit quality and sampled candidate-query benefits; shared executor/CNN stay frozen."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt_evidence.regions import execute_boxes, pixel_iou
from dual_rgbt_evidence.readout import pixel_xywh
from dual_rgbt_global.candidate_policy import build_candidate_policy
from dual_rgbt_global.candidate_tracker import CandidateEvidenceTracker
from dual_rgbt_global.policy import global_geometry
from dual_rgbt.data import read_pair
from dual_rgbt_evidence.regions import crop_region, search_geometry
from dual_rgbt_global.training import EvidencePairs, collate_pairs, query_pixels
from train_global_evidence_policy import cpu_state, cuda_batch, probe_inputs, step_optimizer
from lib.test.utils.hann import hann2d


PREFIXES = ('executor.quality.', 'query_gain.')
PROBE_FIELDS = ('quality', 'primary_box', 'queries', 'kinds', 'ranks', 'candidate_indices',
                'gains', 'box', 'pixel_box', 'score', 'size', 'offset')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sliced_primary(primary, index):
    return {key: primary[key][index:index + 1] for key in ('features', 'score', 'box', 'size', 'offset')}


def sliced_preview(preview, index):
    result = {key: [value[index:index + 1] for value in preview[key]]
              for key in ('pools', 'descriptors', 'broad_features', 'broad_maps')}
    result['history'] = preview['history'][index:index + 1]
    return result


@torch.no_grad()
def sampled_targets(model, batch, history, kind, max_candidates, rng, window):
    preview = model.preview_state(batch['templates'], batch['main_preview'], batch['broad_preview'], history)
    primaries, qualities, queries, gains, boxes, raw_boxes, ranks, counts = [], [], [], [], [], [], [], []
    for modality in range(2):
        primary = model.executor.primary(batch['templates'], batch['main'], modality, window)
        main_raw = pixel_xywh(primary['box'], batch['main_geometry'])
        main_box = execute_boxes(main_raw, batch['image_size'])
        main_iou = pixel_iou(main_box, batch['target'])
        chosen, chosen_ranks, candidate_counts = [], [], []
        for index in range(batch['main'].shape[0]):
            pool = model.candidate_queries(sliced_primary(primary, index), sliced_preview(preview, index),
                modality, batch['main_geometry'][index:index + 1], batch['broad_geometry'][index:index + 1],
                batch['image_size'][index:index + 1], max_candidates)
            eligible = (pool['kinds'][0] == kind).nonzero(as_tuple=False).flatten()
            request = int(eligible[int(rng.randint(len(eligible)))].item())
            chosen.append(pool['geometries'][:, request])
            chosen_ranks.append(int(pool['ranks'][0, request].item()))
            candidate_counts.append(int(pool['candidate_indices'].numel()))
        regions = torch.cat(chosen, 0)
        pixels = query_pixels(batch['images'], batch['means'], regions, kind + 1)
        output = model.query(batch['templates'], pixels, primary, batch['main_geometry'], regions, modality)
        executed = execute_boxes(output['pixel_box'], batch['image_size'])
        value = pixel_iou(executed, batch['target'])
        assert output['auxiliary_tokens'] == (385 if kind == 3 else 193)
        primaries.append(primary)
        qualities.append(main_iou)
        queries.append(regions[:, None])
        gains.append(value - main_iou)
        boxes.append((main_box, executed))
        raw_boxes.append((main_raw, output['pixel_box']))
        ranks.append(chosen_ranks)
        counts.append(candidate_counts)
    return dict(primaries=primaries, quality=torch.stack(qualities, 1), queries=queries,
                gains=gains, boxes=boxes, raw_boxes=raw_boxes, ranks=ranks, counts=counts)


@torch.no_grad()
def probe_prediction(model, batch, window, max_candidates):
    preview = model.preview_state(batch['templates'], batch['main_preview'], batch['broad_preview'], batch['history'])
    modality = int(preview['quality'].argmax(1).item())
    primary = model.executor.primary(batch['templates'], batch['main'], modality, window)
    pool = model.candidate_queries(primary, preview, modality, batch['main_geometry'],
                                  batch['broad_geometry'], batch['image_size'], max_candidates)
    gains = model.candidate_gains(primary, preview, modality, batch['main_geometry'], batch['broad_geometry'],
                                  pool['geometries'], pool['kinds'])
    legacy = model.query_geometries(primary, preview, modality, batch['main_geometry'],
                                   batch['broad_geometry'], batch['image_size'])
    one = model.candidate_queries(primary, preview, modality, batch['main_geometry'],
                                 batch['broad_geometry'], batch['image_size'], 1)
    assert torch.equal(one['geometries'], legacy)
    actual = model.candidate_gains(primary, preview, modality, batch['main_geometry'],
                                  batch['broad_geometry'], one['geometries'], one['kinds'])
    expected = model.gain_predictions(primary, preview, modality, batch['main_geometry'], batch['broad_geometry'], legacy)
    assert torch.equal(actual, expected)
    query_geo = pool['geometries'][:, 2]
    pixels = query_pixels(batch['images'], batch['means'], query_geo, 3)
    output = model.query(batch['templates'], pixels, primary, batch['main_geometry'], query_geo, modality)
    return dict(quality=preview['quality'], primary_box=primary['box'], queries=pool['geometries'],
        kinds=pool['kinds'], ranks=pool['ranks'], candidate_indices=pool['candidate_indices'], gains=gains,
        box=output['box'], pixel_box=output['pixel_box'], score=output['score'], size=output['size'], offset=output['offset'])


@torch.no_grad()
def causal_probe(model, max_candidates):
    sequence = json.loads((ROOT / 'data/manifest.json').read_text())['train'][0]
    tracker = CandidateEvidenceTracker(model, .05, max_candidates)
    before = cpu_state(model)
    tracker.initialize(read_pair(sequence, 0), sequence['boxes'][0])
    queries = 0
    for frame in range(1, 11):
        image = read_pair(sequence, frame)
        previous, templates, history = tracker.state.copy(), tracker.templates.clone(), tracker.history.clone()
        actual, route = tracker.track(image)
        geometry, broad = search_geometry(previous), global_geometry(image)
        means = image.mean((0, 1))
        main = crop_region(image, geometry, 256, means)[None].cuda()
        preview = model.preview_state(templates, crop_region(image, geometry, 64, means)[None].cuda(),
                                       crop_region(image, broad, 256, means)[None].cuda(), history)
        modality = int(preview['quality'].argmax(1).item())
        assert modality == route['modality']
        primary = model.executor.primary(templates, main, modality, tracker.window)
        main_geo, broad_geo, sizes = main.new_tensor(geometry)[None], main.new_tensor(broad)[None], main.new_tensor([image.shape[:2]])
        pool = model.candidate_queries(primary, preview, modality, main_geo, broad_geo, sizes, max_candidates)
        gains = model.candidate_gains(primary, preview, modality, main_geo, broad_geo, pool['geometries'], pool['kinds'])
        selected = int((gains - .05 * model.candidate_costs(pool['kinds'])).argmax(1).item())
        assert selected - 1 == route['request_index'] and pool['candidate_indices'].numel() == route['candidate_count']
        box = pixel_xywh(primary['box'], main_geo)
        if selected:
            index = selected - 1
            kind = int(pool['kinds'][0, index].item())
            assert kind + 1 == route['action'] and int(pool['ranks'][0, index].item()) == route['candidate_rank']
            region = pool['geometries'][:, index]
            pixels = query_pixels([image], [means], region, kind + 1)
            output = model.query(templates, pixels, primary, main_geo, region, modality)
            box = output['pixel_box']
            assert output['auxiliary_tokens'] == route['auxiliary_tokens']
            queries += 1
        else:
            assert route['action'] == route['auxiliary_tokens'] == 0 and route['candidate_rank'] == -1
        expected = execute_boxes(box, sizes)[0].cpu().numpy()
        assert np.array_equal(actual, expected)
    assert all(torch.equal(value, model.state_dict()[key].detach().cpu()) for key, value in before.items())
    return dict(tracked_frames=10, actual_auxiliary_queries=queries, replay_auxiliary_queries=queries,
                selected_execution_bit_equal=True, frozen_model_state_tensors=265,
                scope='Causal real-data execution probe only; frame0 GT initialization, no rollout training or final tracking metrics.')


def save_checkpoint(model, optimizer, run, config, step, before, probe, window):
    after = cpu_state(model)
    frozen = [name for name in before if not name.startswith(PREFIXES)]
    assert len(frozen) == 257 and all(torch.equal(before[name], after[name]) for name in frozen)
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    assert len(changed) == 8 and all(name.startswith(PREFIXES) for name in changed)
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if not name.startswith(PREFIXES))
    assert {int(state['step'].item()) for state in optimizer.state.values()} == {step}
    torch.save(dict(net=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                    step=step, stage='candidate_b'), run / 'latest.pth')
    if config['sanity']:
        expected = probe_prediction(model, probe, window, config['max_candidates'])
        clone = build_candidate_policy(ROOT / config['factory_checkpoint']).cuda().eval()
        clone.load_state_dict(torch.load(run / 'latest.pth', map_location='cpu')['net'], strict=True)
        actual = probe_prediction(clone, probe, window, config['max_candidates'])
        assert tuple(expected) == PROBE_FIELDS and all(torch.equal(expected[key], actual[key]) for key in PROBE_FIELDS)
        report = dict(status='PASS', optimizer_updates=step, frozen_state_tensors=len(frozen),
            changed_state_tensors=changed, strict_reload_prediction_bit_equal=True, probe_fields=list(PROBE_FIELDS),
            four_request_legacy_bit_equal=True, physical_gpu='1',
            scope='Real-data sampled controller admission only; sanity weights cannot initialize formal training.')
        (run / 'SANITY.json').write_text(json.dumps(report, indent=2))
    return frozen, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--sanity-complete-sha256')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1' and os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    plan = json.loads((ROOT / args.plan).read_text())
    review = json.loads((ROOT / plan['source_review']).read_text())
    result_review = json.loads((ROOT / plan['result_review']).read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == result_review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    assert sha(ROOT / 'checkpoints/CADTrack_ep0014.pth.tar') == plan['protected_checkpoint_sha256']
    assert args.run == plan['runs']['sanity' if args.sanity else 'formal']
    if args.sanity:
        assert args.sanity_complete_sha256 is None
    else:
        sanity_run = ROOT / 'runs' / plan['runs']['sanity']
        assert sha(sanity_run / 'COMPLETE.json') == args.sanity_complete_sha256
        native = json.loads((ROOT / 'runs/pipeline_tasks' / ('train_' + plan['runs']['sanity']) / 'status.json').read_text())
        complete = json.loads((sanity_run / 'COMPLETE.json').read_text())
        sanity = json.loads((sanity_run / 'SANITY.json').read_text())
        config = json.loads((sanity_run / 'config.json').read_text())
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0 and native['physical_gpu'] == '1'
        assert complete['sanity'] and complete['updates'] == plan['sanity_updates']
        assert config['plan_sha256'] == sha(ROOT / args.plan) and config['source_sha256'] == plan['source_sha256']
        assert config['input_sha256'] == plan['input_sha256']
        assert sanity['status'] == 'PASS' and sanity['strict_reload_prediction_bit_equal']
        assert tuple(sanity['probe_fields']) == PROBE_FIELDS and sanity['four_request_legacy_bit_equal']
        assert sanity['frozen_state_tensors'] == 257 and len(sanity['changed_state_tensors']) == 8
        assert complete['causal_probe']['tracked_frames'] == 10 and complete['causal_probe']['selected_execution_bit_equal']
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_candidate_policy(ROOT / plan['factory_checkpoint']).cuda().eval()
    checkpoint = torch.load(ROOT / plan['checkpoint'], map_location='cpu')
    assert checkpoint['stage'] == 'localize' and checkpoint['step'] == 6250 and not checkpoint['config']['sanity']
    assert checkpoint['config']['plan_sha256'] == plan['warmup_plan_sha256']
    model.load_state_dict(checkpoint['net'], strict=True)
    del checkpoint
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(PREFIXES))
    before = cpu_state(model)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), stage='candidate_b', plan_sha256=sha(ROOT / args.plan),
        source_sha256=plan['source_sha256'], input_sha256=plan['input_sha256'],
        factory_checkpoint=plan['factory_checkpoint'], max_candidates=plan['max_candidates'], scope=plan['scope'])
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    epochs = 1 if args.sanity else plan['epochs']
    samples = plan['sanity_updates'] * plan['batch_size'] if args.sanity else plan['samples_per_epoch']
    rng = np.random.RandomState(plan['seed'])
    step, seen, routes = 0, set(), {}
    with (run / 'steps.csv').open('w', newline='') as steps, (run / 'sampled_labels.csv').open('w', newline='') as labels:
        step_writer = csv.DictWriter(steps, fieldnames=('epoch', 'step', 'kind', 'loss', 'quality_loss', 'gain_loss'))
        label_writer = csv.DictWriter(labels, fieldnames=('epoch', 'step', 'sequence', 'frame', 'modality', 'kind',
            'candidate_rank', 'candidate_count', 'main_geometry', 'query_geometry', 'main_raw_box', 'query_raw_box', 'main_box', 'query_box',
            'main_iou', 'query_iou', 'delta_from_stop'))
        step_writer.writeheader()
        label_writer.writeheader()
        for epoch in range(1, epochs + 1):
            dataset = EvidencePairs(ROOT / 'data/manifest.json', samples, plan['seed'] + (epoch - 1) * plan['samples_per_epoch'])
            loader = DataLoader(dataset, batch_size=plan['batch_size'], num_workers=plan['workers'], collate_fn=collate_pairs)
            for batch in loader:
                seen.update(batch['sequences'])
                batch = cuda_batch(batch)
                history = batch['main'].new_zeros(batch['main'].shape[0], 8)
                kind = step % 4
                targets = sampled_targets(model, batch, history, kind, plan['max_candidates'], rng, window)
                preview = model.preview_state(batch['templates'], batch['main_preview'], batch['broad_preview'], history)
                qloss = F.smooth_l1_loss(preview['quality'], targets['quality'])
                gloss = batch['main'].new_zeros(())
                kinds = torch.full((batch['main'].shape[0], 1), kind, dtype=torch.long, device=batch['main'].device)
                for modality in range(2):
                    predictions = model.candidate_gains(targets['primaries'][modality], preview, modality,
                        batch['main_geometry'], batch['broad_geometry'], targets['queries'][modality], kinds)[:, 1]
                    gloss = gloss + F.smooth_l1_loss(predictions, targets['gains'][modality]) / 2
                loss = qloss + gloss
                step_optimizer(loss, model, optimizer, PREFIXES, args.sanity)
                step += 1
                row = dict(epoch=epoch, step=step, kind=kind, loss=loss.item(), quality_loss=qloss.item(), gain_loss=gloss.item())
                step_writer.writerow(row)
                for modality in range(2):
                    key = '%d:%d' % (modality, kind)
                    routes[key] = routes.get(key, 0) + batch['main'].shape[0]
                    for index, name in enumerate(batch['sequences']):
                        main_iou = float(targets['quality'][index, modality].item())
                        gain = float(targets['gains'][modality][index].item())
                        query_iou = float(pixel_iou(targets['boxes'][modality][1][index:index + 1], batch['target'][index:index + 1]).item())
                        label_writer.writerow(dict(epoch=epoch, step=step, sequence=name, frame=batch['frames'][index], modality=modality, kind=kind,
                            candidate_rank=targets['ranks'][modality][index], candidate_count=targets['counts'][modality][index],
                            main_geometry=json.dumps(batch['main_geometry'][index].cpu().tolist()),
                            query_geometry=json.dumps(targets['queries'][modality][index, 0].cpu().tolist()),
                            main_raw_box=json.dumps(targets['raw_boxes'][modality][0][index].cpu().tolist()),
                            query_raw_box=json.dumps(targets['raw_boxes'][modality][1][index].cpu().tolist()),
                            main_box=json.dumps(targets['boxes'][modality][0][index].cpu().tolist()),
                            query_box=json.dumps(targets['boxes'][modality][1][index].cpu().tolist()),
                            main_iou=main_iou, query_iou=query_iou, delta_from_stop=gain))
                if step == 1 or step % 25 == 0:
                    steps.flush()
                    labels.flush()
                    print(json.dumps(row), flush=True)
                probe = probe_inputs(batch, history)
            frozen, changed = save_checkpoint(model, optimizer, run, config, step, before, probe, window)
    assert step == (plan['sanity_updates'] if args.sanity else plan['expected_formal_updates'])
    assert len(seen) == (plan['sanity_expected_sequences'] if args.sanity else 882)
    assert len(routes) == 8
    causal = causal_probe(model, plan['max_candidates']) if args.sanity else None
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    assert sha(ROOT / 'checkpoints/CADTrack_ep0014.pth.tar') == plan['protected_checkpoint_sha256']
    report = dict(status='COMPLETE', sanity=args.sanity, updates=step, samples=epochs * samples,
        labelled_queries=2 * epochs * samples, seen_train_sequences=sorted(seen), route_samples=routes,
        frozen_state_tensors=len(frozen), changed_state_tensors=changed, checkpoint_sha256=sha(run / 'latest.pth'),
        private_labels_sha256=sha(run / 'sampled_labels.csv'), causal_probe=causal, scope=plan['scope'])
    (run / 'COMPLETE.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
