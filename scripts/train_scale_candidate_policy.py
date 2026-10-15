"""Fit the two-radius candidate controller using newly executed training labels."""
import argparse
import csv
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
from dual_rgbt_evidence.readout import pixel_xywh
from dual_rgbt_evidence.regions import execute_boxes, pixel_iou
from dual_rgbt_global.scale_candidate_checkpoint import load_scale_candidate_checkpoint
from dual_rgbt_global.scale_candidate_policy import build_scale_candidate_policy
from dual_rgbt_global.training import EvidencePairs, collate_pairs, query_pixels
from lib.test.utils.hann import hann2d
from train_candidate_evidence_policy import PREFIXES, causal_probe, sha, sliced_preview, sliced_primary
from train_global_evidence_policy import cpu_state, cuda_batch, probe_inputs, step_optimizer


PROBE_FIELDS = ('quality', 'primary_box', 'queries', 'kinds', 'ranks', 'candidate_indices',
    'request_cell_indices', 'family_candidate_counts', 'gains', 'box', 'pixel_box', 'score', 'size', 'offset')


@torch.no_grad()
def sampled_targets(model, batch, history, kind, capacity, rng, window):
    preview = model.preview_state(batch['templates'], batch['main_preview'], batch['broad_preview'], history)
    result = {key: [] for key in ('primaries', 'quality', 'queries', 'gains', 'main_box', 'query_box',
                                  'main_raw', 'query_raw', 'ranks', 'cells', 'family_counts', 'unique_counts')}
    for modality in range(2):
        primary = model.executor.primary(batch['templates'], batch['main'], modality, window)
        main_raw = pixel_xywh(primary['box'], batch['main_geometry'])
        main_box = execute_boxes(main_raw, batch['image_size'])
        main_iou = pixel_iou(main_box, batch['target'])
        selected = {key: [] for key in ('queries', 'ranks', 'cells', 'family_counts', 'unique_counts')}
        for index in range(batch['main'].shape[0]):
            pool = model.candidate_queries(sliced_primary(primary, index), sliced_preview(preview, index),
                modality, batch['main_geometry'][index:index + 1], batch['broad_geometry'][index:index + 1],
                batch['image_size'][index:index + 1], capacity)
            eligible = (pool['kinds'][0] == kind).nonzero(as_tuple=False).flatten()
            request = int(eligible[int(rng.randint(len(eligible)))].item())
            selected['queries'].append(pool['geometries'][:, request])
            selected['ranks'].append(int(pool['ranks'][0, request].item()))
            selected['cells'].append(int(pool['request_cell_indices'][0, request].item()))
            selected['family_counts'].append(int(eligible.numel()))
            selected['unique_counts'].append(int(pool['candidate_indices'].numel()))
        regions = torch.cat(selected['queries'], 0)
        pixels = query_pixels(batch['images'], batch['means'], regions, kind + 1)
        output = model.query(batch['templates'], pixels, primary, batch['main_geometry'], regions, modality)
        query_box = execute_boxes(output['pixel_box'], batch['image_size'])
        query_iou = pixel_iou(query_box, batch['target'])
        assert output['auxiliary_tokens'] == (385 if kind == 3 else 193)
        for key, value in dict(primaries=primary, quality=main_iou, queries=regions[:, None],
            gains=query_iou - main_iou, main_box=main_box, query_box=query_box,
            main_raw=main_raw, query_raw=output['pixel_box']).items():
            result[key].append(value)
        for key in ('ranks', 'cells', 'family_counts', 'unique_counts'):
            result[key].append(selected[key])
    result['quality'] = torch.stack(result['quality'], 1)
    return result


@torch.no_grad()
def probe_prediction(model, batch, window, capacity):
    preview = model.preview_state(batch['templates'], batch['main_preview'], batch['broad_preview'], batch['history'])
    modality = int(preview['quality'].argmax(1).item())
    primary = model.executor.primary(batch['templates'], batch['main'], modality, window)
    pool = model.candidate_queries(primary, preview, modality, batch['main_geometry'],
                                  batch['broad_geometry'], batch['image_size'], capacity)
    gains = model.candidate_gains(primary, preview, modality, batch['main_geometry'],
                                 batch['broad_geometry'], pool['geometries'], pool['kinds'])
    one = model.candidate_queries(primary, preview, modality, batch['main_geometry'],
                                 batch['broad_geometry'], batch['image_size'], 1)
    legacy = model.query_geometries(primary, preview, modality, batch['main_geometry'],
                                    batch['broad_geometry'], batch['image_size'])
    assert torch.equal(one['geometries'], legacy)
    actual = model.candidate_gains(primary, preview, modality, batch['main_geometry'],
                                  batch['broad_geometry'], one['geometries'], one['kinds'])
    expected = model.gain_predictions(primary, preview, modality, batch['main_geometry'], batch['broad_geometry'], legacy)
    assert torch.equal(actual, expected)
    region = pool['geometries'][:, 2]
    output = model.query(batch['templates'], query_pixels(batch['images'], batch['means'], region, 3),
                         primary, batch['main_geometry'], region, modality)
    return dict(quality=preview['quality'], primary_box=primary['box'], queries=pool['geometries'],
        kinds=pool['kinds'], ranks=pool['ranks'], candidate_indices=pool['candidate_indices'],
        request_cell_indices=pool['request_cell_indices'], family_candidate_counts=pool['family_candidate_counts'],
        gains=gains, box=output['box'], pixel_box=output['pixel_box'], score=output['score'],
        size=output['size'], offset=output['offset'])


def save_checkpoint(model, optimizer, run, config, step, before, probe, window):
    after = cpu_state(model)
    frozen = [name for name in before if not name.startswith(PREFIXES)]
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    assert len(frozen) == 257 and all(torch.equal(before[name], after[name]) for name in frozen)
    assert len(changed) == 8 and all(name.startswith(PREFIXES) for name in changed)
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if not name.startswith(PREFIXES))
    assert {int(state['step'].item()) for state in optimizer.state.values()} == {step}
    torch.save(dict(net=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                    step=step, stage='scale_candidate_b'), run / 'latest.pth')
    if config['sanity']:
        expected = probe_prediction(model, probe, window, config['candidate_config']['max_candidates'])
        clone, restored = load_scale_candidate_checkpoint(run / 'latest.pth', ROOT)
        clone = clone.cuda().eval()
        assert restored == config
        assert clone.local_radius_fraction == config['candidate_config']['local_radius_fraction']
        assert clone.wide_radius_fraction == config['candidate_config']['wide_radius_fraction']
        actual = probe_prediction(clone, probe, window, restored['candidate_config']['max_candidates'])
        assert tuple(expected) == PROBE_FIELDS and all(torch.equal(expected[key], actual[key]) for key in PROBE_FIELDS)
        (run / 'SANITY.json').write_text(json.dumps(dict(status='PASS', optimizer_updates=step,
            frozen_state_tensors=len(frozen), changed_state_tensors=changed,
            strict_reload_prediction_bit_equal=True, probe_fields=list(PROBE_FIELDS),
            saved_candidate_configuration_restored=True, four_request_legacy_bit_equal=True,
            physical_gpu='1', scope='Real-data controller admission only; M0 weights cannot initialize formal training.'), indent=2))
    return frozen, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--sanity-complete-sha256')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1' and os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    plan = json.loads((ROOT / args.plan).read_bytes())
    review = json.loads((ROOT / plan['source_review']).read_bytes())
    assert review['source_verdict'] == 'PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    assert sha('/home/gaob/CADTrack_ep0014.pth.tar') == plan['protected_checkpoint_sha256']
    assert args.run == plan['runs']['sanity' if args.sanity else 'formal']
    candidate = plan['candidate_config']
    assert candidate['max_candidates'] == 8
    if args.sanity:
        assert args.sanity_complete_sha256 is None
    else:
        sanity_run = ROOT / 'runs' / plan['runs']['sanity']
        assert sha(sanity_run / 'COMPLETE.json') == args.sanity_complete_sha256
        native = json.loads((ROOT / 'runs/pipeline_tasks' / ('train_' + plan['runs']['sanity']) / 'status.json').read_bytes())
        complete = json.loads((sanity_run / 'COMPLETE.json').read_bytes())
        sanity = json.loads((sanity_run / 'SANITY.json').read_bytes())
        config = json.loads((sanity_run / 'config.json').read_bytes())
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0 and native['physical_gpu'] == '1'
        assert native['command'] == [plan['remote_python'], plan['remote_root'] + '/scripts/train_scale_candidate_policy.py',
            '--plan', args.plan, '--run', plan['runs']['sanity'], '--sanity']
        assert complete['status'] == 'COMPLETE' and complete['sanity'] and complete['updates'] == plan['sanity_updates']
        assert complete['stage'] == config['stage'] == 'scale_candidate_b'
        assert config['plan_sha256'] == sha(ROOT / args.plan) and config['source_sha256'] == plan['source_sha256']
        assert config['input_sha256'] == plan['input_sha256'] and config['candidate_config'] == candidate
        assert sanity['status'] == 'PASS' and sanity['strict_reload_prediction_bit_equal']
        assert sanity['saved_candidate_configuration_restored'] and tuple(sanity['probe_fields']) == PROBE_FIELDS
        assert sanity['four_request_legacy_bit_equal'] and sanity['frozen_state_tensors'] == 257
        assert len(sanity['changed_state_tensors']) == 8 and complete['causal_probe']['tracked_frames'] == 10
        assert complete['causal_probe']['selected_execution_bit_equal']
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_scale_candidate_policy(ROOT / plan['factory_checkpoint'],
        candidate['local_radius_fraction'], candidate['wide_radius_fraction']).cuda().eval()
    warmup = torch.load(ROOT / plan['checkpoint'], map_location='cpu')
    assert warmup['stage'] == 'localize' and warmup['step'] == 6250 and not warmup['config']['sanity']
    assert warmup['config']['plan_sha256'] == plan['warmup_plan_sha256']
    model.load_state_dict(warmup['net'], strict=True)
    del warmup
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(PREFIXES))
    before = cpu_state(model)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), stage='scale_candidate_b', plan_sha256=sha(ROOT / args.plan),
        source_sha256=plan['source_sha256'], input_sha256=plan['input_sha256'],
        factory_checkpoint=plan['factory_checkpoint'], candidate_config=candidate, scope=plan['scope'])
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    epochs = 1 if args.sanity else plan['epochs']
    samples = plan['sanity_updates'] * plan['batch_size'] if args.sanity else plan['samples_per_epoch']
    rng, step, seen, routes = np.random.RandomState(plan['seed']), 0, set(), {}
    label_fields = ('epoch', 'step', 'sequence', 'frame', 'modality', 'kind', 'candidate_rank',
        'request_cell', 'family_candidate_count', 'unique_candidate_count', 'main_geometry',
        'query_geometry', 'main_raw_box', 'query_raw_box', 'main_box', 'query_box',
        'main_iou', 'query_iou', 'delta_from_stop')
    with (run / 'steps.csv').open('w', newline='') as steps, (run / 'sampled_labels.csv').open('w', newline='') as labels:
        step_writer = csv.DictWriter(steps, fieldnames=('epoch', 'step', 'kind', 'loss', 'quality_loss', 'gain_loss'))
        label_writer = csv.DictWriter(labels, fieldnames=label_fields)
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
                targets = sampled_targets(model, batch, history, kind, candidate['max_candidates'], rng, window)
                preview = model.preview_state(batch['templates'], batch['main_preview'], batch['broad_preview'], history)
                qloss = F.smooth_l1_loss(preview['quality'], targets['quality'])
                gloss = batch['main'].new_zeros(())
                kinds = torch.full((batch['main'].shape[0], 1), kind, dtype=torch.long, device=batch['main'].device)
                for modality in range(2):
                    prediction = model.candidate_gains(targets['primaries'][modality], preview, modality,
                        batch['main_geometry'], batch['broad_geometry'], targets['queries'][modality], kinds)[:, 1]
                    gloss = gloss + F.smooth_l1_loss(prediction, targets['gains'][modality]) / 2
                loss = qloss + gloss
                step_optimizer(loss, model, optimizer, PREFIXES, args.sanity)
                step += 1
                row = dict(epoch=epoch, step=step, kind=kind, loss=loss.item(), quality_loss=qloss.item(), gain_loss=gloss.item())
                step_writer.writerow(row)
                for modality in range(2):
                    key = '%d:%d' % (modality, kind)
                    routes[key] = routes.get(key, 0) + batch['main'].shape[0]
                    query_ious = pixel_iou(targets['query_box'][modality], batch['target'])
                    for index, name in enumerate(batch['sequences']):
                        label_writer.writerow(dict(epoch=epoch, step=step, sequence=name, frame=batch['frames'][index],
                            modality=modality, kind=kind, candidate_rank=targets['ranks'][modality][index],
                            request_cell=targets['cells'][modality][index], family_candidate_count=targets['family_counts'][modality][index],
                            unique_candidate_count=targets['unique_counts'][modality][index],
                            main_geometry=json.dumps(batch['main_geometry'][index].cpu().tolist()),
                            query_geometry=json.dumps(targets['queries'][modality][index, 0].cpu().tolist()),
                            main_raw_box=json.dumps(targets['main_raw'][modality][index].cpu().tolist()),
                            query_raw_box=json.dumps(targets['query_raw'][modality][index].cpu().tolist()),
                            main_box=json.dumps(targets['main_box'][modality][index].cpu().tolist()),
                            query_box=json.dumps(targets['query_box'][modality][index].cpu().tolist()),
                            main_iou=float(targets['quality'][index, modality].item()), query_iou=float(query_ious[index].item()),
                            delta_from_stop=float(targets['gains'][modality][index].item())))
                if step == 1 or step % 25 == 0:
                    steps.flush()
                    labels.flush()
                    print(json.dumps(row), flush=True)
                probe = probe_inputs(batch, history)
            frozen, changed = save_checkpoint(model, optimizer, run, config, step, before, probe, window)
    assert step == (plan['sanity_updates'] if args.sanity else plan['expected_formal_updates'])
    assert len(seen) == (plan['sanity_expected_sequences'] if args.sanity else 882) and len(routes) == 8
    causal = causal_probe(model, candidate['max_candidates']) if args.sanity else None
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    assert sha('/home/gaob/CADTrack_ep0014.pth.tar') == plan['protected_checkpoint_sha256']
    complete = dict(status='COMPLETE', stage='scale_candidate_b', sanity=args.sanity,
        updates=step, samples=epochs * samples, labelled_queries=2 * epochs * samples,
        seen_train_sequences=sorted(seen), route_samples=routes, candidate_config=candidate,
        frozen_state_tensors=len(frozen), changed_state_tensors=changed,
        checkpoint_sha256=sha(run / 'latest.pth'), private_labels_sha256=sha(run / 'sampled_labels.csv'),
        causal_probe=causal, scope=plan['scope'])
    (run / 'COMPLETE.json').write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
