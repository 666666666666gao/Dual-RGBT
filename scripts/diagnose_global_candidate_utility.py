"""Execute candidate queries on unchanged reference train trajectories."""
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
from diagnose_global_candidates import predictions as candidate_predictions
from diagnose_global_opportunity import PrimaryOnlyTracker, region_coverage, sha
from dual_rgbt.data import read_pair
from dual_rgbt_evidence.readout import pixel_xywh
from dual_rgbt_evidence.regions import crop_region, execute_boxes, pixel_iou, search_geometry
from dual_rgbt_global.policy import build_policy, global_geometry
from dual_rgbt_global.training import query_pixels
from train_global_evidence_policy import cpu_state


def aggregate(rows):
    by_state = {}
    for row in rows:
        by_state.setdefault(row['state'], []).append(row)
    results = {}
    for scope in ('all', 'main_centre_outside'):
        states = [group for group in by_state.values() if scope == 'all' or not group[0]['main_centre_inside']]
        summary = dict(states=len(states), by_k={})
        if states:
            summary['mean_main_iou'] = float(np.mean([group[0]['main_iou'] for group in states]))
            for k in (1, 2, 4, 8):
                summary['by_k'][str(k)] = {}
                for region in ('GLOBAL_LOCAL', 'GLOBAL_WIDE', 'ALL_ACTIONS'):
                    groups = [[r for r in group if (region == 'ALL_ACTIONS' or r['region'] == region) and
                               r['candidate_rank'] < k] for group in states]
                    best = [max(group, key=lambda r: r['iou']) for group in groups]
                    coverage_choice = [max(group, key=lambda r: r['query_gt_coverage']) for group in groups]
                    gains = np.asarray([r['delta_from_stop'] for r in best])
                    summary['by_k'][str(k)][region] = dict(
                        mean_best_query_iou=float(np.mean([r['iou'] for r in best])),
                        mean_best_signed_gain=float(gains.mean()),
                        mean_stop_inclusive_oracle_gain=float(np.maximum(gains, 0).mean()),
                        positive=int((gains > 0).sum()), negative=int((gains < 0).sum()), zero=int((gains == 0).sum()),
                        mean_coverage_choice_iou=float(np.mean([r['iou'] for r in coverage_choice])),
                        coverage_choice_equals_best=sum((a['region'], a['candidate_rank']) == (b['region'], b['candidate_rank'])
                                                        for a, b in zip(coverage_choice, best)))
        results[scope] = summary
    return results


@torch.no_grad()
def probe_predictions(model, inputs):
    query = model.query(inputs['templates'], inputs['pixels'], inputs['primary'],
                        inputs['main_geometry'], inputs['query_geometry'], inputs['modality'])
    candidates = candidate_predictions(model, inputs['templates'][:, 0], inputs['broad'], 1 - inputs['modality'],
                                       inputs['broad_geometry'], inputs['sizes'], inputs['local_side'])
    result = {'query_' + field: query[field] for field in ('box', 'pixel_box', 'score', 'size', 'offset')}
    result.update({'candidate_' + field: candidates[field] for field in ('scores', 'indices', 'points')})
    return result


@torch.no_grad()
def diagnose(reference, fitted, plan, run, sanity):
    sequences = {r['name']: r for r in json.loads((ROOT / 'data/manifest.json').read_text())['train']}
    with (ROOT / plan['content_csv']).open(newline='') as handle:
        previous = [r for r in csv.DictReader(handle) if r['region'] == 'GLOBAL_WIDE' and r['content'] == 'TRUE']
    saved = json.loads((ROOT / plan['candidates_json']).read_text())
    assert len(previous) == len(saved) == 1752
    names = plan['sequence_names'][:1] if sanity else plan['sequence_names']
    tracker = PrimaryOnlyTracker(reference, 0.)
    rows, states, frames = [], 0, 0
    fields = ('sequence', 'frame', 'state', 'modality', 'region', 'candidate_rank', 'candidate_count',
              'auxiliary_tokens', 'main_centre_inside', 'query_centre_inside', 'query_gt_coverage',
              'main_iou', 'iou', 'delta_from_stop', 'main_geometry', 'main_box', 'query_geometry', 'raw_box', 'executed_box')
    with (run / 'query_utilities.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in names:
            sequence = sequences[name]
            boxes = np.asarray(sequence['boxes'], dtype=np.float32)
            valid = set(sequence['valid_frames'])
            tracker.initialize(read_pair(sequence, 0), boxes[0])
            stop = min(len(boxes), plan['sanity_frames'] + 1) if sanity else len(boxes)
            for frame in range(1, stop):
                image = read_pair(sequence, frame)
                templates, history, previous_box = tracker.templates.clone(), tracker.history.clone(), tracker.state.copy()
                actual, modality = tracker.track(image)
                frames += 1
                if frame % plan['label_stride'] or frame not in valid:
                    continue
                old, candidates_before = previous[states], saved[states]
                assert (old['sequence'], int(old['frame']), int(old['state'])) == (name, frame, states)
                assert modality == int(old['modality']) and candidates_before['auxiliary_modality'] == 1 - modality
                means = image.mean((0, 1))
                geometry, broad_geometry = search_geometry(previous_box), global_geometry(image)
                main = crop_region(image, geometry, 256, means)[None].cuda()
                main_preview = crop_region(image, geometry, 64, means)[None].cuda()
                broad = crop_region(image, broad_geometry, 256, means)[None].cuda()
                main_geo, broad_geo = main.new_tensor(geometry)[None], main.new_tensor(broad_geometry)[None]
                sizes = main.new_tensor([image.shape[:2]])
                primary = reference.executor.primary(templates, main, modality, tracker.window)
                factual = execute_boxes(pixel_xywh(primary['box'], main_geo), sizes)
                assert np.array_equal(factual[0].cpu().numpy(), actual)
                local_side = float(geometry[2] / 2)
                predicted = candidate_predictions(fitted, templates[:, 0], broad, 1 - modality, broad_geo, sizes, local_side)
                points = predicted['points']
                assert np.array_equal(points.cpu().numpy(), np.asarray(candidates_before['points'], dtype=np.float32))
                assert len(points) == candidates_before['candidate_count']
                assert geometry[2] == np.asarray(json.loads(old['query_geometry']), dtype=np.float32)[2]
                preview = reference.preview_state(templates, main_preview, broad, history)
                main_queries = reference.query_geometries(primary, preview, modality, main_geo, broad_geo, sizes)[:, :2]
                regions = [('LOCAL_MAIN', -1, 1, main_queries[:, 0]), ('LOCAL_ALTERNATE', -1, 2, main_queries[:, 1])]
                for rank, point in enumerate(points):
                    for region, side, action in (('GLOBAL_LOCAL', main_geo[:, 2:3] / 2, 3), ('GLOBAL_WIDE', main_geo[:, 2:3], 4)):
                        regions.append((region, rank, action, torch.cat([point[None] - side / 2, side], dim=1)))
                # GT supervises executed boxes only after candidate collection.
                target = main.new_tensor(boxes[frame])[None]
                main_iou = float(pixel_iou(factual, target).item())
                assert main_iou == float(old['main_iou'])
                main_inside, _ = region_coverage(boxes[frame], geometry)
                assert main_inside == (old['main_centre_inside'] == 'True')
                for region, rank, action, query_geo in regions:
                    pixels = query_pixels([image], [means], query_geo, action)
                    output = fitted.query(templates, pixels, primary, main_geo, query_geo, modality)
                    executed = execute_boxes(output['pixel_box'], sizes)
                    value = float(pixel_iou(executed, target).item())
                    inside, coverage = region_coverage(boxes[frame], query_geo[0].cpu().numpy())
                    assert output['auxiliary_tokens'] == (385 if action == 4 else 193)
                    row = dict(sequence=name, frame=frame, state=states, modality=modality, region=region,
                        candidate_rank=rank, candidate_count=len(points), auxiliary_tokens=output['auxiliary_tokens'],
                        main_centre_inside=main_inside, query_centre_inside=inside, query_gt_coverage=coverage,
                        main_iou=main_iou, iou=value, delta_from_stop=value - main_iou,
                        main_geometry=json.dumps(geometry.tolist()), main_box=json.dumps(factual[0].cpu().tolist()),
                        query_geometry=json.dumps(query_geo[0].cpu().tolist()), raw_box=json.dumps(output['pixel_box'][0].cpu().tolist()),
                        executed_box=json.dumps(executed[0].cpu().tolist()))
                    assert np.isfinite([main_iou, value, coverage]).all()
                    writer.writerow(row)
                    rows.append(row)
                probe = dict(templates=templates, pixels=pixels, primary=primary, main_geometry=main_geo,
                    query_geometry=query_geo, modality=modality, broad=broad, broad_geometry=broad_geo,
                    sizes=sizes, local_side=local_side)
                states += 1
                if states == 1 or states % 25 == 0:
                    handle.flush()
                    print(json.dumps(dict(tracked_frames=frames, labelled_states=states, auxiliary_queries=len(rows))), flush=True)
    assert states == (2 if sanity else 1752) and frames == (10 if sanity else 8854)
    expected_queries = sum(2 + 2 * r['candidate_count'] for r in saved[:states])
    assert len(rows) == expected_queries
    return dict(sequences=len(names), tracked_frames=frames, labelled_states=states,
        diagnostic_primary_calls=states, auxiliary_query_calls=len(rows),
        auxiliary_token_sum=sum(r['auxiliary_tokens'] for r in rows), per_scope=aggregate(rows)), probe


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
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / n) == v for n, v in plan['source_sha256'].items())
    assert all(sha(ROOT / n) == v for n, v in plan['input_sha256'].items())
    protected = Path('/home/gaob/CADTrack_ep0014.pth.tar')
    assert sha(protected) == plan['protected_checkpoint_sha256']
    assert args.run == plan['runs']['sanity' if args.sanity else 'formal']
    if args.sanity:
        assert args.sanity_complete_sha256 is None
    else:
        sanity_run = ROOT / 'runs' / plan['runs']['sanity']
        assert sha(sanity_run / 'COMPLETE.json') == args.sanity_complete_sha256
        native = json.loads((ROOT / 'runs/pipeline_tasks' / ('diagnose_' + plan['runs']['sanity']) / 'status.json').read_text())
        complete = json.loads((sanity_run / 'COMPLETE.json').read_text())
        config = json.loads((sanity_run / 'config.json').read_text())
        reload = json.loads((sanity_run / 'SANITY.json').read_text())
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0 and native['physical_gpu'] == '1'
        assert config['sanity'] and config['plan_sha256'] == sha(ROOT / args.plan) and config['source_sha256'] == plan['source_sha256']
        assert config['input_sha256'] == plan['input_sha256']
        assert complete['status'] == 'COMPLETE'
        assert complete['sanity'] and complete['labelled_states'] == 2 and complete['tracked_frames'] == 10
        assert complete['reference_frozen_tensors'] == complete['fitted_frozen_tensors'] == 265
        assert complete['optimizer_updates'] == complete['backward_calls'] == 0
        assert reload['status'] == 'PASS' and reload['strict_reload_bit_equal'] and len(reload['probe_fields']) == 8
    for stage in ('reference', 'fitted'):
        native = json.loads((ROOT / plan[stage + '_native_status']).read_text())
        complete = json.loads((ROOT / plan[stage + '_completion']).read_text())
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
        assert complete['updates'] == 6250 and not complete['sanity']
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    models, before = {}, {}
    for stage, stage_name, plan_hash in (('reference', 'b', plan['reference_plan_sha256']),
                                        ('fitted', 'localize', plan['fitted_plan_sha256'])):
        model = build_policy(ROOT / plan['factory_checkpoint']).cuda().eval()
        state = torch.load(ROOT / plan[stage + '_checkpoint'], map_location='cpu')
        assert state['stage'] == stage_name and state['step'] == 6250 and not state['config']['sanity']
        assert state['config']['plan_sha256'] == plan_hash
        model.load_state_dict(state['net'], strict=True)
        del state
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[stage], before[stage] = model, cpu_state(model)
    changed = [n for n in before['reference'] if not torch.equal(before['reference'][n], before['fitted'][n])]
    assert sorted(changed) == sorted(plan['preview_changed_tensors']) and len(changed) == 6
    assert all(n.startswith('executor.preview.net.') for n in changed)
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), source_sha256=plan['source_sha256'], plan_sha256=sha(ROOT / args.plan),
                  input_sha256=plan['input_sha256'], scope=plan['scope'])
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    result, probe = diagnose(models['reference'], models['fitted'], plan, run, args.sanity)
    if args.sanity:
        expected = probe_predictions(models['fitted'], probe)
        clone = build_policy(ROOT / plan['factory_checkpoint']).cuda().eval()
        clone.load_state_dict(torch.load(ROOT / plan['fitted_checkpoint'], map_location='cpu')['net'], strict=True)
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
        actual = probe_predictions(clone, probe)
        assert all(torch.equal(expected[n], actual[n]) for n in expected)
        (run / 'SANITY.json').write_text(json.dumps(dict(status='PASS', strict_reload_bit_equal=True,
            probe_fields=list(expected), probe_auxiliary_query_calls=2), indent=2))
    for stage in models:
        assert all(torch.equal(before[stage][n], v) for n, v in cpu_state(models[stage]).items())
        assert all(p.grad is None for p in models[stage].parameters())
    assert all(sha(ROOT / n) == v for n, v in plan['source_sha256'].items())
    assert all(sha(ROOT / n) == v for n, v in plan['input_sha256'].items())
    assert sha(protected) == plan['protected_checkpoint_sha256']
    result.update(status='COMPLETE', sanity=args.sanity, reference_frozen_tensors=len(before['reference']),
        fitted_frozen_tensors=len(before['fitted']), shared_equal_tensors=259, preview_changed_tensors=changed,
        optimizer_updates=0, backward_calls=0, raw_csv_sha256=sha(run / 'query_utilities.csv'),
        sanity_probe_auxiliary_query_calls=2 if args.sanity else 0, scope=plan['scope'])
    (run / 'COMPLETE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
