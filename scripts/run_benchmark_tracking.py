"""Complete E0/E1 causal tracking on registered inputs; score separately."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import REMOTE_PYTHON, REMOTE_ROOT, evaluation_command, frozen_selection, groundtruth, load_manifest, sha
from dual_rgbt.data import read_pair
from train_global_evidence_policy import cpu_state


def build_model(variant, factory):
    if variant == 'e0':
        import dual_rgbt.data as data
        from profile_crop_api import six_channel_crop
        data.crop_pair = six_channel_crop
        from dual_rgbt.model import DualRGBT
        return DualRGBT().cuda().eval()
    from dual_rgbt_global.candidate_policy import build_candidate_policy
    return build_candidate_policy(factory).cuda().eval()


def build_tracker(variant, model, cost, candidates):
    if variant == 'e0':
        from dual_rgbt.tracker import Tracker
        return Tracker(model, cost)
    from dual_rgbt_global.candidate_tracker import CandidateEvidenceTracker
    return CandidateEvidenceTracker(model, cost, candidates)


def freeze(model):
    model.eval()
    model.requires_grad_(False)


def trained_checkpoint(plan, variant, smoke, checkpoint_sha):
    identity = plan['models'][variant]
    path = ROOT / identity['smoke_checkpoint' if smoke else 'formal_checkpoint']
    assert sha(path) == checkpoint_sha
    if smoke:
        assert checkpoint_sha == identity['smoke_checkpoint_sha256']
    state = torch.load(path, map_location='cpu')
    if not smoke:
        native = json.loads((ROOT / identity['formal_native_status']).read_bytes())
        complete = json.loads((path.parent / 'COMPLETE.json').read_bytes())
        config = json.loads((path.parent / 'config.json').read_bytes())
        training = json.loads((ROOT / identity['training_plan']).read_bytes())
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
        assert native['task'] == Path(identity['formal_native_status']).parent.name
        assert native['physical_gpu'] == ('0' if variant == 'e0' else '1')
        assert state['step'] == plan['formal_counts']['updates']
        assert not state['config']['sanity']
        assert state['config']['run'] == str(path.parent.relative_to(ROOT / 'runs')).replace('\\', '/')
        assert len(training['sequence_names']) == len(set(training['sequence_names'])) == 882
        if variant == 'e0':
            integrity = json.loads((ROOT / identity['training_integrity']).read_bytes())
            assert integrity['status'] == 'PASS' and integrity['actual_wrapper_exit_code'] == 0
            assert integrity['checkpoint_sha256'] == checkpoint_sha
            assert integrity['steps'] == state['step'] and integrity['tracked_frames'] == plan['formal_counts']['tracked_frames']
            assert integrity['sequences_per_round'] == 882 and integrity['exact_supervision_keys'] and integrity['all_losses_finite']
            assert all(sha(ROOT / name) == digest for name, digest in integrity['input_sha256'].items())
            assert integrity['input_sha256'][identity['training_plan']] == plan['input_sha256'][identity['training_plan']]
            assert native['command'] == [REMOTE_PYTHON, 'scripts/run_with_crop_api.py', 'train_trajectory_policy.py',
                '--policy', identity['factory_checkpoint'], '--sequences', '882', '--rounds', '2',
                '--label-stride', '5', '--cost-weight', '0.05', '--seed', str(plan['seed']), '--run', state['config']['run']]
            assert {key: value for key, value in config.items() if key != 'sequence_names'} == state['config']
            assert config['sequence_names'] == training['sequence_names']
            assert state['round'] == complete['rounds'] == 2
            assert complete['sequences_per_round'] == state['config']['sequences'] == 882
            assert complete['steps'] == state['step'] and complete['tracked_frames'] == plan['formal_counts']['tracked_frames']
            assert state['config']['label_stride'] == 5 and state['config']['seed'] == plan['seed']
            assert state['config']['rounds'] == 2 and state['config']['cost_weight'] == .05
            assert state['config']['policy'] == identity['factory_checkpoint']
        else:
            assert native['command'] == [REMOTE_PYTHON, REMOTE_ROOT + '/scripts/train_candidate_trajectory_policy.py',
                '--plan', identity['training_plan'], '--run', state['config']['run'],
                '--sanity-complete-sha256', identity['sanity_complete_sha256']]
            assert config == state['config']
            assert complete['status'] == 'COMPLETE'
            assert state['stage'] == state['config']['stage'] == 'candidate_c'
            assert state['config']['plan_sha256'] == sha(ROOT / identity['training_plan'])
            assert state['config']['source_sha256'] == training['source_sha256']
            assert state['config']['input_sha256'] == training['input_sha256']
            assert state['config']['factory_checkpoint'] == identity['factory_checkpoint'] == training['factory_checkpoint']
            assert state['config']['max_candidates'] == training['max_candidates'] == 8
            assert state['config']['sanity_complete_sha256'] == identity['sanity_complete_sha256']
            assert {key: complete[key] for key in plan['formal_counts']} == plan['formal_counts']
            assert not complete['sanity'] and complete['checkpoint_sha256'] == checkpoint_sha
            assert len(complete['seen_train_sequences']) == len(set(complete['seen_train_sequences'])) == 882
            assert set(complete['seen_train_sequences']) == set(training['sequence_names'])
    elif variant == 'e1':
        assert state['stage'] == 'candidate_b' and state['step'] == 6250 and not state['config']['sanity']
    return path, state


@torch.no_grad()
def reload_probe(variant, model, checkpoint, factory, saved, box, route, expected, cost, candidates):
    clone = build_model(variant, factory)
    clone.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
    freeze(clone)
    tracker = build_tracker(variant, clone, cost, candidates)
    tracker.state = saved['state'].copy()
    tracker.templates = saved['templates'].clone()
    tracker.history = saved['history'].clone()
    window = 'output_window' if variant == 'e0' else 'window'
    setattr(tracker, window, saved['window'].clone())
    tracker.frame, tracker.since_query = saved['frame'], saved['since_query']
    actual_box, actual_route = tracker.track(saved['image'])
    fields = tuple(key for key in route if key != 'seconds')
    assert np.array_equal(actual_box, box)
    assert all(actual_route[key] == route[key] for key in fields)
    assert torch.equal(tracker.history, expected.history) and torch.equal(tracker.templates, expected.templates)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in clone.parameters())
    return dict(status='PASS', strict_reload_box_route_bit_equal=True, route_fields=list(fields),
                strict_reload_templates_history_bit_equal=True, optimizer_updates=0, backward_calls=0)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--variant', choices=('e0', 'e1'), required=True)
    parser.add_argument('--benchmark', choices=('LasHeR', 'RGBT210', 'RGBT234'), required=True)
    parser.add_argument('--split', choices=('train_smoke', 'dev', 'test'), required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--cost-weight', type=float, required=True)
    parser.add_argument('--max-candidates', type=int, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--smoke-frames', type=int, default=0)
    parser.add_argument('--selection-sha256')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1') and os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert 0 <= args.shard < args.shards and args.cost_weight >= 0
    assert args.max_candidates == 0 if args.variant == 'e0' else 1 <= args.max_candidates <= 8
    smoke = bool(args.smoke_frames)
    assert (args.split == 'train_smoke') == smoke
    assert not smoke or (args.benchmark == 'LasHeR' and args.smoke_frames == 11 and args.shards == 1)
    plan = json.loads((ROOT / args.plan).read_bytes())
    assert plan['status'] == 'READY'
    for review_path in (plan['source_review'], plan['metric_review']):
        review = json.loads((ROOT / review_path).read_bytes())
        assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
        assert review['plan_sha256'] == sha(ROOT / args.plan) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    assert sha('/home/gaob/CADTrack_ep0014.pth.tar') == plan['protected_checkpoint_sha256']
    task_path = ROOT / 'runs/pipeline_tasks' / args.task / 'status.json'
    native = json.loads(task_path.read_bytes())
    assert native['task'] == args.task
    assert native['state'] == 'RUNNING' and native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert native['pid'] == os.getppid() and native['command'] == [sys.executable] + sys.argv
    assert native['command'] == evaluation_command(vars(args))
    manifest_path = ROOT / plan['datasets'][args.benchmark][args.split]
    assert sha(manifest_path) == plan['input_sha256'][str(manifest_path.relative_to(ROOT)).replace('\\', '/')]
    manifest = load_manifest(manifest_path)
    assert manifest['benchmark'] == args.benchmark and manifest['split'] == args.split
    entire = manifest['sequences']
    expected_size = plan['split_sizes'][args.benchmark][args.split]
    assert (manifest['sequence_count'], manifest['frame_count']) == (expected_size['sequences'], expected_size['frames'])
    assert smoke or args.split != 'dev' or args.benchmark == 'LasHeR'
    checkpoint, state = trained_checkpoint(plan, args.variant, smoke, args.checkpoint_sha256)
    frozen_sha = None
    if args.split == 'test':
        selection = frozen_selection(ROOT, plan, args.variant, args.checkpoint_sha256, sha(ROOT / args.plan), args.selection_sha256)
        assert selection['configurations'][args.benchmark] == dict(cost_weight=args.cost_weight, max_candidates=args.max_candidates)
        frozen_sha = args.selection_sha256
    elif args.split == 'dev':
        assert args.selection_sha256 is None
        assert args.cost_weight in plan['dev_cost_candidates'] and args.max_candidates == (0 if args.variant == 'e0' else 8)
    else:
        assert args.selection_sha256 is None
        assert args.cost_weight == .05 and args.max_candidates == (0 if args.variant == 'e0' else 8)
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    identity = plan['models'][args.variant]
    factory = ROOT / identity['factory_checkpoint']
    model = build_model(args.variant, factory)
    model.load_state_dict(state['net'], strict=True)
    del state
    freeze(model)
    before = cpu_state(model)
    tracker = build_tracker(args.variant, model, args.cost_weight, args.max_candidates)
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    for name in ('predictions', 'groundtruth', 'traces'):
        (run / name).mkdir()
    selected = entire[args.shard::args.shards]
    config = dict(vars(args), checkpoint=str(checkpoint.relative_to(ROOT)),
                  native_command=native['command'],
                  physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'], plan_sha256=sha(ROOT / args.plan),
                  source_sha256=plan['source_sha256'], manifest_sha256=sha(manifest_path),
                  split_sequences=len(entire), split_frames=manifest['frame_count'], sequences=[row['name'] for row in selected],
                  frozen_selection_sha256=frozen_sha, total_parameters=sum(parameter.numel() for parameter in model.parameters()),
                  annotation_source=manifest['annotation_source'], image_decoding_included=True)
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    seconds, receipts = [], []
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    for sequence in selected:
        name = sequence['name']
        count = min(len(sequence['rgb']), args.smoke_frames) if smoke else len(sequence['rgb'])
        predictions = [np.asarray(sequence['initial_box'], dtype=np.float32)]
        started = time.perf_counter()
        image = read_pair(sequence, 0)
        initialize_seconds = tracker.initialize(image, sequence['initial_box'])
        initialize_with_decode = time.perf_counter() - started
        seconds.append(initialize_with_decode)
        actions, modalities = np.zeros(5, dtype=np.int64), np.zeros(2, dtype=np.int64)
        tokens, online = 0, []
        with (run / 'traces' / (name + '.csv')).open('w', newline='') as handle:
            fields = ['frame', 'modality', 'action', 'confidence', 'seconds', 'seconds_including_decode', 'primary_tokens', 'auxiliary_tokens',
                      'primary_confidence', 'query_confidence', 'query_selected', 'template_updated',
                      'primary_box_x', 'primary_box_y', 'primary_box_w', 'primary_box_h',
                      'query_box_x', 'query_box_y', 'query_box_w', 'query_box_h',
                      'final_box_x', 'final_box_y', 'final_box_w', 'final_box_h']
            if args.variant == 'e1':
                fields += ['request_index', 'candidate_rank', 'candidate_count']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for frame in range(1, count):
                started = time.perf_counter()
                image = read_pair(sequence, frame)
                if smoke and frame == count - 1:
                    window = tracker.output_window if args.variant == 'e0' else tracker.window
                    saved = dict(state=tracker.state.copy(), templates=tracker.templates.clone(), history=tracker.history.clone(),
                                 window=window.clone(), frame=tracker.frame, since_query=tracker.since_query, image=image)
                prediction, route = tracker.track(image)
                elapsed = time.perf_counter() - started
                assert np.isfinite(prediction).all() and (prediction[2:] > 0).all()
                predictions.append(prediction)
                online.append(elapsed)
                actions[route['action']] += 1
                modalities[route['modality']] += 1
                tokens += route['primary_tokens'] + route['auxiliary_tokens']
                writer.writerow(dict(frame=frame, seconds_including_decode=elapsed, **route))
        np.savetxt(run / 'predictions' / (name + '.txt'), predictions, fmt='%.6f', delimiter=',')
        visible, infrared = groundtruth(sequence, args.benchmark)
        if args.benchmark == 'LasHeR':
            np.savetxt(run / 'groundtruth' / (name + '.txt'), visible[:count], fmt='%.9g', delimiter=',')
        else:
            folder = run / 'groundtruth' / name
            folder.mkdir()
            for modality, values in (('visible', visible), ('infrared', infrared)):
                np.savetxt(folder / (modality + '.txt'), values[:count], fmt='%.9g', delimiter=',')
        seconds.extend(online)
        receipt = dict(sequence=name, frames=count, initialized_once=True, initialization_seconds=initialize_seconds,
                       initialization_seconds_including_decode=initialize_with_decode, online_seconds_including_decode=sum(online),
                       action_counts=actions.tolist(), modality_counts=modalities.tolist(), mean_heavy_tokens=tokens / (count - 1))
        receipts.append(receipt)
        (run / 'progress.json').write_text(json.dumps(receipts, indent=2))
        print(json.dumps(receipt), flush=True)
    wall_seconds = time.perf_counter() - wall_start
    after = cpu_state(model)
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(parameter.grad is None for parameter in model.parameters())
    if smoke:
        sanity = reload_probe(args.variant, model, checkpoint, factory, saved, prediction, route, tracker, args.cost_weight, args.max_candidates)
        sanity.update(frozen_state_tensors=len(before), tracked_frames=sum(row['frames'] - 1 for row in receipts))
        (run / 'SANITY.json').write_text(json.dumps(sanity, indent=2))
    assert sha(checkpoint) == args.checkpoint_sha256
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    if frozen_sha is not None:
        assert sha(ROOT / identity['frozen_selection']) == frozen_sha
    timing = np.asarray(seconds)
    assert np.isfinite(timing).all() and (timing > 0).all()
    summary = dict(sequences=len(receipts), frames=sum(row['frames'] for row in receipts), smoke_only=smoke,
                   seconds=float(timing.sum()), wall_seconds=wall_seconds, mean_ms=1000 * float(timing.mean()),
                   p95_ms=1000 * float(np.quantile(timing, .95)), fps=len(timing) / float(timing.sum()),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   image_decoding_included=True, initialization_included=True, model_loading_excluded=True,
                   trace_prediction_writing_excluded_from_service_time=True, checkpoint_verification_excluded_from_service_time=True,
                   frozen_state_tensors=len(before), optimizer_updates=0, backward_calls=0, task=args.task,
                   scope='Complete causal predictions require separate toolkit scoring; train smoke is execution only. Service timing excludes load, file writes and verification.')
    np.save(run / 'frame_seconds.npy', timing)
    (run / ('SMOKE.json' if smoke else 'COMPLETE.json')).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
