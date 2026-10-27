"""Causal smoke/dev/test execution for one completed joint-controller checkpoint."""
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
from benchmark_inputs import groundtruth, load_manifest, sha
from dual_rgbt.data import read_pair
from joint_evaluation_contract import check_files, evaluation_command, evaluation_plan, run_name, selected_configuration, smoke_admission
from run_benchmark_tracking import build_model, build_tracker, freeze, reload_probe
from train_global_evidence_policy import cpu_state


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--benchmark', choices=('LasHeR', 'RGBT210', 'RGBT234'), required=True)
    parser.add_argument('--split', choices=('train_smoke', 'dev', 'test'), required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--cost-weight', type=float, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--smoke-frames', type=int, required=True)
    parser.add_argument('--selection-sha256')
    args = parser.parse_args()
    plan, spec, training = evaluation_plan(ROOT, args.plan)
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1', '2', '3')
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID' and os.environ['OPENCV_FOR_THREADS_NUM'] == '1'
    native = json.loads((ROOT / 'runs/pipeline_tasks' / args.task / 'status.json').read_bytes())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['task'] == args.task and native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert native['command'] == [sys.executable] + sys.argv == evaluation_command(vars(args))
    assert args.run == run_name(plan, args.benchmark, args.split, args.cost_weight, args.shard, args.shards)
    assert args.task == 'eval_' + args.run
    smoke = args.split == 'train_smoke'
    assert smoke == bool(args.smoke_frames)
    manifest_path = spec['datasets'][args.benchmark][args.split]
    manifest = load_manifest(ROOT / manifest_path)
    assert manifest['benchmark'] == args.benchmark and manifest['split'] == args.split
    size = spec['split_sizes'][args.benchmark][args.split]
    assert (manifest['sequence_count'], manifest['frame_count']) == (size['sequences'], size['frames'])
    assert 0 <= args.shard < args.shards <= len(manifest['sequences'])
    if smoke:
        assert args.benchmark == 'LasHeR' and args.smoke_frames == 11 and args.shards == 1
        assert args.cost_weight == training['cost_weight'] and args.selection_sha256 is None
    else:
        smoke_admission(ROOT, args.plan, plan, training)
        if args.split == 'dev':
            assert args.benchmark == 'LasHeR' and args.cost_weight in spec['dev_cost_candidates']
            assert args.selection_sha256 is None
        else:
            selection = selected_configuration(ROOT, args.plan, args.selection_sha256)
            assert selection['configurations'][args.benchmark] == dict(
                cost_weight=args.cost_weight, max_candidates=plan['max_candidates'])
    torch.manual_seed(spec['seed'])
    np.random.seed(spec['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    checkpoint, factory = ROOT / plan['checkpoint'], ROOT / plan['factory_checkpoint']
    state = torch.load(checkpoint, map_location='cpu')
    assert state['stage'] == 'candidate_c' and state['step'] == spec['formal_counts']['updates']
    assert not state['config']['sanity']
    model = build_model('e1', factory)
    model.load_state_dict(state['net'], strict=True)
    del state
    freeze(model)
    total_parameters = sum(p.numel() for p in model.parameters())
    assert total_parameters == spec['total_parameters']
    before = cpu_state(model)
    tracker = build_tracker('e1', model, args.cost_weight, plan['max_candidates'])
    directory = ROOT / 'runs' / args.run
    directory.mkdir(parents=True, exist_ok=False)
    for name in ('predictions', 'groundtruth', 'traces'):
        (directory / name).mkdir()
    sequences = manifest['sequences'][args.shard::args.shards]
    config = dict(vars(args), candidate_id=plan['candidate_id'], checkpoint=plan['checkpoint'],
        checkpoint_sha256=plan['checkpoint_sha256'], max_candidates=plan['max_candidates'],
        plan_sha256=sha(ROOT / args.plan), source_sha256=plan['source_sha256'],
        native_command=native['command'], physical_gpu=native['physical_gpu'],
        manifest_sha256=sha(ROOT / manifest_path), split_sequences=size['sequences'], split_frames=size['frames'],
        sequences=[row['name'] for row in sequences], total_parameters=total_parameters,
        gpu_name=torch.cuda.get_device_name(), input_sizes=dict(template=128, main_search=256,
        local_query=128, wide_query=256, main_preview=64, global_preview=256),
        annotation_source=manifest['annotation_source'], opencv_env_threads=1, torch_threads=1)
    (directory / 'config.json').write_text(json.dumps(config, indent=2))
    timings, receipts = [], []
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    for sequence in sequences:
        name = sequence['name']
        count = 11 if smoke else len(sequence['rgb'])
        predictions = [np.asarray(sequence['initial_box'], dtype=np.float32)]
        started = time.perf_counter()
        tracker.initialize(read_pair(sequence, 0), sequence['initial_box'])
        init_seconds = time.perf_counter() - started
        timings.append(init_seconds)
        actions, modalities = np.zeros(5, dtype=np.int64), np.zeros(2, dtype=np.int64)
        tokens = 0
        with (directory / 'traces' / (name + '.csv')).open('w', newline='') as handle:
            fields = ['frame', 'modality', 'action', 'confidence', 'seconds', 'seconds_including_decode',
                      'primary_tokens', 'auxiliary_tokens', 'request_index', 'candidate_rank', 'candidate_count']
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            for frame in range(1, count):
                started = time.perf_counter()
                image = read_pair(sequence, frame)
                if smoke and frame == count - 1:
                    saved = dict(state=tracker.state.copy(), templates=tracker.templates.clone(),
                        history=tracker.history.clone(), window=tracker.window.clone(), frame=tracker.frame,
                        since_query=tracker.since_query, image=image)
                prediction, route = tracker.track(image)
                elapsed = time.perf_counter() - started
                assert np.isfinite(prediction).all() and (prediction[2:] > 0).all()
                predictions.append(prediction)
                timings.append(elapsed)
                actions[route['action']] += 1
                modalities[route['modality']] += 1
                tokens += route['primary_tokens'] + route['auxiliary_tokens']
                writer.writerow(dict(frame=frame, seconds_including_decode=elapsed, **route))
        np.savetxt(directory / 'predictions' / (name + '.txt'), predictions, fmt='%.6f', delimiter=',')
        visible, infrared = groundtruth(sequence, args.benchmark)
        if args.benchmark == 'LasHeR':
            np.savetxt(directory / 'groundtruth' / (name + '.txt'), visible[:count], fmt='%.9g', delimiter=',')
        else:
            folder = directory / 'groundtruth' / name
            folder.mkdir()
            for modality, gt in (('visible', visible), ('infrared', infrared)):
                np.savetxt(folder / (modality + '.txt'), gt[:count], fmt='%.9g', delimiter=',')
        receipts.append(dict(sequence=name, frames=count, initialized_once=True, initialization_seconds=init_seconds,
            action_counts=actions.tolist(), modality_counts=modalities.tolist(), heavy_token_sum=tokens))
        (directory / 'progress.json').write_text(json.dumps(receipts, indent=2))
        print(json.dumps(receipts[-1]), flush=True)
    assert all(torch.equal(before[name], value) for name, value in cpu_state(model).items())
    assert all(p.grad is None for p in model.parameters())
    summary = dict(task=args.task, checkpoint_sha256=plan['checkpoint_sha256'], plan_sha256=sha(ROOT / args.plan),
        sequences=len(receipts), frames=len(timings), smoke_only=smoke, backward_calls=0, optimizer_updates=0,
        frozen_state_tensors=len(before),
        total_parameters=total_parameters, gpu_name=config['gpu_name'], input_sizes=config['input_sizes'],
        wall_seconds=time.perf_counter()-wall_started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), image_decoding_included=True,
        initialization_included=True, model_loading_excluded=True, service_excludes_file_writes_and_verification=True)
    if smoke:
        summary['reload'] = reload_probe('e1', model, checkpoint, factory, saved, prediction, route,
                                         tracker, args.cost_weight, plan['max_candidates'])
    timing = np.asarray(timings)
    expected_frames = 11 if smoke else sum(len(row['rgb']) for row in sequences)
    assert len(timing) == expected_frames and np.isfinite(timing).all() and (timing > 0).all()
    summary.update(seconds=float(timing.sum()), fps=len(timing)/float(timing.sum()),
        mean_ms=1000*float(timing.mean()), p95_ms=1000*float(np.quantile(timing, .95)))
    check_files(ROOT, plan['source_sha256'])
    check_files(ROOT, plan['input_sha256'])
    np.save(directory / 'frame_seconds.npy', timing)
    (directory / ('SMOKE.json' if smoke else 'COMPLETE.json')).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
