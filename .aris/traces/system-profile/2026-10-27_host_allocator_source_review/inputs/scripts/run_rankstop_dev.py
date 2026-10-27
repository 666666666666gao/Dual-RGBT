"""Evaluate the completed Rank/STOP repair on train smoke or full LasHeR dev."""
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
from benchmark_inputs import load_manifest, sha
from dual_rgbt.data import read_pair
from run_benchmark_tracking import build_model, build_tracker, freeze, reload_probe
from train_global_evidence_policy import cpu_state


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--cost-weight', required=True, type=float)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_bytes())
    assert plan['status'] == 'READY'
    review = json.loads((ROOT / plan['source_review']).read_bytes())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(plan_path) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    assert sha(ROOT / 'checkpoints/CADTrack_ep0014.pth.tar') == plan['protected_checkpoint_sha256']
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1')
    native = json.loads((ROOT / 'runs/pipeline_tasks' / args.task / 'status.json').read_bytes())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['task'] == args.task and native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert native['command'] == [sys.executable] + sys.argv
    key = 'smoke' if args.smoke else str(args.cost_weight)
    registered = plan['runs'][key]
    assert (args.run, args.task, args.cost_weight) == (registered['run'], registered['task'], registered['cost_weight'])
    split = 'train_smoke' if args.smoke else 'dev'
    if not args.smoke:
        smoke = plan['runs']['smoke']
        admission = json.loads((ROOT / 'runs' / smoke['run'] / 'SMOKE.json').read_bytes())
        smoke_native = json.loads((ROOT / 'runs/pipeline_tasks' / smoke['task'] / 'status.json').read_bytes())
        assert smoke_native['state'] == 'COMPLETE' and smoke_native['exit_code'] == 0
        assert admission['checkpoint_sha256'] == plan['checkpoint_sha256'] and admission['plan_sha256'] == sha(plan_path)
        assert admission['reload']['status'] == 'PASS' and admission['smoke_only']
    audit = json.loads((ROOT / plan['training_audit']).read_bytes())
    assert audit['status'] == 'ARTIFACT_CHECKS_PASS_WITH_PROVENANCE_WARNING'
    assert audit['checkpoint_sha256'] == plan['checkpoint_sha256']
    assert audit['actual_process_exit_code'] is None
    assert audit['counts'] == dict(updates=182182, initializes=1764, tracked_frames=917106)
    checkpoint = ROOT / plan['checkpoint']
    state = torch.load(checkpoint, map_location='cpu')
    assert state['step'] == 182182 and state['stage'] == 'candidate_c' and not state['config']['sanity']
    assert state['config']['plan_sha256'] == plan['training_plan_sha256']
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    factory = ROOT / plan['factory_checkpoint']
    model = build_model('e1', factory)
    model.load_state_dict(state['net'], strict=True)
    del state
    freeze(model)
    before = cpu_state(model)
    tracker = build_tracker('e1', model, args.cost_weight, 8)
    manifest_path = ROOT / plan['manifests'][split]
    manifest = load_manifest(manifest_path)
    assert manifest['benchmark'] == 'LasHeR' and manifest['split'] == split
    expected_size = (1, 176) if args.smoke else (97, 54646)
    assert (manifest['sequence_count'], manifest['frame_count']) == expected_size
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    for name in ('predictions', 'groundtruth', 'traces'):
        (run / name).mkdir()
    config = dict(vars(args), split=split, smoke_frames=11 if args.smoke else 0,
                  checkpoint=plan['checkpoint'], checkpoint_sha256=plan['checkpoint_sha256'],
                  source_sha256=plan['source_sha256'], plan_sha256=sha(plan_path),
                  manifest_sha256=sha(manifest_path), native_command=native['command'],
                  physical_gpu=native['physical_gpu'], shard=0, shards=1,
                  total_parameters=sum(parameter.numel() for parameter in model.parameters()),
                  gpu_name=torch.cuda.get_device_name(),
                  input_sizes=dict(template=128, main_search=256, local_query=128, wide_query=256,
                                   main_preview=64, global_preview=256),
                  split_sequences=expected_size[0], split_frames=expected_size[1],
                  sequences=[row['name'] for row in manifest['sequences']],
                  training_provenance_warnings=audit['warnings'])
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    seconds, receipts = [], []
    torch.cuda.reset_peak_memory_stats()
    start_wall = time.perf_counter()
    for sequence in manifest['sequences']:
        name = sequence['name']
        count = 11 if args.smoke else len(sequence['rgb'])
        predictions = [np.asarray(sequence['initial_box'], dtype=np.float32)]
        started = time.perf_counter()
        tracker.initialize(read_pair(sequence, 0), sequence['initial_box'])
        seconds.append(time.perf_counter() - started)
        actions, modalities = np.zeros(5, dtype=np.int64), np.zeros(2, dtype=np.int64)
        with (run / 'traces' / (name + '.csv')).open('w', newline='') as handle:
            fields = ['frame', 'modality', 'action', 'confidence', 'seconds', 'seconds_including_decode',
                      'primary_tokens', 'auxiliary_tokens', 'request_index', 'candidate_rank', 'candidate_count']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for frame in range(1, count):
                started = time.perf_counter()
                image = read_pair(sequence, frame)
                if args.smoke and frame == count - 1:
                    saved = dict(state=tracker.state.copy(), templates=tracker.templates.clone(), history=tracker.history.clone(),
                                 window=tracker.window.clone(), frame=tracker.frame, since_query=tracker.since_query, image=image)
                prediction, route = tracker.track(image)
                elapsed = time.perf_counter() - started
                assert np.isfinite(prediction).all() and (prediction[2:] > 0).all()
                seconds.append(elapsed)
                predictions.append(prediction)
                actions[route['action']] += 1
                modalities[route['modality']] += 1
                writer.writerow(dict(frame=frame, seconds_including_decode=elapsed, **route))
        np.savetxt(run / 'predictions' / (name + '.txt'), predictions, fmt='%.6f', delimiter=',')
        np.savetxt(run / 'groundtruth' / (name + '.txt'), np.asarray(sequence['boxes'], dtype=np.float32)[:count], fmt='%.9g', delimiter=',')
        receipts.append(dict(sequence=name, frames=count, action_counts=actions.tolist(), modality_counts=modalities.tolist()))
        (run / 'progress.json').write_text(json.dumps(receipts, indent=2))
        print(json.dumps(receipts[-1]), flush=True)
    summary = dict(checkpoint_sha256=plan['checkpoint_sha256'], plan_sha256=sha(plan_path),
                   sequences=len(receipts), frames=len(seconds), smoke_only=args.smoke, task=args.task,
                   backward_calls=0, optimizer_updates=0, wall_seconds=time.perf_counter() - start_wall,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   training_provenance_warnings=audit['warnings'])
    if args.smoke:
        summary['reload'] = reload_probe('e1', model, checkpoint, factory, saved, prediction, route, tracker, args.cost_weight, 8)
    assert all(torch.equal(before[key], value) for key, value in cpu_state(model).items())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    timing = np.asarray(seconds)
    assert np.isfinite(timing).all() and (timing > 0).all()
    assert len(seconds) == (11 if args.smoke else 54646)
    summary.update(total_parameters=config['total_parameters'], gpu_name=config['gpu_name'],
                   input_sizes=config['input_sizes'], fps=len(timing) / float(timing.sum()),
                   mean_ms=1000 * float(timing.mean()), p95_ms=1000 * float(np.quantile(timing, .95)),
                   initialization_included=True, image_decoding_included=True, model_loading_excluded=True)
    np.save(run / 'frame_seconds.npy', timing)
    (run / ('SMOKE.json' if args.smoke else 'COMPLETE.json')).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
