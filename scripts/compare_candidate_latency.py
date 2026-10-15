"""Paired native CADTrack/E1 service timing; no accuracy scoring or training."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import frozen_selection, load_manifest, sha
from dual_rgbt.data import read_pair
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.test.tracker.cadtrack import CADTrack
from lib.test.utils import TrackerParams
from run_benchmark_tracking import build_model, build_tracker, freeze, trained_checkpoint, reload_probe
from train_global_evidence_policy import cpu_state


def timing_command(values):
    command = ['/home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python',
               '/home/gaob/Dual-RGBT/scripts/compare_candidate_latency.py']
    for name in ('plan', 'checkpoint_sha256', 'selection_sha256', 'task', 'run', 'mode'):
        command += ['--' + name.replace('_', '-'), str(values[name])]
    return command


def statistics(rows, arm, clock):
    seconds = np.asarray([row[arm + '_' + clock + '_seconds'] for row in rows])
    assert len(seconds) and np.isfinite(seconds).all() and (seconds > 0).all()
    return dict(frames=len(seconds), seconds=float(seconds.sum()),
                mean_ms=1000 * float(seconds.mean()), p95_ms=1000 * float(np.quantile(seconds, .95)),
                fps=len(seconds) / float(seconds.sum()))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'checkpoint-sha256', 'selection-sha256', 'task', 'run'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--mode', choices=('smoke', 'full'), required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0' and os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    plan = json.loads((ROOT / args.plan).read_bytes())
    review = json.loads((ROOT / plan['source_review']).read_bytes())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == digest for name, digest in plan['input_sha256'].items())
    evaluation = json.loads((ROOT / plan['evaluation_plan']).read_bytes())
    native = json.loads((ROOT / 'runs/pipeline_tasks' / args.task / 'status.json').read_bytes())
    assert native['state'] == 'RUNNING' and native['physical_gpu'] == '0' and native['task'] == args.task
    assert native['pid'] == os.getppid() and native['command'] == [sys.executable] + sys.argv == timing_command(vars(args))
    checkpoint, state = trained_checkpoint(evaluation, 'e1', False, args.checkpoint_sha256)
    selection = frozen_selection(ROOT, evaluation, 'e1', args.checkpoint_sha256,
                                 sha(ROOT / plan['evaluation_plan']), args.selection_sha256)
    chosen = selection['configurations']['LasHeR']
    manifest_path = ROOT / evaluation['datasets']['LasHeR']['dev']
    manifest = load_manifest(manifest_path)
    assert (manifest['benchmark'], manifest['split'], manifest['sequence_count'], manifest['frame_count']) == ('LasHeR', 'dev', 97, 54646)
    sequences = manifest['sequences'][:1] if args.mode == 'smoke' else manifest['sequences']
    counts = [16 if args.mode == 'smoke' else len(row['rgb']) for row in sequences]
    assert all(len(row['rgb']) >= count > 9 for row, count in zip(sequences, counts))
    torch.manual_seed(evaluation['seed'])
    np.random.seed(evaluation['seed'])
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    baseline_path = Path('/home/gaob/CADTrack_ep0014.pth.tar')
    assert sha(baseline_path) == evaluation['protected_checkpoint_sha256']
    update_config_from_file(str(ROOT / 'experiments/cadtrack/cadtrack.yaml'))
    params = TrackerParams()
    params.cfg, params.checkpoint = cfg, str(baseline_path)
    params.template_factor, params.template_size = cfg.TEST.TEMPLATE_FACTOR, cfg.TEST.TEMPLATE_SIZE
    params.search_factor, params.search_size = cfg.TEST.SEARCH_FACTOR, cfg.TEST.SEARCH_SIZE
    params.save_all_boxes, params.debug = False, 0
    baseline = CADTrack(params)
    freeze(baseline.network)
    factory = ROOT / evaluation['models']['e1']['factory_checkpoint']
    model = build_model('e1', factory)
    model.load_state_dict(state['net'], strict=True)
    del state
    freeze(model)
    ours = build_tracker('e1', model, chosen['cost_weight'], chosen['max_candidates'])
    trackers = dict(baseline=baseline, e1=ours)
    networks = dict(baseline=baseline.network, e1=model)
    before = {name: cpu_state(network) for name, network in networks.items()}
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), native_command=native['command'], physical_gpu='0',
                  plan_sha256=sha(ROOT / args.plan), source_sha256=plan['source_sha256'],
                  checkpoint=str(checkpoint.relative_to(ROOT)), selection=chosen,
                  manifest_sha256=sha(manifest_path), sequences=[row['name'] for row in sequences],
                  frames_per_sequence=dict(zip([row['name'] for row in sequences], counts)),
                  baseline_checkpoint_sha256=sha(baseline_path),
                  parameters={name: sum(p.numel() for p in network.parameters()) for name, network in networks.items()},
                  gpu_name=torch.cuda.get_device_name(0), torch_threads=torch.get_num_threads(),
                  opencv_threads=cv2.getNumThreads(), both_models_resident=True)
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    records, progress = [], []
    fields = ['sequence', 'frame', 'first', 'baseline_service_seconds', 'baseline_tracking_seconds',
              'e1_service_seconds', 'e1_tracking_seconds', 'action', 'modality',
              'primary_tokens', 'auxiliary_tokens', 'request_index', 'candidate_rank']
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    with (run / 'timings.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sequence_index, (sequence, count) in enumerate(zip(sequences, counts)):
            initial_box = np.asarray(sequence['initial_box'], dtype=np.float32)
            for frame in range(count):
                order = ['baseline', 'e1'] if (sequence_index + frame) % 2 == 0 else ['e1', 'baseline']
                row = dict(sequence=sequence['name'], frame=frame, first=order[0])
                for name in order:
                    tracker = trackers[name]
                    if args.mode == 'smoke' and name == 'e1' and frame == count - 1:
                        saved = dict(state=tracker.state.copy(), templates=tracker.templates.clone(),
                                     history=tracker.history.clone(), window=tracker.window.clone(),
                                     frame=tracker.frame, since_query=tracker.since_query)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    image = read_pair(sequence, frame)
                    decoded = time.perf_counter()
                    if frame == 0:
                        tracker.initialize(image, {'init_bbox': initial_box.tolist()} if name == 'baseline' else initial_box)
                        box = initial_box
                    else:
                        result = tracker.track(image)
                        if name == 'baseline':
                            box = np.asarray(result['target_bbox'], dtype=np.float32)
                        else:
                            box, route = result
                            for key in ('action', 'modality', 'primary_tokens', 'auxiliary_tokens', 'request_index', 'candidate_rank'):
                                row[key] = route[key]
                    torch.cuda.synchronize()
                    finished = time.perf_counter()
                    row[name + '_service_seconds'] = finished - started
                    row[name + '_tracking_seconds'] = finished - decoded
                    assert np.isfinite(box).all() and (box[2:] > 0).all()
                    if args.mode == 'smoke' and name == 'e1' and frame == count - 1:
                        saved['image'], last_box, last_route = image, box, route
                writer.writerow(row)
                records.append(row)
            progress.append(dict(sequence=sequence['name'], frames=count))
            (run / 'progress.json').write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress[-1]), flush=True)
    wall_seconds = time.perf_counter() - wall_start
    peaks = dict(allocated_bytes=torch.cuda.max_memory_allocated(), reserved_bytes=torch.cuda.max_memory_reserved())
    for name, network in networks.items():
        after = cpu_state(network)
        assert all(torch.equal(before[name][key], after[key]) for key in before[name])
        assert all(parameter.grad is None for parameter in network.parameters())
    if args.mode == 'smoke':
        sanity = reload_probe('e1', model, checkpoint, factory, saved, last_box, last_route, ours,
                              chosen['cost_weight'], chosen['max_candidates'])
        (run / 'SANITY.json').write_text(json.dumps(sanity, indent=2))
    assert sha(checkpoint) == args.checkpoint_sha256
    assert sha(ROOT / evaluation['models']['e1']['frozen_selection']) == args.selection_sha256
    assert sha(baseline_path) == evaluation['protected_checkpoint_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == digest for name, digest in plan['input_sha256'].items())
    warm = [row for row in records if row['frame'] >= 9]
    online = [row for row in records if row['frame'] > 0]
    summary = dict(status='COMPLETE', task=args.task, smoke_only=args.mode == 'smoke',
                   sequences=len(sequences), frames=len(records), warm_frames=len(warm), tracked_frames=len(online),
                   full={name: {clock: statistics(records, name, clock) for clock in ('service', 'tracking')} for name in trackers},
                   warm={name: {clock: statistics(warm, name, clock) for clock in ('service', 'tracking')} for name in trackers},
                   action_counts=[sum(row['action'] == a for row in online) for a in range(5)],
                   modality_counts=[sum(row['modality'] == m for row in online) for m in range(2)],
                   query_frames=sum(row['action'] != 0 for row in online),
                   heavy_tokens=sum(row['primary_tokens'] + row['auxiliary_tokens'] for row in online),
                   shared_process_peak=peaks, memory_scope='Both resident models and their states; not isolated per-model VRAM.',
                   wall_seconds=wall_seconds, frozen_state_equal=True, optimizer_updates=0, backward_calls=0,
                   accuracy_evaluated=False, future_gt_read_by_tracker=False, initialization_included=True,
                   service_includes_decode=True, tracking_excludes_decode=True,
                   loading_file_writes_verification_excluded_from_service=True,
                   timings_sha256=sha(run / 'timings.csv'), config_sha256=sha(run / 'config.json'),
                   scope='Paired causal LasHeR dev timing only. Alternating order, shared file cache and two resident models remain measurement limitations.')
    assert summary['frames'] == sum(counts) and summary['warm_frames'] == sum(count - 9 for count in counts)
    (run / 'COMPLETE.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
