"""Measure independent native controller training replicas beside the owned GPU1 run."""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi', '-i', '1',
        '--query-gpu=memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()


def training_observation(plan):
    result = []
    for item in plan['coexisting_runs']:
        native = json.loads((ROOT / 'runs/pipeline_tasks' / item['task'] / 'status.json').read_bytes())
        assert native['state'] == 'RUNNING' and native['pid'] == item['wrapper_pid']
        with (ROOT / 'runs' / item['run'] / 'trajectory_steps.csv').open(newline='') as handle:
            rows = [row for row in csv.DictReader(handle) if None not in row.values()]
        assert rows and int(rows[-1]['step']) == len(rows)
        result.append(dict(task=item['task'], gpu=item['gpu'], logged_updates=len(rows)))
    return dict(observed=time.time(), runs=result)


def worker(plan, arm, index, release, finish, messages):
    import math
    import numpy as np
    import torch
    from dual_rgbt.data import read_pair
    from dual_rgbt_global.candidate_policy import build_candidate_policy
    from dual_rgbt_global.candidate_tracker import CandidateEvidenceTracker
    from train_candidate_evidence_policy import sampled_targets
    from train_candidate_joint_gainpair_trajectory import PREFIXES, causal_batch, selected_target, controller_loss
    from train_global_evidence_policy import cpu_state, step_optimizer

    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    assert os.environ['OPENCV_FOR_THREADS_NUM'] == '1'
    base = json.loads((ROOT / plan['training_plan']).read_bytes())
    torch.manual_seed(base['seed'])
    np.random.seed(base['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_candidate_policy(ROOT / base['factory_checkpoint']).cuda().eval()
    checkpoint = torch.load(ROOT / base['checkpoint'], map_location='cpu')
    assert checkpoint['stage'] == 'candidate_b' and checkpoint['step'] == 6250
    model.load_state_dict(checkpoint['net'], strict=True)
    initial_net = checkpoint['net']
    del checkpoint
    assert sum(p.numel() for p in model.parameters()) == 103218887
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(PREFIXES))
    before = cpu_state(model)
    manifest = json.loads((ROOT / 'data/manifest.json').read_bytes())
    by_name = {row['name']: row for row in manifest['train']}
    directory = ROOT / plan['output'] / arm
    torch.cuda.synchronize()
    messages.put(dict(state='READY', index=index, pid=os.getpid()))
    release.wait()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    streams = []
    for stream_index in range(index, len(plan['sequence_names']), plan['current_processes']):
        sequence = by_name[plan['sequence_names'][stream_index]]
        model.load_state_dict(initial_net, strict=True)
        optimizer = torch.optim.AdamW([
            dict(params=[p for n, p in model.named_parameters() if n.startswith('executor.preview.')],
                 lr=base['preview_learning_rate']),
            dict(params=[p for n, p in model.named_parameters() if n.startswith(('executor.quality.', 'query_gain.'))],
                 lr=base['learning_rate'])], weight_decay=base['weight_decay'])
        tracker = CandidateEvidenceTracker(model, base['cost_weight'], base['max_candidates'])
        rng = np.random.RandomState(base['seed'])
        tracker.initialize(read_pair(sequence, 0), sequence['boxes'][0])
        valid = set(sequence['valid_frames'])
        rows = []
        for frame in range(1, len(sequence['boxes'])):
            image = read_pair(sequence, frame)
            previous, templates, history = tracker.state.copy(), tracker.templates.clone(), tracker.history.clone()
            executed, route = tracker.track(image)
            if frame % base['label_stride'] or frame not in valid:
                continue
            batch = causal_batch(image, previous, templates, sequence['boxes'][frame])
            state_after, templates_after, history_after = tracker.state.copy(), tracker.templates.clone(), tracker.history.clone()
            kind = len(rows) % 4
            targets = sampled_targets(model, batch, history, kind, base['max_candidates'], rng, tracker.window)
            preview = model.preview_state(templates, batch['main_preview'], batch['broad_preview'], history)
            selected = selected_target(model, batch, preview, targets, kind, route, executed,
                                       base['cost_weight'], base['max_candidates'])
            assert np.array_equal(tracker.state, state_after)
            assert torch.equal(tracker.templates, templates_after) and torch.equal(tracker.history, history_after)
            losses = controller_loss(model, batch, preview, targets, kind, selected, route['modality'])
            step_optimizer(losses[0], model, optimizer, PREFIXES, False)
            numbers = [float(v.item()) for v in losses]
            assert all(math.isfinite(v) for v in numbers)
            rows.append(dict(frame=frame, box=executed.tolist(), route=route, losses=numbers))
            if len(rows) == plan['updates_per_stream']:
                break
        assert len(rows) == plan['updates_per_stream']
        payload = json.dumps(rows, sort_keys=True).encode()
        (directory / ('stream%d_steps.json' % stream_index)).write_bytes(payload)
        streams.append(dict(index=stream_index, sequence=sequence['name'], updates=len(rows),
            tracked_frames=rows[-1]['frame'], steps_sha256=hashlib.sha256(payload).hexdigest(),
            prediction_route_sha256=hashlib.sha256(json.dumps([
                dict(frame=row['frame'], box=row['box'], route={k: v for k, v in row['route'].items() if k != 'seconds'})
                for row in rows], sort_keys=True).encode()).hexdigest()))
    torch.cuda.synchronize()
    finished = time.perf_counter()
    seconds = finished - started
    after = cpu_state(model)
    frozen = [n for n in before if not n.startswith(PREFIXES)]
    changed = [n for n in before if not torch.equal(before[n], after[n])]
    assert len(frozen) == 251 and all(torch.equal(before[n], after[n]) for n in frozen)
    assert len(changed) == 14 and all(n.startswith(PREFIXES) for n in changed)
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith(PREFIXES))
    assert {int(v['step'].item()) for v in optimizer.state.values()} == {len(rows)}
    result = dict(state='RESULT', index=index, pid=os.getpid(), updates=sum(v['updates'] for v in streams),
        tracked_frames=sum(v['tracked_frames'] for v in streams), initializes=len(streams), training_seconds=seconds,
        training_finished_perf_counter=finished, streams=streams,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        frozen_state_tensors=len(frozen), changed_state_tensors=changed, total_parameters=103218887)
    (directory / ('worker%d.json' % index)).write_text(json.dumps(result, indent=2))
    messages.put(result)
    finish.wait()


def run_arm(plan, arm, count):
    directory = ROOT / plan['output'] / arm
    directory.mkdir(parents=True, exist_ok=False)
    ctx = mp.get_context('spawn')
    release, finish, messages = ctx.Event(), ctx.Event(), ctx.Queue()
    processes = []
    whole_started = time.perf_counter()
    try:
        for index in range(count):
            p = ctx.Process(target=worker, args=(dict(plan, current_processes=count), arm, index, release, finish, messages))
            p.start()
            processes.append(p)
        ready = [messages.get(timeout=180) for _ in processes]
        assert all(v['state'] == 'READY' for v in ready) and {v['index'] for v in ready} == set(range(count))
        assert {v['pid'] for v in ready} == {p.pid for p in processes}
        before = gpu_snapshot()
        concurrent_before = training_observation(plan)
        started = time.perf_counter()
        release.set()
        results = [messages.get(timeout=900) for _ in processes]
        work_and_validation_seconds = time.perf_counter() - started
        assert all(v['state'] == 'RESULT' for v in results)
        assert {v['index'] for v in results} == set(range(count))
        after = gpu_snapshot()
        concurrent_after = training_observation(plan)
    finally:
        # Prior scaling R1 actually left owned workers alive after a failure.
        # Release both barriers and reap only processes created by this arm.
        release.set()
        finish.set()
        for p in processes:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join()
        messages.close()
        messages.join_thread()
    assert all(p.exitcode == 0 for p in processes)
    assert sorted(v['index'] for w in results for v in w['streams']) == list(range(len(plan['sequence_names'])))
    wall_seconds = max(v['training_finished_perf_counter'] for v in results) - started
    result = dict(arm=arm, processes=count, total_updates=sum(v['updates'] for v in results),
        training_wall_seconds=wall_seconds,
        training_and_validation_wall_seconds=work_and_validation_seconds,
        aggregate_updates_per_second=sum(v['updates'] for v in results) / wall_seconds,
        aggregate_training_track_frames_per_second=sum(v['tracked_frames'] for v in results) / wall_seconds,
        spawn_to_join_seconds=time.perf_counter()-whole_started,
        gpu_before_release_csv=before, gpu_after_work_csv=after,
        coexisting_training_before=concurrent_before, coexisting_training_after=concurrent_after,
        child_exit_codes=[p.exitcode for p in processes], workers=sorted(results, key=lambda v: v['index']))
    (directory / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_bytes())
    review = json.loads((ROOT / plan['source_review']).read_bytes())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan) and review['source_sha256'] == plan['source_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1' and os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert os.environ['OPENCV_FOR_THREADS_NUM'] == '1'
    assert all(sha(ROOT / k) == v for k, v in plan['source_sha256'].items())
    assert all(sha(ROOT / k) == v for k, v in plan['input_sha256'].items())
    native = json.loads((ROOT / 'runs/pipeline_tasks' / plan['task'] / 'status.json').read_bytes())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['physical_gpu'] == '1' and native['command'] == [sys.executable] + sys.argv
    occupied = plan['coexisting_native_status']
    current = json.loads((ROOT / occupied).read_bytes())
    assert current['state'] == 'RUNNING' and current['pid'] == plan['coexisting_wrapper_pid']
    children = (Path('/proc') / str(current['pid']) / 'task' / str(current['pid']) / 'children').read_text().split()
    assert children == [str(plan['coexisting_child_pid'])]
    assert (Path('/proc') / children[0] / 'cmdline').read_bytes().decode().split(chr(0))[:-1] == current['command']
    manifest = json.loads((ROOT / 'data/manifest.json').read_bytes())
    base = json.loads((ROOT / plan['training_plan']).read_bytes())
    assert len(plan['sequence_names']) == len(set(plan['sequence_names'])) == 8
    assert set(plan['sequence_names']).issubset(base['sequence_names'])
    by_name = {row['name']: row for row in manifest['train']}
    assert all(sum(f > 0 and f % base['label_stride'] == 0 for f in by_name[name]['valid_frames']) >= plan['updates_per_stream'] for name in plan['sequence_names'])
    output = ROOT / plan['output']
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(dict(plan=plan, plan_sha256=sha(ROOT / args.plan)), indent=2))
    results = [run_arm(plan, item['name'], item['processes']) for item in plan['arms']]
    assert all(sha(ROOT / k) == v for k, v in plan['source_sha256'].items())
    assert all(sha(ROOT / k) == v for k, v in plan['input_sha256'].items())
    result = dict(status='COMPLETE', total_parameters=103218887, arms=results,
        saved_checkpoints=0, source_sha256=plan['source_sha256'], input_sha256=plan['input_sha256'],
        scope=plan['scope'])
    (output / 'COMPLETE.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
