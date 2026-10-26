"""Fixed TRAIN inference work at 1/2/4/8 GPU0 processes; no model changes."""
import argparse
import cProfile
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pstats
import resource
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def gpu_sample():
    text = subprocess.check_output(['nvidia-smi', '-i', '0,1',
        '--query-gpu=index,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], universal_newlines=True)
    return dict(epoch=time.time(), gpus=[list(map(int, row.split(','))) for row in text.splitlines()])


def trajectory(tracker, sequence, frames, read_pair, np):
    initial = np.asarray(sequence['boxes'][0], dtype=np.float32)
    tracker.initialize(read_pair(sequence, 0), initial)
    boxes, routes = [initial.tolist()], []
    keys = ('modality', 'action', 'request_index', 'candidate_rank', 'candidate_count',
            'confidence', 'primary_tokens', 'auxiliary_tokens')
    for frame in range(1, frames):
        box, route = tracker.track(read_pair(sequence, frame))
        boxes.append(box.tolist())
        routes.append({key: route[key] for key in keys})
    state = hashlib.sha256()
    for value in (tracker.state, tracker.templates.detach().cpu().numpy(), tracker.history.detach().cpu().numpy()):
        state.update(value.tobytes())
    return dict(sequence=sequence['name'], predictions=boxes, routes=routes, final_state_sha256=state.hexdigest())


def worker(plan, assigned, barrier, ready, output, profile, cv_threads):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'scripts'))
    import cv2
    import numpy as np
    import torch
    from dual_rgbt.data import read_pair
    from run_benchmark_tracking import build_model, build_tracker, freeze
    from train_global_evidence_policy import cpu_state
    loaded = time.time()
    cv2.setNumThreads(cv_threads)
    assert cv2.getNumThreads() == cv_threads
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_model('e1', ROOT / plan['factory_checkpoint'])
    checkpoint = torch.load(ROOT / plan['checkpoint'], map_location='cpu')
    assert checkpoint['step'] == 182182 and checkpoint['stage'] == 'candidate_c'
    model.load_state_dict(checkpoint['net'], strict=True)
    del checkpoint
    freeze(model)
    before = cpu_state(model)
    tracker = build_tracker('e1', model, plan['cost_weight'], 8)
    with torch.no_grad():
        trajectory(tracker, assigned[0], 5, read_pair, np)
        torch.cuda.synchronize()
        ready.put(dict(pid=os.getpid(), allocated=torch.cuda.memory_allocated(),
                       reserved=torch.cuda.memory_reserved(), load_seconds=time.time()-loaded))
        barrier.wait(timeout=180)
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        profiler = cProfile.Profile() if profile else None
        if profiler:
            profiler.enable()
        results = [trajectory(tracker, row, plan['frames_per_sequence'], read_pair, np) for row in assigned]
        torch.cuda.synchronize()
        finished = time.time()
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(Path(output).with_suffix('.prof')))
        assert all(torch.equal(before[key], value) for key, value in cpu_state(model).items())
    report = dict(pid=os.getpid(), started=started, finished=finished, results=results,
                  total_parameters=sum(value.numel() for value in model.parameters()),
                  frames=len(assigned)*plan['frames_per_sequence'],
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  gpu_name=torch.cuda.get_device_name(), cv2_threads=cv2.getNumThreads(),
                  torch_threads=torch.get_num_threads(), no_backward_or_optimizer=True,
                  model_state_bit_equal=True, profile=profile)
    Path(output).write_text(json.dumps(report) + '\n')


def monitor(stop, samples):
    while not stop.is_set():
        samples.append(gpu_sample())
        stop.wait(1)


def run_arm(plan, rows, count, directory, cv_threads, profile=False):
    assert gpu_sample()['gpus'][0][1] < 500
    directory.mkdir()
    context = mp.get_context('spawn')
    barrier, ready = context.Barrier(count+1), context.Queue()
    stop, samples = threading.Event(), []
    sampler = threading.Thread(target=monitor, args=(stop, samples))
    sampler.start()
    processes = []
    began = time.time()
    for index in range(count):
        process = context.Process(target=worker, args=(plan, rows[index::count], barrier, ready,
            str(directory / ('worker%d.json' % index)), profile, cv_threads))
        process.start()
        processes.append(process)
    admissions = [ready.get(timeout=180) for _ in processes]
    resource_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    released = time.time()
    barrier.wait(timeout=180)
    for process in processes:
        process.join()
        assert process.exitcode == 0, (process.pid, process.exitcode)
    joined = time.time()
    resource_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    stop.set()
    sampler.join()
    assert samples
    reports = [json.loads((directory / ('worker%d.json' % index)).read_bytes()) for index in range(count)]
    assert len({row['total_parameters'] for row in reports}) == 1
    start = min(row['started'] for row in reports)
    end = max(row['finished'] for row in reports)
    frames = sum(row['frames'] for row in reports)
    timed_samples = [row for row in samples if start <= row['epoch'] <= end]
    assert timed_samples
    result = dict(processes=count, opencv_threads=cv_threads, profile=profile, admissions=admissions,
                  total_parameters=reports[0]['total_parameters'],
                  total_frames=frames, started=start, finished=end,
                  inference_wall_seconds=end-start, aggregate_fps=frames/(end-start),
                  parent_makespan_seconds=joined-released, end_to_end_seconds=joined-began,
                  end_to_end_fps=frames/(joined-began),
                  cpu_children_seconds=(resource_after.ru_utime+resource_after.ru_stime-
                                        resource_before.ru_utime-resource_before.ru_stime),
                  gpu0_peak_used_mib=max(row['gpus'][0][1] for row in samples),
                  gpu0_mean_utilization=sum(row['gpus'][0][3] for row in timed_samples)/len(timed_samples),
                  worker_exit_codes=[process.exitcode for process in processes],
                  worker_pids=[process.pid for process in processes], samples=samples)
    results = {item['sequence']: item for row in reports for item in row['results']}
    assert len(results) == len(rows)
    (directory / 'COMPLETE.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='samples'}), flush=True)
    return result, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_bytes())
    review = json.loads((ROOT / plan['source_review']).read_bytes())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(path)
    assert all(sha(ROOT / name) == value for name,value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name,value in plan['input_sha256'].items())
    assert args.task == plan['task'] and args.run == plan['run']
    native = json.loads((ROOT/'runs/pipeline_tasks'/args.task/'status.json').read_bytes())
    assert native['state']=='RUNNING' and native['pid']==os.getppid()
    assert native['physical_gpu']=='0' and native['command']==[sys.executable]+sys.argv
    terminal = json.loads((ROOT / plan['completed_dev_status']).read_bytes())
    assert terminal['state']=='COMPLETE' and terminal['exit_code']==0
    assert not (Path('/proc')/str(terminal['pid'])).exists()
    manifest = json.loads((ROOT / plan['manifest']).read_bytes())
    by_name = {row['name']:row for row in manifest['train']}
    rows = [by_name[name] for name in plan['sequence_names']]
    assert len(rows)==8 and all(len(row['rgb'])>=plan['frames_per_sequence'] for row in rows)
    output = ROOT/'profile_output'/args.run
    output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(json.dumps(dict(plan=plan,plan_sha256=sha(path),native=native),indent=2)+'\n')
    # Warm the same filesystem pages for every arm; decoding remains in measured work.
    for row in rows:
        for kind in ('rgb','tir'):
            for image in row[kind][:plan['frames_per_sequence']]:
                Path(image).read_bytes()
    arms = []
    reference = None
    for index,arm in enumerate(plan['arms']):
        count, cv_threads = arm['processes'], arm['opencv_threads']
        if arms:
            memory = gpu_sample()['gpus'][0]
            projected = max(row['gpu0_peak_used_mib']/row['processes'] for row in arms)*count
            assert projected < memory[1]+memory[2]-1024, ('Memory admission failed', count, projected)
        result, predictions = run_arm(plan,rows,count,output/('arm%d_n%d_cv%d'%(index,count,cv_threads)),cv_threads)
        if reference is None:
            reference = predictions
        assert predictions==reference, 'Predictions/routes/final state changed with concurrency.'
        result['serial_prediction_route_state_equal'] = True
        arms.append(result)
    profiled, hotspots = [], []
    for cv_threads in (80,1):
        directory = output/('cprofile_cv%d'%cv_threads)
        result, _ = run_arm(plan,rows[:1],1,directory,cv_threads,profile=True)
        profiled.append(result)
        stats = pstats.Stats(str(directory/'worker0.prof'))
        hotspots.append(dict(opencv_threads=cv_threads,functions=[
            dict(file=name[0],line=name[1],function=name[2],calls=value[1],
                 self_seconds=value[2],cumulative_seconds=value[3])
            for name,value in sorted(stats.stats.items(),key=lambda x:x[1][3],reverse=True)[:40]]))
    assert all(sha(ROOT/name)==value for name,value in plan['source_sha256'].items())
    assert all(sha(ROOT/name)==value for name,value in plan['input_sha256'].items())
    best = max(arms,key=lambda row:row['aggregate_fps'])
    report = dict(status='COMPLETE',arms=arms,hotspots=hotspots,profile_arms=profiled,
                  checkpoint_sha256=plan['checkpoint_sha256'],total_parameters=arms[0]['total_parameters'],
                  best_processes=best['processes'],best_opencv_threads=best['opencv_threads'],
                  no_model_or_method_change=True,
                  scope='Same8 TRAIN streams, fixed frame count and exact output/state parity. '
                        'Short throughput profile, not accuracy or full97 performance. '
                        'GPU1 continues dev; GPU0 only. Profiled arm excluded from speed comparison.')
    (output/'COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE',best_processes=report['best_processes'],
                          best_opencv_threads=report['best_opencv_threads'],
                          output=str(output))),flush=True)


if __name__=='__main__':
    main()
