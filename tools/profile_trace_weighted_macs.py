"""Profile convolution/matrix MACs by the observed dev route/shape mixture."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import load_manifest, sha
from dual_rgbt.data import read_pair
from joint_evaluation_contract import check_files, evaluation_plan
from run_benchmark_tracking import build_model, build_tracker, freeze
from train_global_evidence_policy import cpu_state

MAC_EVENTS = {'aten::conv2d', 'aten::mm', 'aten::bmm', 'aten::addmm'}
POINTWISE_EVENTS = {'aten::add', 'aten::mul'}


def counts(profiler):
    events = {e.key: float(e.flops) for e in profiler.key_averages() if e.flops}
    assert set(events) <= MAC_EVENTS | POINTWISE_EVENTS, events
    return dict(conv_matrix_macs=sum(v for k, v in events.items() if k in MAC_EVENTS) / 2,
        supported_arithmetic_flops=sum(events.values()), event_flops=events)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan_digest = sha(plan_path)
    plan = json.loads(plan_path.read_text())
    assert plan['mode'] == 'trace_weighted_macs_only' and not plan['accuracy_evaluation']
    assert sha(Path(__file__)) == plan['profiler_sha256']
    parent, spec, _ = evaluation_plan(ROOT, plan['parent_evaluation_plan'])
    native_path = ROOT / 'runs/pipeline_tasks' / args.task / 'status.json'
    native = json.loads(native_path.read_text())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['command'] == [sys.executable] + sys.argv
    assert native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID' and os.environ['OPENCV_FOR_THREADS_NUM'] == '1'
    report_path = ROOT / plan['benchmark_report']
    assert sha(report_path) == plan['benchmark_report_sha256']
    report = json.loads(report_path.read_text())
    assert report['status'] == 'SCORED_COMPLETE' and report['split'] == 'dev'
    assert report['checkpoint_sha256'] == parent['checkpoint_sha256']
    assert report['cost_weight'] == .15 and report['max_candidates'] == parent['max_candidates'] == 8
    assert (report['sequences'], report['frames']) == (97, 54646)
    manifest = load_manifest(ROOT / spec['datasets']['LasHeR']['dev'])
    assert sha(ROOT / spec['datasets']['LasHeR']['dev']) == report['manifest_sha256']
    mixture, used, seen = Counter(), {}, []
    for run in report['run_names']:
        directory = ROOT / 'runs' / run
        cfg_path = directory / 'config.json'
        assert sha(cfg_path) == report['run_artifacts_sha256'][cfg_path.relative_to(ROOT).as_posix()]
        cfg = json.loads(cfg_path.read_text())
        for name in cfg['sequences']:
            seen.append(name)
            path = directory / 'traces' / (name + '.csv')
            key = path.relative_to(ROOT).as_posix()
            used[key] = sha(path)
            assert used[key] == report['run_artifacts_sha256'][key]
            with path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert [int(row['frame']) for row in rows] == list(range(1, len(rows)+1))
            mixture.update((int(row['modality']), int(row['action']), int(row['candidate_count'])) for row in rows)
    assert len(seen) == len(set(seen)) == 97 and set(seen) == {s['name'] for s in manifest['sequences']}
    assert sum(mixture.values()) == 54549
    torch.manual_seed(2027)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_model('e1', ROOT / parent['factory_checkpoint'])
    model.load_state_dict(torch.load(ROOT / parent['checkpoint'], map_location='cpu')['net'], strict=True)
    freeze(model)
    assert sum(p.numel() for p in model.parameters()) == report['total_parameters'] == spec['total_parameters']
    before = cpu_state(model)
    original_preview, original_gains = model.preview_state, model.candidate_gains
    settings = {}

    def forced_preview(*values):
        result = original_preview(*values)
        result['quality'].fill_(0)
        result['quality'][0, settings['modality']] = 1
        return result

    def forced_gains(primary, preview, modality, main, broad, queries, kinds):
        result = original_gains(primary, preview, modality, main, broad, queries, kinds)
        slot = 0 if settings['action'] == 0 else int((kinds[0] == settings['action']-1).nonzero()[0, 0].item()) + 1
        result.fill_(-1e9)
        result[0, slot] = 1e9
        return result

    model.preview_state, model.candidate_gains = forced_preview, forced_gains
    sequence = manifest['sequences'][0]
    first, image = read_pair(sequence, 0), read_pair(sequence, 1)
    tracker = build_tracker('e1', model, .15, 8)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
        tracker.initialize(first, sequence['initial_box'])
    initialization = counts(prof)
    measured = []
    for (modality, action, candidates), frames in sorted(mixture.items()):
        settings.update(modality=modality, action=action)
        tracker = build_tracker('e1', model, .15, candidates)
        tracker.initialize(first, sequence['initial_box'])
        tracker.track(image)
        tracker = build_tracker('e1', model, .15, candidates)
        tracker.initialize(first, sequence['initial_box'])
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
            _, route = tracker.track(image)
        assert (route['modality'], route['action'], route['candidate_count']) == (modality, action, candidates)
        assert route['auxiliary_tokens'] == (0 if action == 0 else 385 if action == 4 else 193)
        measured.append(dict(modality=modality, action=action, candidate_count=candidates, frames=frames, **counts(prof)))
        print(json.dumps(measured[-1]), flush=True)
    model.preview_state, model.candidate_gains = original_preview, original_gains
    assert all(torch.equal(before[k], v) for k, v in cpu_state(model).items())
    assert all(p.grad is None for p in model.parameters())
    check_files(ROOT, parent['source_sha256'])
    check_files(ROOT, used)
    assert sha(report_path) == plan['benchmark_report_sha256'] and sha(Path(__file__)) == plan['profiler_sha256']
    assert sha(plan_path) == plan_digest
    total = sum(row['frames'] * row['conv_matrix_macs'] for row in measured)
    result = dict(status='PROFILE_COMPLETE', task=args.task, native_command=native['command'],
        plan_sha256=plan_digest, benchmark_report_sha256=sha(report_path),
        checkpoint_sha256=parent['checkpoint_sha256'], source_sha256=dict(parent['source_sha256'], **{Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))}),
        used_trace_sha256=used, total_parameters=sum(p.numel() for p in model.parameters()),
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(), physical_gpu=native['physical_gpu'],
        trace_tracking_frames=54549, initialization_frames=97, measured_cases=measured,
        initialization=initialization, mean_tracking_macs=total/54549,
        mean_macs_including_initialization=(total+97*initialization['conv_matrix_macs'])/54646,
        mean_supported_tracking_flops=sum(row['frames']*row['supported_arithmetic_flops'] for row in measured)/54549,
        optimizer_updates=0, backward_calls=0,
        scope='Shape-weighted estimate from all observed (modality,action,candidate_count) groups. Forced diagnostic routes retain original preview/gain computation. Reduced maximum K reproduces observed candidate-count shapes, not original accuracy trajectories. MACs count profiler-supported convolution and matrix-multiply events; pointwise arithmetic is separate. Normalization, softmax, sampling, decoding and other unsupported operations are not MAC-counted. No FPS or accuracy measurement.')
    output = ROOT / args.output
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('total_parameters', 'mean_tracking_macs', 'mean_macs_including_initialization')}), flush=True)


if __name__ == '__main__':
    main()
