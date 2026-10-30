"""Dev-only causal ablation: retain initial templates with unchanged weights."""
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
from joint_evaluation_contract import check_files, evaluation_plan
from run_benchmark_tracking import build_model, build_tracker, freeze
from train_global_evidence_policy import cpu_state
from query_evidence_capture import FIELDS, QueryEvidenceCapture


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--smoke-frames', type=int, choices=(0, 31), default=0)
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text())
    assert plan['mode'] == 'static_template_diagnostic' and plan['test_eligible'] is False
    check_files(ROOT, plan['source_sha256'])
    parent, spec, _ = evaluation_plan(ROOT, plan['parent_evaluation_plan'])
    assert plan['checkpoint_sha256'] == plan['parent_checkpoint_sha256'] == parent['checkpoint_sha256']
    assert plan['checkpoint'] == parent['checkpoint']
    reference = json.loads((ROOT / 'runs' / plan['reference_run'] / 'config.json').read_text())
    assert reference['checkpoint_sha256'] == parent['checkpoint_sha256']
    assert reference['cost_weight'] == plan['cost_weight'] == .15
    assert reference['max_candidates'] == parent['max_candidates'] == 8
    native = json.loads((ROOT / 'runs/pipeline_tasks' / args.task / 'status.json').read_text())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['command'] == [sys.executable] + sys.argv
    assert native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID' and os.environ['OPENCV_FOR_THREADS_NUM'] == '1'
    manifest_path = ROOT / spec['datasets']['LasHeR']['dev']
    assert sha(manifest_path) == reference['manifest_sha256']
    manifest = load_manifest(manifest_path)
    assert (manifest['sequence_count'], manifest['frame_count']) == (97, 54646)
    sequences = manifest['sequences'][:1] if args.smoke_frames else manifest['sequences']
    torch.manual_seed(spec['seed'])
    np.random.seed(spec['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_model('e1', ROOT / parent['factory_checkpoint'])
    checkpoint = ROOT / parent['checkpoint']
    assert sha(checkpoint) == parent['checkpoint_sha256']
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
    freeze(model)
    before = cpu_state(model)
    tracker = build_tracker('e1', model, plan['cost_weight'], parent['max_candidates'])
    evidence = QueryEvidenceCapture(model, plan['cost_weight'])
    directory = ROOT / 'runs' / args.run
    directory.mkdir(parents=True, exist_ok=False)
    for name in ('predictions', 'groundtruth', 'traces', 'evidence'):
        (directory / name).mkdir()
    config = dict(vars(args), benchmark='LasHeR', split='dev', shard=0, shards=1,
        checkpoint=parent['checkpoint'], checkpoint_sha256=parent['checkpoint_sha256'],
        parent_checkpoint_sha256=parent['checkpoint_sha256'], plan_sha256=sha(ROOT / args.plan),
        cost_weight=plan['cost_weight'], max_candidates=parent['max_candidates'],
        source_sha256=dict(parent['source_sha256'], **plan['source_sha256']),
        manifest_sha256=sha(manifest_path), native_command=native['command'],
        split_sequences=97, split_frames=54646, sequences=[s['name'] for s in sequences],
        total_parameters=sum(p.numel() for p in model.parameters()),
        evaluation_class=plan['mode'], test_eligible=False,
        evidence_schema='primary_executed_box_and_selected_gain_v1',
        intervention='Restore both initial templates after every frame; leave predicted box/history/query decisions causal',
        fps_caveat='Concurrent load; template crops are still computed then discarded. Not an optimized speed benchmark.')
    assert config['total_parameters'] == spec['total_parameters']
    (directory / 'config.json').write_text(json.dumps(config, indent=2))
    timings, receipts = [], []
    wall_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    suppressed = 0
    for sequence in sequences:
        name = sequence['name']
        count = args.smoke_frames or len(sequence['rgb'])
        predictions = [np.asarray(sequence['initial_box'], dtype=np.float32)]
        started = time.perf_counter()
        tracker.initialize(read_pair(sequence, 0), sequence['initial_box'])
        timings.append(time.perf_counter() - started)
        initial_templates = tracker.templates.clone()
        sequence_suppressed = 0
        with (directory / 'traces' / (name + '.csv')).open('w', newline='') as handle, \
                (directory / 'evidence' / (name + '.csv')).open('w', newline='') as evidence_handle:
            fields = ['frame', 'modality', 'action', 'confidence', 'seconds', 'seconds_including_decode',
                'primary_tokens', 'auxiliary_tokens', 'request_index', 'candidate_rank', 'candidate_count',
                'committed_template_update']
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            evidence_writer = csv.DictWriter(evidence_handle, FIELDS)
            evidence_writer.writeheader()
            for frame in range(1, count):
                started = time.perf_counter()
                image = read_pair(sequence, frame)
                evidence.begin(image, tracker.state)
                prediction, route = tracker.track(image)
                sequence_suppressed += int(not torch.equal(tracker.templates, initial_templates))
                tracker.templates.copy_(initial_templates)
                assert torch.equal(tracker.templates, initial_templates)
                elapsed = time.perf_counter() - started
                evidence_writer.writerow(evidence.finish(frame, route, prediction))
                assert np.isfinite(prediction).all() and (prediction[2:] > 0).all()
                predictions.append(prediction)
                timings.append(elapsed)
                writer.writerow(dict(frame=frame, seconds_including_decode=elapsed, committed_template_update=0, **route))
        np.savetxt(directory / 'predictions' / (name + '.txt'), predictions, fmt='%.6f', delimiter=',')
        visible, _ = groundtruth(sequence, 'LasHeR')
        np.savetxt(directory / 'groundtruth' / (name + '.txt'), visible[:count], fmt='%.9g', delimiter=',')
        suppressed += sequence_suppressed
        receipts.append(dict(sequence=name, frames=count, initialized_once=True,
            templates_fixed=True, template_mutations_reverted=sequence_suppressed))
        (directory / 'progress.json').write_text(json.dumps(receipts, indent=2))
        print(json.dumps(receipts[-1]), flush=True)
    assert all(torch.equal(before[k], v) for k, v in cpu_state(model).items())
    assert all(p.grad is None for p in model.parameters())
    if args.smoke_frames:
        assert suppressed > 0, 'Smoke must actually exercise a template mutation'
    timing = np.asarray(timings)
    assert len(timing) == (args.smoke_frames or 54646)
    assert np.isfinite(timing).all() and (timing > 0).all()
    summary = dict(task=args.task, sequences=len(receipts), frames=len(timing),
        checkpoint_sha256=parent['checkpoint_sha256'], plan_sha256=config['plan_sha256'],
        optimizer_updates=0, backward_calls=0, frozen_state_tensors=len(before),
        total_parameters=config['total_parameters'], templates_fixed=True,
        template_mutations_reverted=suppressed, wall_seconds=time.perf_counter()-wall_started,
        seconds=float(timing.sum()), fps=len(timing)/float(timing.sum()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        evaluation_class=plan['mode'], test_eligible=False)
    check_files(ROOT, config['source_sha256'])
    check_files(ROOT, parent['input_sha256'])
    assert sha(checkpoint) == parent['checkpoint_sha256']
    assert sha(ROOT / args.plan) == config['plan_sha256']
    np.save(directory / 'frame_seconds.npy', timing)
    (directory / ('SMOKE.json' if args.smoke_frames else 'COMPLETE.json')).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
