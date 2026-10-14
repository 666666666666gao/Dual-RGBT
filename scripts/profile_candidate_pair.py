"""Persistent disposable B-shaped worker for a controlled one-versus-two-job comparison."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profile_candidate_concurrency import update, log_update
from dual_rgbt_global.candidate_policy import build_candidate_policy
from dual_rgbt_global.training import EvidencePairs, collate_pairs
from train_candidate_evidence_policy import PREFIXES, save_checkpoint, sha
from train_global_evidence_policy import cpu_state, cuda_batch
from lib.test.utils.hann import hann2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text())
    review = json.loads((ROOT / plan['source_review']).read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    assert sha('/home/gaob/CADTrack_ep0014.pth.tar') == plan['protected_checkpoint_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1' and os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert args.run == plan['run'] and plan['seconds'] in (180, 660)
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_candidate_policy(ROOT / plan['factory_checkpoint']).cuda().eval()
    checkpoint = torch.load(ROOT / plan['checkpoint'], map_location='cpu')
    assert checkpoint['stage'] == 'localize' and checkpoint['step'] == 6250 and not checkpoint['config']['sanity']
    model.load_state_dict(checkpoint['net'], strict=True)
    del checkpoint
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(PREFIXES))
    before = cpu_state(model)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=1e-4, weight_decay=1e-4)
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    config = dict(sanity=True, factory_checkpoint=plan['factory_checkpoint'], max_candidates=8,
                  scope='Disposable pair-profile worker, never a formal checkpoint.')
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    iterator = iter(DataLoader(EvidencePairs(ROOT / 'data/manifest.json', 1000000, plan['seed']),
                              batch_size=8, num_workers=2, collate_fn=collate_pairs))
    rng = np.random.RandomState(plan['seed'])
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    with (run / 'steps.csv').open('w', newline='') as steps_file, (run / 'sampled_labels.csv').open('w', newline='') as labels_file, (run / 'completion_times.csv').open('w', newline='') as times_file:
        step_writer = csv.DictWriter(steps_file, ('epoch', 'step', 'kind', 'loss', 'quality_loss', 'gain_loss'))
        label_writer = csv.DictWriter(labels_file, ('epoch', 'step', 'sequence', 'frame', 'modality', 'kind',
            'candidate_rank', 'candidate_count', 'main_geometry', 'query_geometry', 'main_raw_box', 'query_raw_box',
            'main_box', 'query_box', 'main_iou', 'query_iou', 'delta_from_stop'))
        step_writer.writeheader()
        label_writer.writeheader()
        time_writer = csv.DictWriter(times_file, ('step', 'completed_epoch'))
        time_writer.writeheader()
        for step in range(4):
            batch = cuda_batch(next(iterator))
            probe, result = update(model, batch, optimizer, rng, window, step, True)
            log_update(batch, result, step + 1, step_writer, label_writer, steps_file, labels_file)
            time_writer.writerow(dict(step=step + 1, completed_epoch=time.time()))
        frozen, changed = save_checkpoint(model, optimizer, run, config, 4, before, probe, window)
        torch.cuda.synchronize()
        started = time.time()
        (run / 'MEASURE_STARTED.json').write_text(json.dumps(dict(pid=os.getpid(), physical_gpu='1', started=started, seconds=plan['seconds'],
            m0_updates=4, strict_reload_pass=True, frozen_state_tensors=len(frozen), changed_state_tensors=changed), indent=2))
        measured = 0
        stop_path = run / 'PROFILE_STOP'
        while time.time() - started < plan['seconds'] and not stop_path.exists():
            batch = cuda_batch(next(iterator))
            _, result = update(model, batch, optimizer, rng, window, measured + 4, False)
            measured += 1
            log_update(batch, result, measured + 4, step_writer, label_writer, steps_file, labels_file)
            time_writer.writerow(dict(step=measured + 4, completed_epoch=time.time()))
        torch.cuda.synchronize()
        completed = time.time()
    after = cpu_state(model)
    assert all(torch.equal(before[name], after[name]) for name in frozen)
    assert {int(state['step'].item()) for state in optimizer.state.values()} == {measured + 4}
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    report = dict(status='COMPLETE', pid=os.getpid(), physical_gpu='1', measured_updates=measured, started=started, completed=completed,
        seconds=completed-started, requested_seconds=plan['seconds'], stop_requested=stop_path.exists(),
        updates_per_second=measured/(completed-started), batch_size=8,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        frozen_state_tensors=len(frozen), m0_updates=4, strict_reload_pass=True,
        private_timing_ledger_sha256=sha(run / 'completion_times.csv'),
        scope='Disposable B-shaped performance actor only. Main actor persists through all three comparison windows.')
    (run / 'COMPLETE.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
