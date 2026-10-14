"""Short throughput workload matching candidate B; no official weights or metrics are changed."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt_global.candidate_policy import build_candidate_policy
from dual_rgbt_evidence.regions import pixel_iou
from dual_rgbt_global.training import EvidencePairs, collate_pairs
from train_candidate_evidence_policy import PREFIXES, sampled_targets, save_checkpoint, sha
from train_global_evidence_policy import cpu_state, cuda_batch, probe_inputs, step_optimizer
from lib.test.utils.hann import hann2d


def update(model, batch, optimizer, rng, window, step, sanity):
    history = batch['main'].new_zeros(batch['main'].shape[0], 8)
    kind = step % 4
    targets = sampled_targets(model, batch, history, kind, 8, rng, window)
    preview = model.preview_state(batch['templates'], batch['main_preview'], batch['broad_preview'], history)
    qloss = F.smooth_l1_loss(preview['quality'], targets['quality'])
    gloss = batch['main'].new_zeros(())
    kinds = torch.full((batch['main'].shape[0], 1), kind, dtype=torch.long, device=batch['main'].device)
    for modality in range(2):
        prediction = model.candidate_gains(targets['primaries'][modality], preview, modality,
            batch['main_geometry'], batch['broad_geometry'], targets['queries'][modality], kinds)[:, 1]
        gloss = gloss + F.smooth_l1_loss(prediction, targets['gains'][modality]) / 2
    loss = qloss + gloss
    step_optimizer(loss, model, optimizer, PREFIXES, sanity)
    return probe_inputs(batch, history), dict(targets=targets, kind=kind, loss=loss,
                                            quality_loss=qloss, gain_loss=gloss)


def log_update(batch, result, step, step_writer, label_writer, steps_file, labels_file):
    targets, kind = result['targets'], result['kind']
    row = dict(epoch=1, step=step, kind=kind,
               **{name: result[name].item() for name in ('loss', 'quality_loss', 'gain_loss')})
    step_writer.writerow(row)
    for modality in range(2):
        for index, name in enumerate(batch['sequences']):
            label_writer.writerow(dict(epoch=1, step=step, sequence=name, frame=batch['frames'][index], modality=modality, kind=kind,
                candidate_rank=targets['ranks'][modality][index], candidate_count=targets['counts'][modality][index],
                main_geometry=json.dumps(batch['main_geometry'][index].cpu().tolist()),
                query_geometry=json.dumps(targets['queries'][modality][index, 0].cpu().tolist()),
                main_raw_box=json.dumps(targets['raw_boxes'][modality][0][index].cpu().tolist()),
                query_raw_box=json.dumps(targets['raw_boxes'][modality][1][index].cpu().tolist()),
                main_box=json.dumps(targets['boxes'][modality][0][index].cpu().tolist()),
                query_box=json.dumps(targets['boxes'][modality][1][index].cpu().tolist()),
                main_iou=float(targets['quality'][index, modality].item()),
                query_iou=float(pixel_iou(targets['boxes'][modality][1][index:index + 1], batch['target'][index:index + 1]).item()),
                delta_from_stop=float(targets['gains'][modality][index].item())))
    if step == 1 or step % 25 == 0:
        steps_file.flush()
        labels_file.flush()
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1' and os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    review = json.loads((ROOT / plan['source_review']).read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    assert sha('/home/gaob/CADTrack_ep0014.pth.tar') == plan['protected_checkpoint_sha256']
    assert args.run == plan['run'] and plan['seconds'] == 180
    torch.manual_seed(2027)
    np.random.seed(2027)
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
                  scope='Disposable throughput workload, not formal model training.')
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    loader = DataLoader(EvidencePairs(ROOT / 'data/manifest.json', 1000000, 2027),
                        batch_size=8, num_workers=2, collate_fn=collate_pairs)
    iterator = iter(loader)
    rng = np.random.RandomState(2027)
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    with (run / 'steps.csv').open('w', newline='') as steps_file, (run / 'sampled_labels.csv').open('w', newline='') as labels_file:
        step_writer = csv.DictWriter(steps_file, ('epoch', 'step', 'kind', 'loss', 'quality_loss', 'gain_loss'))
        label_writer = csv.DictWriter(labels_file, ('epoch', 'step', 'sequence', 'frame', 'modality', 'kind',
            'candidate_rank', 'candidate_count', 'main_geometry', 'query_geometry', 'main_raw_box', 'query_raw_box',
            'main_box', 'query_box', 'main_iou', 'query_iou', 'delta_from_stop'))
        step_writer.writeheader()
        label_writer.writeheader()
        for step in range(4):
            batch = cuda_batch(next(iterator))
            probe, result = update(model, batch, optimizer, rng, window, step, True)
            log_update(batch, result, step + 1, step_writer, label_writer, steps_file, labels_file)
        frozen, changed = save_checkpoint(model, optimizer, run, config, 4, before, probe, window)
        torch.cuda.synchronize()
        started = time.time()
        (run / 'MEASURE_STARTED.json').write_text(json.dumps(dict(started=started, seconds=180,
            m0_updates=4, strict_reload_pass=True, frozen_state_tensors=len(frozen), changed_state_tensors=changed), indent=2))
        measured = 0
        while time.time() - started < 180:
            batch = cuda_batch(next(iterator))
            _, result = update(model, batch, optimizer, rng, window, measured + 4, False)
            measured += 1
            log_update(batch, result, measured + 4, step_writer, label_writer, steps_file, labels_file)
        torch.cuda.synchronize()
        completed = time.time()
    after = cpu_state(model)
    assert all(torch.equal(before[name], after[name]) for name in frozen)
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name, value in plan['input_sha256'].items())
    result = dict(status='COMPLETE', measured_updates=measured, batch_size=8, started=started,
        completed=completed, seconds=completed-started, updates_per_second=measured/(completed-started),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        m0_updates=4, frozen_state_tensors=len(frozen), strict_reload_pass=True,
        scope='Single short B-shaped concurrent workload. No official weights, data or test metrics changed.')
    (run / 'COMPLETE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
