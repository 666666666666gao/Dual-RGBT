"""Time the original stripped CADTrack online path; never recompute accuracy."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from dual_rgbt.data import read_pair
from launch_server25_training import startup


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_state(pid):
    return next(v.split()[1] for v in (Path('/proc') / str(pid) / 'status').read_text().splitlines()
                if v.startswith('State:'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_bytes())
    review = json.loads((ROOT / plan['source_review']).read_bytes())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan)
    assert review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / n) == h for n, h in plan['source_sha256'].items())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0' and os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    checkpoint = ROOT / plan['checkpoint']
    assert sha(checkpoint) == plan['checkpoint_sha256']
    assert sha(ROOT / 'data/manifest.json') == plan['manifest_sha256']
    manifest = json.loads((ROOT / 'data/manifest.json').read_bytes())
    by_name = {v['name']:v for v in manifest['train']}
    sequences = [by_name[n] for n in plan['sequence_names']]
    assert len(sequences) == 3
    torch.manual_seed(plan['seed']); np.random.seed(plan['seed']); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    execution_path = ROOT / 'reports/SERVER25_REMAINING_GRID_EXECUTION_20261027.json'
    assert sha(execution_path) == plan['training_execution_sha256']
    execution = json.loads(execution_path.read_bytes())
    cases = [c for c in execution['cases'] if c['plan']['physical_gpu'] == 0]
    assert len(cases) == 2
    owners = [startup(c) for c in cases]
    groups = [v['native']['pid'] for v in owners]
    children = [v['child_pid'] for v in owners]
    assert all(os.getpgid(p) == p and (Path('/proc') / str(p)).stat().st_uid == os.getuid() for p in groups)
    assert all(os.getpgid(p) in groups and (Path('/proc') / str(p)).stat().st_uid == os.getuid() for p in children)
    uuid = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],universal_newlines=True).strip()
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],universal_newlines=True)
    actual = {int(v.split(',')[1]) for v in apps.splitlines() if v.split(',')[0].strip() == uuid}
    assert actual == set(children)
    run = ROOT / 'runs' / plan['run']; run.mkdir(parents=True, exist_ok=False)
    (run / 'ownership_before_pause.json').write_text(json.dumps(owners, indent=2)+'\n')
    paused_at = time.time()
    records = []
    initializations = []
    # These two process groups are deliberately paused by this measurement.
    # Restore them even if the baseline import, forward, or timing assertions fail.
    try:
        for pid in groups: os.killpg(pid, signal.SIGSTOP)
        time.sleep(2)
        assert all(process_state(p) == 'T' for p in groups + children)
        from lib.config.cadtrack.config import cfg, update_config_from_file
        from lib.test.tracker.cadtrack import CADTrack
        from lib.test.utils import TrackerParams
        update_config_from_file(str(ROOT / 'experiments/cadtrack/cadtrack.yaml'))
        params = TrackerParams(); params.cfg = cfg
        params.template_factor, params.template_size = cfg.TEST.TEMPLATE_FACTOR, cfg.TEST.TEMPLATE_SIZE
        params.search_factor, params.search_size = cfg.TEST.SEARCH_FACTOR, cfg.TEST.SEARCH_SIZE
        params.checkpoint = str(checkpoint); params.save_all_boxes = False; params.debug = 0
        tracker = CADTrack(params)
        total = sum(p.numel() for p in tracker.network.parameters())
        assert total == 103138437
        torch.cuda.reset_peak_memory_stats()
        for sequence in sequences:
            torch.cuda.synchronize(); start = time.perf_counter()
            tracker.initialize(read_pair(sequence, 0), dict(init_bbox=sequence['boxes'][0]))
            torch.cuda.synchronize(); initializations.append(time.perf_counter()-start)
            for frame in range(1, plan['warmup_frames'] + 1):
                tracker.track(read_pair(sequence, frame))
            for frame in range(plan['warmup_frames'] + 1, plan['warmup_frames'] + plan['measured_frames'] + 1):
                torch.cuda.synchronize(); begin = time.perf_counter()
                image = read_pair(sequence, frame); decoded = time.perf_counter()
                output = tracker.track(image)
                torch.cuda.synchronize(); end = time.perf_counter()
                assert np.isfinite(output['target_bbox']).all()
                records.append(dict(sequence=sequence['name'], frame=frame, decode_seconds=decoded-begin,
                                    service_seconds=end-decoded, end_to_end_seconds=end-begin))
        assert len(records) == 600
        def statistics(key):
            values = np.asarray([v[key] for v in records])
            assert np.isfinite(values).all() and (values > 0).all()
            return dict(fps=len(values)/values.sum(), mean_ms=1000*values.mean(), p95_ms=1000*np.percentile(values,95))
        result = dict(status='BASELINE_PARAMETER_AND_LATENCY_SAMPLE_COMPLETE', total_parameters=total,
                      checkpoint=plan['checkpoint'], checkpoint_sha256=plan['checkpoint_sha256'],
                      frames=len(records), sequences=plan['sequence_names'], warmup_per_sequence=plan['warmup_frames'],
                      search_size=params.search_size, template_size=params.template_size, template_number=cfg.DATA.TEMPLATE.NUMBER,
                      end_to_end=statistics('end_to_end_seconds'), service=statistics('service_seconds'),
                      initialization_seconds=initializations, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), device=torch.cuda.get_device_name(0),
                      source_plan_sha256=sha(ROOT / args.plan), precision='FP32; matmul and cuDNN TF32 disabled',
                      scope='600-frame TRAIN timing sample, original CADTrack tracker/template rule and SOTep14 checkpoint. End-to-end includes RGB/TIR decode; service excludes decode. Initializations reported separately. No accuracy or trainable-parameter evaluation, no matched Ours speedup claim. Two owned same-card trainers paused; other GPUs/CPU jobs continue.')
    finally:
        for pid in groups: os.killpg(pid, signal.SIGCONT)
        time.sleep(1)
        restored = {str(p):process_state(p) for p in groups + children}
        (run / 'resume.json').write_text(json.dumps(dict(paused_seconds=time.time()-paused_at, states=restored),indent=2)+'\n')
        assert all(v not in ('T','t') for v in restored.values())
    with (run / 'frame_seconds.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(records[0])); writer.writeheader(); writer.writerows(records)
    result['resume'] = json.loads((run / 'resume.json').read_bytes())
    (run / 'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
