"""Bracket E0 timing with the original baseline on the same GPU and TRAIN frames."""
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
from profile_cadtrack_baseline import process_state, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    args = ap.parse_args()
    plan = json.loads((ROOT / args.plan).read_bytes())
    review = json.loads((ROOT / plan['source_review']).read_bytes())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan)
    assert review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / n) == h for n, h in plan['source_sha256'].items())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0' and os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert os.environ['OPENCV_FOR_THREADS_NUM'] == '1'
    for n, h in plan['input_sha256'].items():
        assert sha(ROOT / n) == h
    manifest = json.loads((ROOT / 'data/manifest.json').read_bytes())
    by_name = {s['name']: s for s in manifest['train']}
    sequences = [by_name[n] for n in plan['sequence_names']]
    assert len(sequences) == 3
    execution = json.loads((ROOT / 'reports/SERVER25_REMAINING_GRID_EXECUTION_20261027.json').read_bytes())
    cases = [c for c in execution['cases'] if c['plan']['physical_gpu'] == 0]
    assert len(cases) == 2
    owners = [startup(c) for c in cases]
    groups = [v['native']['pid'] for v in owners]
    children = [v['child_pid'] for v in owners]
    assert all(os.getpgid(p) == p and (Path('/proc') / str(p)).stat().st_uid == os.getuid() for p in groups)
    assert all(os.getpgid(p) in groups and (Path('/proc') / str(p)).stat().st_uid == os.getuid() for p in children)
    uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], universal_newlines=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], universal_newlines=True)
    assert {int(s.split(',')[1]) for s in apps.splitlines() if s.split(',')[0].strip() == uuid} == set(children)
    directory = ROOT / 'runs' / plan['run']
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'ownership_before_pause.json').write_text(json.dumps(owners, indent=2) + '\n')
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    records, phases = [], []
    paused_at = time.time()
    # This measurement deliberately pauses exactly two verified owned groups.
    # The previous baseline's real import failure demonstrated why finally is required.
    try:
        for pid in groups:
            os.killpg(pid, signal.SIGSTOP)
        time.sleep(2)
        assert all(process_state(p) == 'T' for p in groups + children)
        from lib.config.cadtrack.config import cfg, update_config_from_file
        from lib.test.tracker.cadtrack import CADTrack
        from lib.test.utils import TrackerParams
        update_config_from_file(str(ROOT / 'experiments/cadtrack/cadtrack.yaml'))
        params = TrackerParams()
        params.cfg = cfg
        params.template_factor, params.template_size = cfg.TEST.TEMPLATE_FACTOR, cfg.TEST.TEMPLATE_SIZE
        params.search_factor, params.search_size = cfg.TEST.SEARCH_FACTOR, cfg.TEST.SEARCH_SIZE
        params.checkpoint = str(ROOT / plan['baseline_checkpoint'])
        params.save_all_boxes, params.debug = False, 0
        baseline = CADTrack(params)
        baseline.network.requires_grad_(False)
        from run_benchmark_tracking import build_model, build_tracker, freeze
        model = build_model('e0', None)
        state = torch.load(ROOT / plan['e0_checkpoint'], map_location='cpu')
        model.load_state_dict(state['net'], strict=True)
        del state
        freeze(model)
        assert sum(p.numel() for p in baseline.network.parameters()) == 103138437
        assert sum(p.numel() for p in model.parameters()) == 103374215
        assert (params.search_size, params.template_size, cfg.DATA.TEMPLATE.NUMBER) == (256, 128, 2)
        for phase in plan['phases']:
            is_baseline = phase['model'] == 'baseline'
            tracker = baseline if is_baseline else build_tracker('e0', model, phase['cost_weight'], 0)
            initialized = []
            action_count = 0
            for sequence in sequences:
                torch.cuda.synchronize()
                start = time.perf_counter()
                tracker.initialize(read_pair(sequence, 0), dict(init_bbox=sequence['boxes'][0]) if is_baseline else sequence['boxes'][0])
                torch.cuda.synchronize()
                initialized.append(time.perf_counter() - start)
                for frame in range(1, plan['warmup_frames'] + 1):
                    tracker.track(read_pair(sequence, frame))
                for frame in range(plan['warmup_frames'] + 1, plan['warmup_frames'] + plan['measured_frames'] + 1):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    image = read_pair(sequence, frame)
                    decoded = time.perf_counter()
                    prediction = tracker.track(image)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    box = prediction['target_bbox'] if is_baseline else prediction[0]
                    assert np.isfinite(box).all()
                    if not is_baseline:
                        action_count += int(prediction[1]['action'] != 0)
                    records.append(dict(phase=phase['name'], sequence=sequence['name'], frame=frame,
                                        decode_seconds=decoded-begin, service_seconds=end-decoded,
                                        end_to_end_seconds=end-begin))
            selected = [r for r in records if r['phase'] == phase['name']]
            assert len(selected) == 600
            statistics = {}
            for label in ['service', 'end_to_end']:
                values = np.asarray([r[label + '_seconds'] for r in selected])
                assert np.isfinite(values).all() and (values > 0).all()
                statistics[label] = dict(fps=len(values)/values.sum(), mean_ms=1000*values.mean(), p95_ms=1000*np.percentile(values, 95))
            phases.append(dict(phase, total_parameters=103138437 if is_baseline else 103374215,
                               frames=len(selected), initialization_seconds=initialized,
                               auxiliary_calls=action_count if not is_baseline else None, **statistics))
            print(json.dumps(phases[-1]), flush=True)
    finally:
        for pid in groups:
            os.killpg(pid, signal.SIGCONT)
        time.sleep(1)
        restored = {str(p): process_state(p) for p in groups + children}
        (directory / 'resume.json').write_text(json.dumps(dict(paused_seconds=time.time()-paused_at, states=restored), indent=2) + '\n')
        assert all(s not in ('T', 't') for s in restored.values())
    assert len(records) == 2400 and len(phases) == 4
    with (directory / 'frame_seconds.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    assert all(sha(ROOT / n) == h for n, h in plan['source_sha256'].items())
    result = dict(status='PAIRED_BASELINE_E0_LATENCY_COMPLETE', phases=phases,
                  plan_sha256=sha(ROOT / args.plan), frame_csv_sha256=sha(directory / 'frame_seconds.csv'),
                  resume=json.loads((directory / 'resume.json').read_bytes()),
                  device=torch.cuda.get_device_name(0), precision='FP32, TF32 disabled',
                  scope='Same600TRAINframes per phase,20warmup per sequence, fixed baseline/E0cost0/E0cost0.05/baseline order. Initialization separate, no prediction accuracy or trainable parameter evaluation. Own GPU0 trainers paused; other GPU/CPU tasks continue. Historical E0 fixed checkpoint, not current in-training32 models.')
    (directory / 'COMPLETE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
