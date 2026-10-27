"""Seeded CUDA update plus eleven real TRAIN frames, before accepting server25."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_inputs import load_manifest
from dual_rgbt.data import read_pair
from dual_rgbt_global.candidate_policy import build_candidate_policy
from dual_rgbt_global.candidate_tracker import CandidateEvidenceTracker
from run_benchmark_tracking import freeze, reload_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    assert Path(sys.prefix).resolve() == ROOT / 'conda-envs/dual_rgbt'
    assert sys.version_info[:2] == (3, 7)
    assert torch.__version__ == '1.12.1+cu113' and timm.__version__ == '0.9.12'
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    kernel = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4)).cuda()
    x = torch.randn(8, 8, device='cuda')
    target = torch.randn(8, 4, device='cuda')
    before = [v.detach().clone() for v in kernel.parameters()]
    optimizer = torch.optim.AdamW(kernel.parameters(), lr=1e-3)
    loss = (kernel(x) - target).square().mean()
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in kernel.parameters())
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, kernel.parameters()))
    torch.save(kernel.state_dict(), run / 'kernel.pth')
    clone = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4)).cuda()
    clone.load_state_dict(torch.load(run / 'kernel.pth'), strict=True)
    assert torch.equal(kernel(x), clone(x))
    bridge = np.arange(8, dtype=np.float32)
    assert np.array_equal(torch.from_numpy(bridge).numpy(), bridge)
    checkpoint = ROOT / 'runs/global_candidate_policy_formal_20261014/latest.pth'
    factory = ROOT / 'runs/stage_b_refreshed_20261013/latest.pth'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == 'b84d00454717cae56048753e0fe54c1d6b92cfcd2820c01fa65296508488ccb3'
    model = build_candidate_policy(factory).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
    freeze(model)
    total = sum(v.numel() for v in model.parameters())
    assert total == 103218887
    manifest = load_manifest(ROOT / 'data/benchmarks/lasher_train_smoke.json')
    assert manifest['benchmark'] == 'LasHeR' and manifest['split'] == 'train_smoke'
    sequence = manifest['sequences'][0]
    tracker = CandidateEvidenceTracker(model, .05, 8)
    tracker.initialize(read_pair(sequence, 0), sequence['initial_box'])
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for frame in range(1, 11):
            image = read_pair(sequence, frame)
            if frame == 10:
                saved = dict(state=tracker.state.copy(), templates=tracker.templates.clone(),
                             history=tracker.history.clone(), window=tracker.window.clone(),
                             frame=tracker.frame, since_query=tracker.since_query, image=image)
            box, route = tracker.track(image)
            assert np.isfinite(box).all()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    reload = reload_probe('e1', model, checkpoint, factory, saved, box, route, tracker, .05, 8)
    result = dict(status='ENVIRONMENT_KERNEL_AND_REAL_TRACKER_PASS',
                  kernel=dict(loss=loss.item(),optimizer_updates=1,strict_reload_bit_equal=True),
                  tracker=dict(sequence=sequence['name'],frames=11,tracked_frames=10,
                               optimizer_updates=0,reload=reload,total_parameters=total,
                               smoke_fps=10/seconds,smoke_seconds=seconds),
                  python=sys.version.split()[0],torch=torch.__version__,timm=timm.__version__,
                  prefix=sys.prefix,device=torch.cuda.get_device_name(0),
                  scope='Environment admission only. Tiny-kernel update and fixed B-checkpoint TRAIN tracking; smoke FPS is not formal benchmark FPS or accuracy.')
    (run / 'COMPLETE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
