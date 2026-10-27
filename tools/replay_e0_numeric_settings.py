"""Replay fixed historical E0 DEV prefixes; diagnose numerical trajectory differences."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_bytes())
    review = json.loads((ROOT / plan['source_review']).read_bytes())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(ROOT / args.plan)
    assert review['source_sha256'] == plan['source_sha256']
    for name, digest in dict(plan['source_sha256'], **plan['input_sha256']).items():
        assert sha(ROOT / name) == digest, name
    native = json.loads((ROOT / 'runs/pipeline_tasks' / plan['task'] / 'status.json').read_bytes())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['task'] == plan['task'] and native['physical_gpu'] == '0'
    assert native['command'] == [sys.executable, str(Path(__file__).resolve()), '--plan', args.plan]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    directory = ROOT / 'runs' / plan['run']
    directory.mkdir(parents=True, exist_ok=False)
    from run_benchmark_tracking import build_model, build_tracker, freeze
    from dual_rgbt.data import read_pair
    manifest = json.loads((ROOT / 'data/manifest.json').read_bytes())
    sequences = {s['name']: s for s in manifest['dev']}
    references = json.loads((ROOT / plan['reference_prefixes']).read_bytes())
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.set_num_threads(1)
    defaults = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    cudnn_benchmark=torch.backends.cudnn.benchmark)
    model = build_model('e0', None)
    model.load_state_dict(torch.load(ROOT / plan['checkpoint'], map_location='cpu')['net'], strict=True)
    freeze(model)
    rows = []
    for phase, flag in enumerate(plan['cudnn_tf32_phases']):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = flag
        for name in plan['sequence_names']:
            sequence = sequences[name]
            tracker = build_tracker('e0', model, 0., 0)
            initial = np.asarray(sequence['boxes'][0], dtype=np.float32)
            tracker.initialize(read_pair(sequence, 0), initial)
            predictions, routes = [initial.tolist()], []
            for frame in range(1, plan['frames_per_sequence']):
                box, route = tracker.track(read_pair(sequence, frame))
                predictions.append(box.tolist())
                routes.append({key: value for key, value in route.items() if key != 'seconds'})
            # Compare using the exact six-decimal text representation of archived predictions.
            stored = np.asarray([[float('%.6f' % v) for v in box] for box in predictions])
            comparisons = {}
            for kind, expected in references['sequences'][name].items():
                expected = np.asarray(expected)
                assert stored.shape == expected.shape
                comparisons[kind] = dict(exact_frames=int(np.all(stored == expected, axis=1).sum()),
                                         max_abs_box_difference=float(np.abs(stored-expected).max()))
            rows.append(dict(phase=phase, cudnn_tf32=flag, sequence=name,
                             predictions=predictions, routes=routes, comparisons=comparisons))
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    result = dict(status='COMPLETE', checkpoint_sha256=plan['input_sha256'][plan['checkpoint']],
                  defaults_at_process_start=defaults, torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), rows=rows, optimizer_updates=0,
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  scope=plan['scope'])
    (directory / 'COMPLETE.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE', comparisons=[{k:v for k,v in r.items()
                            if k not in ('predictions', 'routes')} for r in rows])), flush=True)


if __name__ == '__main__':
    main()
