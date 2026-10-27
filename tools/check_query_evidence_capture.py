"""Compare unobserved and observed real TRAIN trajectories with fixed parent weights."""
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
    from dual_rgbt.data import read_pair
    from run_benchmark_tracking import build_model, build_tracker, freeze
    from query_evidence_capture import QueryEvidenceCapture
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_model('e1', ROOT / plan['factory_checkpoint'])
    parent = torch.load(ROOT / plan['checkpoint'], map_location='cpu')
    assert parent['stage'] == 'candidate_b' and parent['step'] == 6250 and not parent['config']['sanity']
    model.load_state_dict(parent['net'], strict=True)
    freeze(model)
    directory = ROOT / 'runs' / plan['run']
    directory.mkdir(parents=True, exist_ok=False)
    sequences = {s['name']: s for s in json.loads((ROOT / 'data/manifest.json').read_bytes())['train']}
    rows = []
    capture = None
    for observed in (False, True):
        if observed:
            capture = QueryEvidenceCapture(model, 0.)
        for cost in plan['costs']:
            if observed:
                capture.cost_weight = cost
            for name in plan['sequence_names']:
                sequence = sequences[name]
                tracker = build_tracker('e1', model, cost, plan['max_candidates'])
                initial = np.asarray(sequence['boxes'][0], dtype=np.float32)
                tracker.initialize(read_pair(sequence, 0), initial)
                boxes, routes, evidence = [initial.tolist()], [], []
                for frame in range(1, plan['frames_per_sequence']):
                    image = read_pair(sequence, frame)
                    if observed:
                        capture.begin(image, tracker.state)
                    box, route = tracker.track(image)
                    boxes.append(box.tolist())
                    routes.append({k:v for k,v in route.items() if k != 'seconds'})
                    if observed:
                        evidence.append(capture.finish(frame, route, box))
                row = dict(observed=observed, cost=cost, sequence=name, boxes=boxes,
                           routes=routes, evidence=evidence,
                           template_sha256=hashlib.sha256(tracker.templates.cpu().numpy().tobytes()).hexdigest(),
                           history=tracker.history.cpu().tolist())
                if observed:
                    previous = next(r for r in rows if not r['observed'] and r['cost']==cost and r['sequence']==name)
                    for key in ('boxes', 'routes', 'template_sha256', 'history'):
                        assert row[key] == previous[key], key
                rows.append(row)
    actions = [r['action'] for row in rows if row['observed'] for r in row['routes']]
    assert 0 in actions and any(a > 0 for a in actions)
    assert all(torch.equal(model.state_dict()[n].cpu(), value) for n,value in parent['net'].items())
    assert all(p.grad is None for p in model.parameters())
    result = dict(status='CAPTURE_REAL_TRAJECTORY_PASS', rows=rows,
                  plan_sha256=sha(ROOT / args.plan),
                  checkpoint_sha256=plan['input_sha256'][plan['checkpoint']],
                  stop_frames=actions.count(0), query_frames=sum(a>0 for a in actions),
                  parameters_unchanged=True, optimizer_updates=0, predictions_routes_templates_history_exact=True,
                  scope='Fixed parent B6250; real TRAIN prefixes, not final model or accuracy evaluation.')
    (directory / 'COMPLETE.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
