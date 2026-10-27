"""Read-only CPU verification of four explicitly authorized completed probes."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import torch

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ROOT = Path('/data2/gb/Dual-RGBT')
NAMES = ['host_allocator_r4_default_20261027_sanity',
         'host_allocator_r4_compact_20261027_sanity',
         'host_allocator_trim_20261027_sanity',
         'host_allocator_warm_20261027_sanity']

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load(name):
    run = ROOT / 'runs' / name
    native = json.loads((ROOT / 'runs/pipeline_tasks' / ('train_' + name) / 'status.json').read_bytes())
    complete = json.loads((run / 'COMPLETE.json').read_bytes())
    probe = json.loads((run / 'SANITY.json').read_bytes())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert complete['status'] == 'COMPLETE' and complete['sanity']
    assert (complete['updates'], complete['tracked_frames'], complete['initializes']) == (4, 20, 1)
    assert complete['frozen_state_tensors'] == 251 and len(complete['changed_state_tensors']) == 14
    assert probe['strict_reload_prediction_bit_equal'] and probe['status'] == 'PASS'
    weight_sha = digest(run / 'latest.pth')
    assert weight_sha == complete['checkpoint_sha256']
    state = torch.load(str(run / 'latest.pth'), map_location='cpu')
    assert state['step'] == 4 and state['config']['sanity'] and state['stage'] == 'candidate_c'
    assert state['config'] == json.loads((run / 'config.json').read_bytes())
    assert len(state['net']) == 265
    assert all(t.device.type == 'cpu' and torch.isfinite(t).all().item() for t in state['net'].values())
    files = {n: digest(run / n) for n in ['config.json', 'COMPLETE.json', 'SANITY.json', 'trajectory_steps.csv', 'query_labels.csv']}
    assert files['query_labels.csv'] == complete['private_labels_sha256']
    rows = list(csv.DictReader((run / 'trajectory_steps.csv').read_text().splitlines()))
    labels = list(csv.DictReader((run / 'query_labels.csv').read_text().splitlines()))
    assert len(rows) == 4 and [int(r['step']) for r in rows] == [1, 2, 3, 4]
    assert len(labels) == complete['labelled_queries'] == 11
    assert {int(v['step'].item()) if isinstance(v['step'], torch.Tensor) else int(v['step']) for v in state['optimizer']['state'].values()} == {4}
    return state, dict(run=name, native=native, weight_sha256=weight_sha,
                      metadata_sha256=files, csv_rows=len(rows), label_rows=len(labels),
                      state_tensor_count=265, all_state_tensors_finite=True,
                      state_element_count=sum(t.numel() for t in state['net'].values()),
                      checkpoint_config_matches_disk=True, optimizer_step=4)

def compare_optimizer(a, b):
    differences = []
    scalar_differences = []
    tensor_count = 0
    max_abs = 0.0
    def walk(x, y, path):
        nonlocal tensor_count, max_abs
        if isinstance(x, torch.Tensor):
            tensor_count += 1
            assert x.shape == y.shape and x.dtype == y.dtype
            assert x.device.type == y.device.type == 'cpu'
            assert torch.isfinite(x).all().item() and torch.isfinite(y).all().item()
            delta = float((x - y).abs().max().item())
            max_abs = max(max_abs, delta)
            close = torch.allclose(x, y, atol=1e-8, rtol=1e-6)
            if not torch.equal(x, y):
                differences.append(dict(path=path, max_abs=delta, tolerance_pass=close))
            return close
        if isinstance(x, dict):
            assert x.keys() == y.keys()
            outcomes = [walk(x[k], y[k], path + '.' + str(k)) for k in x]
            return all(outcomes)
        if isinstance(x, (list, tuple)):
            assert type(x) is type(y) and len(x) == len(y)
            outcomes = [walk(u, v, path + '.' + str(i)) for i, (u, v) in enumerate(zip(x, y))]
            return all(outcomes)
        if x != y:
            scalar_differences.append(path)
        return x == y
    close = walk(a, b, 'optimizer')
    return dict(tensor_count=tensor_count, bit_equal=not differences and not scalar_differences,
                tolerance_pass=close, atol=1e-8, rtol=1e-6, max_abs=max_abs,
                tensor_differences=differences, scalar_differences=scalar_differences)

baseline, base_receipt = load(NAMES[0])
receipts = [base_receipt]
comparisons = []
for name in NAMES[1:]:
    candidate, receipt = load(name)
    receipts.append(receipt)
    assert tuple(candidate['net']) == tuple(baseline['net'])
    differences = [key for key in baseline['net'] if not torch.equal(baseline['net'][key], candidate['net'][key])]
    assert not differences
    optimizer = compare_optimizer(baseline['optimizer'], candidate['optimizer'])
    assert optimizer['tolerance_pass']
    csv_equal = {n: (ROOT/'runs'/NAMES[0]/n).read_bytes() == (ROOT/'runs'/name/n).read_bytes()
                 for n in ['trajectory_steps.csv', 'query_labels.csv']}
    assert all(csv_equal.values())
    comparisons.append(dict(reference=NAMES[0], candidate=name, state_tensors_bit_equal=265,
                            optimizer=optimizer, csv_bit_equal=csv_equal))
    del candidate
print(json.dumps(dict(status='COMPLETED_CPU_VERIFICATION_PASS', cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                      threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                      torch_version=torch.__version__, probes=receipts, comparisons=comparisons,
                      scope='One bounded SSH CPU-only read of four completed memory-probe weights and associated completed metadata. No model instantiated, GPU query, forward, backward, training, active job access, or run modification.')))
