"""Read-only source/artifact review and CPU scalar fixtures; never imports the trainer."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / 'reports/global_candidate_joint_quality_trajectory_plan_20261026.json'
SOURCE = 'scripts/train_candidate_joint_quality_trajectory.py'
BASE = 'scripts/train_candidate_rankstop_trajectory.py'
PLAN_SHA = '014101511170dc2742c1cca4662035bae7a2b106f0f6e38cc10310eda090d42f'
SOURCE_SHA = 'a19e7ae96787250ded975e89f2aaeb87a0f9cb3ab7b631e4fe9294ba4d0c2331'
plan = json.loads(PLAN_PATH.read_text())
assert hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() == PLAN_SHA
assert hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA
for name, expected in plan['source_sha256'].items():
    data = (ROOT / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, name
    ast.parse(data, filename=name)
old_map = json.loads((ROOT / 'reports/rankstop_dev_plan_server26_20261026.json').read_text())['source_sha256']
assert len(old_map) == 48
assert {k: v for k, v in plan['source_sha256'].items() if k != SOURCE} == old_map
trees = {name: ast.parse((ROOT / name).read_text()) for name in (SOURCE, BASE)}
functions = {name: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
             for name, tree in trees.items()}
unchanged = ['trajectory_counts', 'causal_batch', 'selected_target', 'write_label']
for name in unchanged:
    assert ast.dump(functions[SOURCE][name]) == ast.dump(functions[BASE][name]), name
new_loss, old_loss = functions[SOURCE]['controller_loss'], functions[BASE]['controller_loss']
# The entire old gain/STOP implementation is retained, excluding only its return tuple.
assert [ast.dump(n) for n in new_loss.body[5:-1]] == [ast.dump(n) for n in old_loss.body[1:-1]]
local_inputs = []
for name, expected in plan['input_sha256'].items():
    p = ROOT / name
    if p.exists():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, name
        local_inputs.append(name)
payload = dict(plan=plan, source=(ROOT / SOURCE).read_text(), baseline=(ROOT / BASE).read_text(),
               source_sha256=SOURCE_SHA, plan_sha256=PLAN_SHA)
payload64 = base64.b64encode(json.dumps(payload).encode()).decode()
remote = r'''
import ast, base64, hashlib, json, os
from pathlib import Path
import torch
from torch.nn import functional as F
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
payload = json.loads(base64.b64decode('__PAYLOAD__'))
plan = payload['plan']
root = Path('/data2/gb/Dual-RGBT')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
inputs = {}
for name, expected in plan['input_sha256'].items():
    actual = sha(root / name)
    assert actual == expected, (name, actual, expected)
    inputs[name] = actual
source48 = {}
for name, expected in plan['source_sha256'].items():
    if name == 'scripts/train_candidate_joint_quality_trajectory.py':
        continue
    actual = sha(root / name)
    assert actual == expected, (name, actual, expected)
    source48[name] = actual
manifest = json.loads((root / 'data/manifest.json').read_text())
by_name = {r['name']: r for r in manifest['train']}
assert len(by_name) == len(plan['sequence_names']) == 882
assert set(by_name) == set(plan['sequence_names'])
assert len(manifest['dev']) == 97
assert not set(by_name).intersection(r['name'] for r in manifest['dev'])
rows = [by_name[name] for name in plan['sequence_names']]
counts = dict(tracked_frames=plan['rounds'] * sum(len(r['boxes']) - 1 for r in rows),
              updates=plan['rounds'] * sum(sum(i > 0 and i % plan['label_stride'] == 0 for i in r['valid_frames']) for r in rows),
              initializes=plan['rounds'] * len(rows))
assert counts == plan['expected_formal_counts']
assert plan['rounds'] == 2 and plan['label_stride'] == 5
assert all(0 in r['valid_frames'] for r in rows)
assert [i for i in rows[0]['valid_frames'] if 0 < i <= 20 and i % 5 == 0] == [5, 10, 15, 20]
b_complete = json.loads((root / plan['b_complete']).read_text())
b_native = json.loads((root / plan['b_native_status']).read_text())
assert b_native['state'] == 'COMPLETE' and b_native['exit_code'] == 0 and b_native['physical_gpu'] == '1'
assert b_complete['status'] == 'COMPLETE' and not b_complete['sanity']
assert (b_complete['updates'], b_complete['samples'], b_complete['labelled_queries']) == (6250, 50000, 100000)
assert set(b_complete['seen_train_sequences']) == set(by_name)
assert b_complete['frozen_state_tensors'] == 257 and len(b_complete['changed_state_tensors']) == 8
assert b_complete['route_samples'] == {'%d:%d' % (r, k): 8 * ((6250 + 3 - k) // 4) for r in range(2) for k in range(4)}
assert b_complete['checkpoint_sha256'] == plan['input_sha256'][plan['checkpoint']]
checkpoint = torch.load(root / plan['checkpoint'], map_location='cpu')
assert checkpoint['stage'] == 'candidate_b' and checkpoint['step'] == 6250 and not checkpoint['config']['sanity']
b_plan = json.loads((root / 'reports/global_candidate_policy_plan.json').read_text())
assert checkpoint['config']['plan_sha256'] == plan['b_plan_sha256']
assert checkpoint['config']['source_sha256'] == b_plan['source_sha256']
assert checkpoint['config']['input_sha256'] == b_plan['input_sha256']
prefixes = ('executor.preview.', 'executor.quality.', 'query_gain.')
adapted = [name for name in checkpoint['net'] if name.startswith(prefixes)]
frozen = [name for name in checkpoint['net'] if not name.startswith(prefixes)]
assert len(checkpoint['net']) == 265 and len(adapted) == 14 and len(frozen) == 251
assert sum(name.startswith('executor.preview.') for name in adapted) == 6
assert sum(name.startswith('executor.quality.') for name in adapted) == 4
assert sum(name.startswith('query_gain.') for name in adapted) == 4
assert all(tensor.device.type == 'cpu' for tensor in checkpoint['net'].values())
assert len(checkpoint['optimizer']['state']) == 8
assert {int(state['step'].item()) for state in checkpoint['optimizer']['state'].values()} == {6250}
del checkpoint
def extract(text, name):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node])
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.fix_missing_locations(module), '<audited-source>', 'exec'), namespace)
    return namespace[name]
new_loss = extract(payload['source'], 'controller_loss')
old_loss = extract(payload['baseline'], 'controller_loss')
class ScalarScores:
    def __init__(self, values):
        self.values, self.calls = values, 0
    def candidate_gains(self, primary, preview, role, main_geometry, broad_geometry, queries, kinds):
        index = role if self.calls < 2 else 2
        self.calls += 1
        score = self.values[index:index + 1, None]
        return torch.cat((torch.zeros_like(score), score), dim=1)
def invoke(fn, quality, target_quality, values, target_gains, extra):
    model = ScalarScores(values)
    batch = {'main': torch.zeros(1), 'main_geometry': None, 'broad_geometry': None}
    targets = {'quality': torch.tensor([target_quality]), 'primaries': [None, None],
               'queries': [None, None], 'gains': [torch.tensor([v]) for v in target_gains[:2]]}
    selected = dict(extra_query=extra, kind=2, geometry=torch.zeros(1, 3), gain=torch.tensor([target_gains[2]]))
    return fn(model, batch, {'quality': quality}, targets, 0, selected, 0)
rank_cases = [
    ('rgb_better', [0.0, 0.0], [0.9, 0.2], [-0.7, 0.7]),
    ('tir_better', [0.0, 0.0], [0.2, 0.9], [0.7, -0.7]),
    ('wrong_order', [-0.2, 0.4], [0.9, 0.2], [-0.7, 0.7]),
    ('tie', [0.8, -0.3], [0.5, 0.5], [0.0, 0.0]),
    ('zero_tie', [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
    ('margin_satisfied', [1.0, 0.0], [0.9, 0.2], [0.0, 0.0]),
    ('exact_margin', [0.9, 0.2], [0.9, 0.2], [0.0, 0.0]),
    ('small_target_gap', [0.0, 0.0], [0.501, 0.5], [-0.001, 0.001]),
]
rank_results = []
for name, predicted, target, gradient in rank_cases:
    q = torch.tensor([predicted], requires_grad=True)
    values = torch.zeros(3, requires_grad=True)
    out = invoke(new_loss, q, target, values, [0.2, -0.2, 0.1], False)
    dq = torch.autograd.grad(out[4], q)[0]
    assert torch.allclose(dq, torch.tensor([gradient]), atol=1e-7), (name, dq)
    assert all(torch.isfinite(x).all() for x in out)
    if target[0] == target[1]:
        assert out[4].item() == 0.0
    rank_results.append(dict(name=name, quality_rank=out[4].item(), rank_gradient=dq.tolist()))
gain_results = []
gain_cases = [(False, [0.2, -0.3, 0.4]), (True, [0.2, -0.3, 0.4]),
              (False, [0.0, 0.0, 0.0]), (True, [0.005, -0.005, 0.0]),
              (False, [0.01, -0.01, 0.0]), (True, [-0.2, 0.3, -0.4])]
for index, (extra, targets) in enumerate(gain_cases):
    q1 = torch.tensor([[0.2, 0.4]], requires_grad=True)
    q2 = q1.detach().clone().requires_grad_()
    v1 = torch.tensor([0.01, -0.02, 0.03], requires_grad=True)
    v2 = v1.detach().clone().requires_grad_()
    new = invoke(new_loss, q1, [0.8, 0.2], v1, targets, extra)
    old = invoke(old_loss, q2, [0.8, 0.2], v2, targets, extra)
    assert torch.equal(new[2], old[2])
    assert torch.equal(new[3], old[1])
    assert torch.equal(torch.autograd.grad(new[2], v1)[0], torch.autograd.grad(old[2], v2)[0])
    gain_results.append(dict(index=index, extra=extra, gain_stop_value=new[2].item()))
# Exercise the actual new native-wrapper assertions with valid and mutated records.
source_tree = ast.parse(payload['source'])
main = next(n for n in source_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
native_asserts = [n for n in main.body if isinstance(n, ast.Assert) and any(isinstance(x, ast.Name) and x.id == 'own_native' for x in ast.walk(n))]
assert len(native_asserts) == 3
native_code = compile(ast.fix_missing_locations(ast.Module(body=native_asserts)), '<native-admission>', 'exec')
class O:
    getppid = staticmethod(lambda: 123)
class S:
    executable = '/python'
    argv = ['trainer.py', '--plan', 'plan.json', '--run', 'formal']
record = dict(state='RUNNING', pid=123, task=plan['tasks']['formal'], physical_gpu='0', command=[S.executable] + S.argv)
namespace = dict(own_native=record, os=O, sys=S, task=plan['tasks']['formal'], plan=plan)
exec(native_code, namespace)
rejected = []
for key, bad in [('state', 'COMPLETE'), ('pid', 124), ('task', 'other'), ('physical_gpu', '1'), ('command', ['/python', 'other.py'])]:
    changed = dict(record)
    changed[key] = bad
    namespace['own_native'] = changed
    try:
        exec(native_code, namespace)
    except AssertionError:
        rejected.append(key)
    else:
        raise AssertionError('native gate accepted invalid ' + key)
assert len(rejected) == 5
# CPU scalar optimizer fixture verifies disjoint groups and independent state counters.
p = torch.tensor(1.0, requires_grad=True)
h = torch.tensor(1.0, requires_grad=True)
opt = torch.optim.AdamW([dict(params=[p], lr=plan['preview_learning_rate']),
                         dict(params=[h], lr=plan['learning_rate'])], weight_decay=plan['weight_decay'])
assert len(opt.state) == 0
for _ in range(4):
    opt.zero_grad(set_to_none=True)
    (p.square() + h.square()).backward()
    opt.step()
assert [g['lr'] for g in opt.param_groups] == [1e-5, 1e-4]
assert {int(s['step'].item()) for s in opt.state.values()} == {4}
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS', torch_version=torch.__version__, cuda_initialized=False,
    archived_input_sha256=inputs, inherited48_remote_sha256=source48,
    formal_counts=counts, train_sequences=882, dev_sequences=97, train_dev_disjoint=True,
    sanity_counts=dict(tracked_frames=20, updates=4, initializes=1),
    b_parent=dict(stage='candidate_b', step=6250, native_exit_code=0, state_tensors=265,
        adapted_state_tensors=adapted, frozen_state_tensors=251, historical_optimizer_states=8,
        historical_optimizer_step=6250),
    loss_fixtures=rank_results, unchanged_gain_fixtures=gain_results,
    native_identity_valid=True, native_identity_rejections=rejected,
    optimizer_fixture=dict(initial_state_count=0, rates=[1e-5, 1e-4], final_scalar_steps=4),
    no_trainer_import=True, no_neural_forward=True, no_active_query=True), indent=2))
'''.replace('__PAYLOAD__', payload64)
(OUT / 'remote_cpu_fixture.py').write_text(remote, encoding='utf-8')
result = subprocess.run(['ssh', '2026', "env CUDA_VISIBLE_DEVICES='' /data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python -"],
                        input=remote, text=True, capture_output=True)
(OUT / 'remote_cpu_stdout.json').write_text(result.stdout, encoding='utf-8')
(OUT / 'remote_cpu_stderr.txt').write_text(result.stderr, encoding='utf-8')
assert result.returncode == 0, (result.returncode, result.stderr)
remote_result = json.loads(result.stdout)
assert remote_result['status'] == 'PASS'
summary = dict(status='PASS', plan_sha256=PLAN_SHA, source_sha256=SOURCE_SHA,
               local_sources_verified=49, local_python_ast_parsed=49, unchanged_original_sources=48,
               identical_functions=unchanged, gain_stop_ast_identical=True,
               local_inputs_verified=local_inputs, remote_inputs_verified=len(remote_result['archived_input_sha256']),
               remote_source48_verified=len(remote_result['inherited48_remote_sha256']),
               scalar_rank_fixtures=8, gain_compatibility_fixtures=6,
               native_identity_fixtures=6, optimizer_scalar_fixture=1,
               remote_receipt='remote_cpu_stdout.json')
(OUT / 'cpu_verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
