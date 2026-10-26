"""Preliminary code-only review. No trainer import, neural execution or plan approval."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
NEW = 'scripts/train_candidate_joint_utility_trajectory.py'
OLD = 'scripts/train_candidate_joint_quality_trajectory.py'
DIAGNOSTIC = 'reports/RANKSTOP_UTILITY_TARGET_DIAGNOSTIC_20261026.json'
EXPECTED = {NEW: 'a7fa276cc4a8490ed76e0ae45ee3bdd0aa82f21f77bbfeff1c84908932c802d9',
            OLD: 'a19e7ae96787250ded975e89f2aaeb87a0f9cb3ab7b631e4fe9294ba4d0c2331'}
sources = {name: (ROOT / name).read_text() for name in EXPECTED}
for name, expected in EXPECTED.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
trees = {name: ast.parse(text) for name, text in sources.items()}
funcs = {name: {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)} for name, tree in trees.items()}
unchanged = ['trajectory_counts', 'causal_batch', 'selected_target', 'save_checkpoint', 'write_label']
for name in unchanged:
    assert ast.dump(funcs[NEW][name]) == ast.dump(funcs[OLD][name]), name
assert [ast.dump(n) for n in funcs[NEW]['controller_loss'].body[:5]] == [ast.dump(n) for n in funcs[OLD]['controller_loss'].body[:5]]
class RemoveNewLogging(ast.NodeTransformer):
    def visit_Tuple(self, node):
        node = self.generic_visit(node)
        node.elts = [n for n in node.elts if not (isinstance(n, ast.Name) and n.id in ('ustop', 'pairrank'))
                     and not (isinstance(n, ast.Constant) and n.value in ('utility_stop_loss', 'request_pair_rank_loss'))]
        return node
    def visit_Call(self, node):
        node = self.generic_visit(node)
        node.keywords = [n for n in node.keywords if n.arg not in ('utility_stop_loss', 'request_pair_rank_loss', 'query_supervision')]
        return node
for name in ('trajectory_stage', 'main'):
    reduced = RemoveNewLogging().visit(copy.deepcopy(funcs[NEW][name]))
    assert ast.dump(reduced) == ast.dump(funcs[OLD][name]), name
for name in ('controller_loss', 'trajectory_stage'):
    for method in ('candidate_gains', 'query', 'preview_state', 'track'):
        counts = [sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == method
                      for n in ast.walk(funcs[source][name])) for source in (OLD, NEW)]
        assert counts[0] == counts[1], (name, method, counts)
old_review = json.loads((ROOT / 'reports/JOINT_QUALITY_TRAIN_SOURCE_REVIEW_20261026.json').read_text())
assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in old_review['source_sha256'].items())
diagnostic = json.loads((ROOT / DIAGNOSTIC).read_text())
payload = dict(source=sources[NEW], baseline=sources[OLD], diagnostic=diagnostic)
payload64 = base64.b64encode(json.dumps(payload).encode()).decode()
remote = r'''
import ast, base64, csv, hashlib, json, os
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
torch.set_num_threads(1)
payload = json.loads(base64.b64decode('__PAYLOAD__'))
def extract(text, names):
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {'torch': torch, 'F': F}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes)), '<audited-source>', 'exec'), ns)
    return ns
ns = extract(payload['source'], ('utility_order_loss', 'controller_loss'))
utility, loss = ns['utility_order_loss'], ns['controller_loss']
old_loss = extract(payload['baseline'], ('controller_loss',))['controller_loss']
cost_local = float(np.float32(193 / 385))
def analytic(prediction, target, costs):
    total, derivatives, utilities = 0.0, [], []
    for p, t, c in zip(prediction, target, costs):
        derivative, local = 0.0, []
        for b in (0.0, float(np.float32(.05)), float(np.float32(.15))):
            u, v = t - b * c, p - b * c
            sign = 1.0 if u > 0 else (-1.0 if u < 0 else 0.0)
            gap = abs(u) - sign * v
            total += abs(u) * max(gap, 0.0)
            if gap > 0:
                derivative -= u
            local.append(u)
        derivatives.append(derivative / (3 * len(prediction)))
        utilities.append(local)
    return total / (3 * len(prediction)), derivatives, utilities
cases = [
    ('positive_query', [0.0], [.3], [cost_local]),
    ('negative_query', [.1], [-.3], [cost_local]),
    ('query_stop_cost_reversal_under', [0.0], [.02], [cost_local]),
    ('query_stop_cost_reversal_over', [.05], [.02], [cost_local]),
    ('zero_target', [.1], [0.0], [1.0]),
    ('budget_tie', [.1], [.05], [1.0]),
    ('pair_cost_positive', [.1], [.02], [1.0 - cost_local]),
    ('pair_cost_negative', [-.1], [-.02], [cost_local - 1.0]),
    ('pair_equal_cost', [0.0], [.2], [0.0]),
    ('pair_all_tied', [.1], [0.0], [0.0]),
    ('batch_dimension', [.1, -.2, .5], [.3, .05, -.1], [cost_local, 1.0, cost_local - 1.0]),
]
helper_results = []
for name, predicted, targets, costs in cases:
    p = torch.tensor(predicted, requires_grad=True)
    t, c = torch.tensor(targets), torch.tensor(costs)
    result = utility(p, t, c)
    gradient = torch.autograd.grad(result, p)[0]
    value, reference, utilities = analytic(p.detach().tolist(), t.tolist(), c.tolist())
    assert result.dim() == 0 and np.isclose(result.item(), value, atol=2e-7, rtol=2e-6), (name, result, value)
    assert torch.allclose(gradient, torch.tensor(reference), atol=2e-7, rtol=2e-6), (name, gradient, reference)
    helper_results.append(dict(name=name, loss=result.item(), gradient=gradient.tolist(), target_utilities=utilities))
assert helper_results[2]['gradient'][0] < 0 < helper_results[3]['gradient'][0]
assert helper_results[6]['gradient'][0] > 0 > helper_results[7]['gradient'][0]
assert helper_results[9]['loss'] == 0.0 and helper_results[9]['gradient'] == [0.0]
exact_cases = 0
for dtype in (torch.float32, torch.float64):
    for cost in (0., cost_local, 1., 1. - cost_local, cost_local - 1.):
        t = torch.tensor([-.8, -.15, -.05, 0., .02, .05, .15, .8], dtype=dtype)
        p = t.clone().requires_grad_()
        out = utility(p, t, torch.full_like(t, cost))
        assert out.item() == 0.0
        assert torch.count_nonzero(torch.autograd.grad(out, p)[0]).item() == 0
        exact_cases += 1
class ScalarModel:
    def __init__(self, values):
        self.values = values
        self.calls = []
        self.executor = type('Executor', (), {})()
        self.executor.token_cost = torch.tensor([0., 193. / 385., 193. / 385., 193. / 385., 1.])
    def candidate_gains(self, primary, preview, role, main_geometry, broad_geometry, queries, kinds):
        index = role if len(self.calls) < 2 else 2
        self.calls.append((role, tuple(queries.shape), tuple(kinds.shape)))
        value = self.values[index:index + 1, None]
        return torch.cat((torch.zeros_like(value), value), 1)
def invoke(fn, predictions, target_values, role, kind, selected_kind, extra, exact_quality=False):
    model = ScalarModel(predictions)
    quality = torch.tensor([[.8, .2] if exact_quality else [.2, .4]], requires_grad=True)
    target_quality = torch.tensor([[.8, .2]])
    batch = dict(main=torch.zeros(1), main_geometry=None, broad_geometry=None)
    targets = dict(quality=target_quality, primaries=[None, None], queries=[torch.zeros(1, 1, 3)] * 2,
                   gains=[torch.tensor([g]) for g in target_values[:2]])
    selected = dict(extra_query=extra, kind=selected_kind, geometry=torch.zeros(1, 3), gain=torch.tensor([target_values[2]]))
    output = fn(model, batch, {'quality': quality}, targets, kind, selected, role)
    return output, quality, model
pair_results = []
for role in range(2):
    for kind in range(4):
        for selected_kind in range(4):
            values = torch.tensor([.2, .1, 0.], requires_grad=True)
            target = [.01, .03, .08]
            output, quality, model = invoke(loss, values, target, role, kind, selected_kind, True)
            assert len(output) == 7 and all(v.dim() == 0 and torch.isfinite(v) for v in output)
            assert [x[0] for x in model.calls] == [0, 1, role]
            assert all(shape == (1, 1, 3) and ks == (1, 1) for _, shape, ks in model.calls)
            gradient = torch.autograd.grad(output[6], values, retain_graph=True)[0]
            c = model.executor.token_cost
            pred_diff = values[2] - values[role]
            target_diff = torch.tensor(target[2]) - torch.tensor(target[role])
            cost_diff = c[selected_kind + 1] - c[kind + 1]
            value, ref, _ = analytic([pred_diff.item()], [target_diff.item()], [cost_diff.item()])
            expected = torch.zeros(3)
            expected[2], expected[role] = ref[0], -ref[0]
            assert torch.allclose(gradient, expected, atol=2e-7, rtol=2e-6)
            assert gradient[1 - role].item() == 0.0
            assert torch.isclose(output[6], torch.tensor(value), atol=2e-7, rtol=2e-6)
            assert abs(gradient[2].item()) > 0
            old, old_quality, old_model = invoke(old_loss, values, target, role, kind, selected_kind, True)
            assert len(old_model.calls) == len(model.calls) == 3
            for ni, oi in ((1, 1), (3, 3), (4, 4)):
                assert torch.equal(output[ni], old[oi])
            assert torch.equal(torch.autograd.grad(output[1], quality)[0], torch.autograd.grad(old[1], old_quality)[0])
            exact = torch.tensor(target, requires_grad=True)
            exact_output, exact_q, _ = invoke(loss, exact, target, role, kind, selected_kind, True, exact_quality=True)
            assert exact_output[0].item() == 0.0
            exact_grad = torch.autograd.grad(exact_output[0], (exact, exact_q))
            assert all(torch.count_nonzero(g).item() == 0 for g in exact_grad)
            pair_results.append(dict(role=role, sampled_kind=kind, selected_kind=selected_kind, pair_gradient=gradient.tolist()))
no_extra_cases = 0
for role in range(2):
    for kind in range(4):
        values = torch.tensor([.2, -.1, 0.], requires_grad=True)
        out, _, model = invoke(loss, values, [.01, .03, .08], role, kind, 3, False)
        assert out[6].item() == 0.0 and len(model.calls) == 2
        assert torch.autograd.grad(out[0], values)[0][2].item() == 0.0
        no_extra_cases += 1
# A concrete limitation: pointwise zero does not establish conditional-mean calibration.
# For T=1 with probability1/4 and T=0 otherwise, at cost1 and no pair term,
# E[SmoothL1(p,T)+.25*utility_order_loss(p,T,1)] has nonzero derivative at E[T]=.25.
def risk(p):
    return .25 * (F.smooth_l1_loss(p, torch.ones_like(p)) + .25 * utility(p, torch.ones_like(p), torch.ones_like(p))) + \
           .75 * (F.smooth_l1_loss(p, torch.zeros_like(p)) + .25 * utility(p, torch.zeros_like(p), torch.ones_like(p)))
mean = torch.tensor([.25], requires_grad=True)
mean_risk = risk(mean)
mean_gradient = torch.autograd.grad(mean_risk, mean)[0].item()
minimizer = torch.tensor([.25 + .25 * (11. / 60.)], requires_grad=True)
minimum_risk = risk(minimizer)
minimum_gradient = torch.autograd.grad(minimum_risk, minimizer)[0].item()
assert mean_gradient < -.04 and abs(minimum_gradient) < 1e-7 and minimum_risk < mean_risk
calibration = dict(conditional_mean=.25, target_one_probability=.25, query_cost=1.,
    derivative_at_mean=mean_gradient, interior_minimizer=minimizer.item(), derivative_at_minimizer=minimum_gradient,
    risk_at_mean=mean_risk.item(), risk_at_minimizer=minimum_risk.item(),
    interpretation='CPU synthetic counterexample to a conditional-mean calibration theorem; not a measured model statistic.')
# Independently recount the already archived diagnostic, without importing its producer.
root = Path('/data2/gb/Dual-RGBT/runs/global_candidate_rankstop_trajectory_formal_20261018')
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
saved = payload['diagnostic']
assert digest(root / 'query_labels.csv') == saved['input_sha256']['labels']
assert digest(root / 'trajectory_steps.csv') == saved['input_sha256']['steps']
decisions = {}
with (root / 'trajectory_steps.csv').open(newline='') as f:
    for d in csv.DictReader(f):
        key = int(d['step'])
        assert key not in decisions
        decisions[key] = (int(d['chosen_modality']), int(d['chosen_action']), int(d['chosen_rank']),
            d['round'], d['sequence'], d['frame'], d['extra_selected_query'])
assert len(decisions) == 182182 and set(decisions) == set(range(1, 182183))
gains, kinds, purposes = [], [], []
pairs = {key: Counter() for key in ('all', 'different_kind', 'same_kind_other_rank')}
sampled, seen_extras, previous_step = {}, set(), 0
with (root / 'query_labels.csv').open(newline='') as f:
    for row in csv.DictReader(f):
        step = int(row['step'])
        if step != previous_step:
            if previous_step:
                assert set(sampled) == {0, 1}
            assert step == previous_step + 1
            sampled, previous_step = {}, step
        d = decisions[step]
        assert (row['round'], row['sequence'], row['frame']) == d[3:6]
        gain, kind, purpose = np.float32(row['delta_from_stop']), int(row['kind']), row['purpose']
        gains.append(gain); kinds.append(kind); purposes.append(purpose)
        if purpose == 'sampled':
            role = int(row['modality'])
            assert role not in sampled
            sampled[role] = row
            continue
        assert purpose == 'selected_extra' and step not in seen_extras and set(sampled) == {0, 1}
        seen_extras.add(step)
        assert (int(row['modality']), kind + 1, int(row['candidate_rank'])) == d[:3] and d[6] == 'True'
        alternative = sampled[d[0]]
        assert all(row[k] == alternative[k] for k in ('main_box', 'main_geometry', 'main_iou'))
        other_kind = int(alternative['kind'])
        same_kind = kind == other_kind
        assert not same_kind or row['candidate_rank'] != alternative['candidate_rank']
        other_gain = np.float32(alternative['delta_from_stop'])
        cost = np.float32(1. if kind == 3 else 193. / 385.)
        other_cost = np.float32(1. if other_kind == 3 else 193. / 385.)
        u, other_u = np.float32(gain - np.float32(.05) * cost), np.float32(other_gain - np.float32(.05) * other_cost)
        values = dict(pairs=1, sampled_raw_gain_better_over_0p01=int(other_gain - gain > np.float32(.01)),
            sampled_net_utility_better_over_0p01=int(other_u - u > np.float32(.01)),
            selected_negative_utility_sampled_positive=int(u < 0 < other_u),
            both_utilities_negative=int(u < 0 and other_u < 0))
        values['selected_kind_' + str(kind)] = 1
        for group in ('all', 'same_kind_other_rank' if same_kind else 'different_kind'):
            pairs[group].update(values)
assert previous_step == 182182 and set(sampled) == {0, 1}
assert all((step in seen_extras) == (d[6] == 'True') for step, d in decisions.items())
assert len(gains) == saved['labels'] == 399401
assert {k: dict(v) for k, v in pairs.items()} == saved['pairs']
g, k, purpose = np.asarray(gains, dtype=np.float32), np.asarray(kinds), np.asarray(purposes)
c = np.where(k == 3, np.float32(1), np.float32(193 / 385)).astype(np.float32)
active = np.abs(g) > np.float32(.01)
isolated = g.copy()
small = active & (np.abs(g) < np.float32(.05))
isolated[small] = np.sign(g[small]) * np.float32(.05)
groups = []
for label_purpose in ('sampled', 'selected_extra'):
    for kind in range(4):
        mask = (purpose == label_purpose) & (k == kind)
        row = dict(purpose=label_purpose, kind=kind, labels=int(mask.sum()), exact_gain_violates_fixed_margin=int((mask & small).sum()))
        for budget in (0., .05, .15):
            true = g - np.float32(budget) * c
            ideal = isolated - np.float32(budget) * c
            row['idealized_wrong_call_cost_' + str(budget)] = int((mask & (true <= 0) & (ideal > 0)).sum())
        groups.append(row)
assert groups == saved['label_groups']
target, cost, optimum = np.float32(.02), np.float32(.05) * np.float32(193 / 385), np.float32(.05)
toy = dict(raw_gain=float(target), query_cost=float(cost), calibrated_utility=float(target - cost),
    isolated_old_optimum_prediction=float(optimum), isolated_predicted_utility=float(optimum - cost),
    old_loss_at_calibrated=float(.25 * (.05 - float(target))),
    old_loss_at_isolated_optimum=float(.5 * (float(optimum) - float(target)) ** 2))
assert toy == saved['toy']
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PRELIMINARY_CPU_CHECKS_PASS', plan_approval=False, runtime_approval=False,
    torch_version=torch.__version__, cuda_initialized=False, neural_execution=False,
    utility_helper_cases=helper_results, exact_target_cases=exact_cases,
    exact_target_predictions_checked=80, controller_pair_cases=pair_results,
    controller_pair_exact_target_cases=32, no_extra_cases=no_extra_cases,
    unchanged_quality_cases=32, expected_value_calibration_counterexample=calibration,
    archived_diagnostic=dict(status='EXACT_RECOUNT_PASS', input_sha256=saved['input_sha256'],
        updates=len(decisions), labels=len(gains), selected_extra_pairs=len(seen_extras),
        pairs={key: dict(value) for key, value in pairs.items()}, label_groups=groups, toy=toy)), indent=2))
'''.replace('__PAYLOAD__', payload64)
(OUT / 'remote_cpu_fixture.py').write_text(remote, encoding='utf-8')
result = subprocess.run(['ssh', '2026', "env CUDA_VISIBLE_DEVICES='' /data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python -"],
                        input=remote, text=True, capture_output=True)
(OUT / 'remote_cpu_stdout.json').write_text(result.stdout, encoding='utf-8')
(OUT / 'remote_cpu_stderr.txt').write_text(result.stderr, encoding='utf-8')
assert result.returncode == 0, (result.returncode, result.stderr)
checked = json.loads(result.stdout)
assert checked['status'] == 'PRELIMINARY_CPU_CHECKS_PASS'
summary = dict(status='PRELIMINARY_CODE_CPU_PASS', plan_approval=False, runtime_approval=False,
    source_sha256=EXPECTED, unchanged_helper_functions=unchanged, quality_ast_unchanged=True,
    trajectory_ast_unchanged_except_logging=True, main_ast_unchanged_except_config_label=True,
    inherited_source_hashes=49, neural_and_teacher_call_sites_unchanged=True,
    utility_helper_cases=11, exact_target_cases=10, controller_pair_cases=32,
    no_extra_cases=8, unchanged_quality_cases=32,
    diagnostic_recount='399401 labels /182182 updates /35037 pairs exact',
    diagnostic_sha256=hashlib.sha256((ROOT / DIAGNOSTIC).read_bytes()).hexdigest(),
    diagnostic_producer_sha256=hashlib.sha256((ROOT / 'tools/analyze_rankstop_utility_targets.py').read_bytes()).hexdigest(),
    expected_value_calibration_counterexample=checked['expected_value_calibration_counterexample'])
(OUT / 'cpu_verification.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
