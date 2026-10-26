"""Revision2 code/CPU checks only; preserve the prior candidate record."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
NEW = 'scripts/train_candidate_joint_gainpair_trajectory.py'
OLD = 'scripts/train_candidate_joint_quality_trajectory.py'
HASHES = {NEW: '6cede0c60dbba2cb76483ffef0e0dd31571838b6bd63fb3cb8e122d538f8357b',
          OLD: 'a19e7ae96787250ded975e89f2aaeb87a0f9cb3ab7b631e4fe9294ba4d0c2331'}
for name, expected in HASHES.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
archived = OUT.parent / 'preliminary_candidate_source.py'
assert hashlib.sha256(archived.read_bytes()).hexdigest() == 'a7fa276cc4a8490ed76e0ae45ee3bdd0aa82f21f77bbfeff1c84908932c802d9'
with (OUT / 'candidate_source.py').open('xb') as f:
    f.write((ROOT / NEW).read_bytes())
sources = {name: (ROOT / name).read_text() for name in HASHES}
trees = {name: ast.parse(text) for name, text in sources.items()}
funcs = {name: {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)} for name, tree in trees.items()}
unchanged = ['trajectory_counts', 'causal_batch', 'selected_target', 'save_checkpoint', 'write_label']
for name in unchanged:
    assert ast.dump(funcs[NEW][name]) == ast.dump(funcs[OLD][name]), name
assert [ast.dump(n) for n in funcs[NEW]['controller_loss'].body[:5]] == [ast.dump(n) for n in funcs[OLD]['controller_loss'].body[:5]]
class NormalizeLogging(ast.NodeTransformer):
    def visit_Tuple(self, node):
        node = self.generic_visit(node)
        node.elts = [n for n in node.elts if not (isinstance(n, ast.Name) and n.id in ('greg', 'pair'))
            and not (isinstance(n, ast.Constant) and n.value in ('gain_regression_loss', 'request_pair_loss'))]
        return node
    def visit_Call(self, node):
        node = self.generic_visit(node)
        node.keywords = [n for n in node.keywords if n.arg not in ('gain_regression_loss', 'request_pair_loss', 'query_supervision')]
        return node
for name in ('trajectory_stage', 'main'):
    reduced = NormalizeLogging().visit(copy.deepcopy(funcs[NEW][name]))
    assert ast.dump(reduced) == ast.dump(funcs[OLD][name]), name
for name in ('controller_loss', 'trajectory_stage'):
    for method in ('candidate_gains', 'query', 'preview_state', 'track'):
        counts = [sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == method
                      for n in ast.walk(funcs[source][name])) for source in (OLD, NEW)]
        assert counts[0] == counts[1], (name, method)
old_review = json.loads((ROOT / 'reports/JOINT_QUALITY_TRAIN_SOURCE_REVIEW_20261026.json').read_text())
assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in old_review['source_sha256'].items())
payload = dict(source=sources[NEW], baseline=sources[OLD])
encoded = base64.b64encode(json.dumps(payload).encode()).decode()
remote = r'''
import ast, base64, json, os
import torch
from torch.nn import functional as F
assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
torch.set_num_threads(1)
payload = json.loads(base64.b64decode('__PAYLOAD__'))
def extract(text):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'controller_loss')
    ns = {'torch': torch, 'F': F}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node])), '<reviewed-controller>', 'exec'), ns)
    return ns['controller_loss']
loss, old_loss = extract(payload['source']), extract(payload['baseline'])
class ScalarModel:
    def __init__(self, values):
        self.values, self.calls = values, []
    def candidate_gains(self, primary, preview, role, main_geometry, broad_geometry, queries, kinds):
        index = role if len(self.calls) < 2 else 2
        self.calls.append((role, tuple(queries.shape), tuple(kinds.shape)))
        score = self.values[index:index + 1, None]
        return torch.cat((torch.zeros_like(score), score), 1)
def invoke(fn, predictions, targets, role, kind=0, selected_kind=3, extra=True, exact_quality=True):
    model = ScalarModel(predictions)
    quality = predictions.new_tensor([[.8, .2] if exact_quality else [.2, .4]], requires_grad=True)
    target_quality = predictions.new_tensor([[.8, .2]])
    target_values = predictions.new_tensor(targets) if not isinstance(targets, torch.Tensor) else targets
    batch = dict(main=predictions.new_zeros(1), main_geometry=None, broad_geometry=None)
    teacher = dict(quality=target_quality, primaries=[None, None],
        queries=[predictions.new_zeros(1, 1, 3)] * 2, gains=[target_values[i:i + 1] for i in range(2)])
    selected = dict(extra_query=extra, kind=selected_kind, geometry=predictions.new_zeros(1, 3), gain=target_values[2:3])
    result = fn(model, batch, {'quality': quality}, teacher, kind, selected, role)
    return result, quality, model
pair_cases = []
for role in range(2):
    for kind in range(4):
        for selected_kind in range(4):
            prediction = torch.tensor([.2, -.1, .0], requires_grad=True)
            target = torch.tensor([.01, .03, -.08])
            out, quality, model = invoke(loss, prediction, target, role, kind, selected_kind, exact_quality=False)
            assert len(out) == 7 and all(v.dim() == 0 and torch.isfinite(v) for v in out)
            assert [r for r, _, _ in model.calls] == [0, 1, role]
            assert all(qs == (1, 1, 3) and ks == (1, 1) for _, qs, ks in model.calls)
            residual = prediction.detach() - target
            pair_residual = residual[2] - residual[role]
            a = torch.zeros(3); a[2], a[role] = 1., -1.
            raw_gradient = torch.autograd.grad(out[5], prediction, retain_graph=True)[0]
            pair_gradient = torch.autograd.grad(out[6], prediction, retain_graph=True)[0]
            total_gradient = torch.autograd.grad(out[2], prediction, retain_graph=True)[0]
            assert torch.allclose(raw_gradient, residual / 3., atol=1e-7)
            assert torch.allclose(pair_gradient, a * pair_residual, atol=1e-7)
            assert torch.allclose(total_gradient, residual / 3. + a * pair_residual, atol=1e-7)
            assert pair_gradient[1 - role].item() == 0.0
            assert pair_gradient[2].item() == -pair_gradient[role].item()
            assert torch.isclose(out[5], .5 * residual.square().mean())
            assert torch.isclose(out[6], .5 * pair_residual.square())
            old, old_quality, old_model = invoke(old_loss, prediction, target, role, kind, selected_kind, exact_quality=False)
            for ni, oi in ((1, 1), (3, 3), (4, 4)):
                assert torch.equal(out[ni], old[oi])
            assert torch.equal(torch.autograd.grad(out[1], quality)[0], torch.autograd.grad(old[1], old_quality)[0])
            assert len(model.calls) == len(old_model.calls) == 3
            exact_prediction = target.clone().requires_grad_()
            exact, exact_q, _ = invoke(loss, exact_prediction, target, role, kind, selected_kind)
            assert exact[0].item() == 0.0
            assert all(torch.count_nonzero(g).item() == 0 for g in torch.autograd.grad(exact[0], (exact_prediction, exact_q)))
            pair_cases.append(dict(role=role, sampled_kind=kind, selected_kind=selected_kind,
                raw_gradient=raw_gradient.tolist(), pair_gradient=pair_gradient.tolist()))
special_cases = []
for name, predicted, target in [
    ('tie_wrong_difference', [.2, .2, -.1], [.1, .1, .1]),
    ('tie_exact_difference', [.2, .2, .2], [.1, .1, .1]),
    ('negative_gains', [0., 0., 0.], [-.2, -.4, -.1]),
    ('all_zero_targets', [.1, -.1, .2], [0., 0., 0.]),
    ('common_shift_anchor', [.3, .5, .4], [.1, .3, .2]),
]:
    for role in range(2):
        p, t = torch.tensor(predicted, dtype=torch.float64, requires_grad=True), torch.tensor(target, dtype=torch.float64)
        out, _, _ = invoke(loss, p, t, role)
        residual = p.detach() - t
        a = p.new_tensor([-1., 0., 1.] if role == 0 else [0., -1., 1.])
        grad = torch.autograd.grad(out[2], p)[0]
        assert torch.allclose(grad, residual / 3 + a * (residual[2] - residual[role]), atol=1e-14)
        if name in ('tie_exact_difference', 'common_shift_anchor'):
            assert out[6].item() < 1e-30 and out[5].item() > 0
        special_cases.append(dict(name=name, role=role, gain_loss=out[2].item(), pair_loss=out[6].item(), gradient=grad.tolist()))
no_extra_cases = []
for role in range(2):
    for kind in range(4):
        p, t = torch.tensor([.2, -.1, .7], requires_grad=True), torch.tensor([.01, .03, -.08])
        out, _, model = invoke(loss, p, t, role, kind, extra=False)
        grad = torch.autograd.grad(out[2], p)[0]
        expected = torch.cat(((p.detach()[:2] - t[:2]) / 2., torch.zeros(1)))
        assert out[6].item() == 0.0 and len(model.calls) == 2 and torch.allclose(grad, expected, atol=1e-7)
        no_extra_cases.append(dict(role=role, sampled_kind=kind, gradient=grad.tolist()))
# The former scalar counterexample now minimizes at the mean under squared error.
def scalar_risk(p):
    values = p.expand(3)
    zero = invoke(loss, values, values.new_zeros(3), 0, extra=False)[0][2]
    one = invoke(loss, values, values.new_ones(3), 0, extra=False)[0][2]
    return .75 * zero + .25 * one
pmean = torch.tensor([.25], dtype=torch.float64, requires_grad=True)
at_mean = scalar_risk(pmean)
mean_gradient = torch.autograd.grad(at_mean, pmean)[0].item()
pbiased = torch.tensor([.25 + 11. / 240.], dtype=torch.float64, requires_grad=True)
at_biased = scalar_risk(pbiased)
biased_gradient = torch.autograd.grad(at_biased, pbiased)[0].item()
assert abs(mean_gradient) < 1e-14 and biased_gradient > .04 and at_mean < at_biased
scalar_result = dict(mean=.25, derivative_at_mean=mean_gradient, prior_hinge_minimizer=pbiased.item(),
    derivative_at_prior_hinge_minimizer=biased_gradient, risk_at_mean=at_mean.item(), risk_at_prior_hinge_minimizer=at_biased.item())
# Conditional/common-state statement: fixed positive label-independent raw weights,
# a fixed observed candidate set and freely specifiable prediction vector.
# Correlation of the joint labels is allowed; no network/population guarantee follows.
torch.manual_seed(2027)
joint_results = []
for law in range(3):
    main_iou = torch.rand(17, 2, dtype=torch.float64)
    sampled_iou = torch.rand(17, 2, dtype=torch.float64)
    extra_iou = .6 * sampled_iou[:, 0] + .4 * torch.rand(17, dtype=torch.float64)
    probability = torch.rand(17, dtype=torch.float64); probability /= probability.sum()
    for role in range(2):
        labels = torch.cat((sampled_iou - main_iou, (extra_iou - main_iou[:, role])[:, None]), dim=1)
        # Common primary IoU cancels algebraically in the same-role target difference.
        assert torch.allclose(labels[:, 2] - labels[:, role], extra_iou - sampled_iou[:, role], atol=2e-16, rtol=0)
        for extra in (False, True):
            mu = (probability[:, None] * labels).sum(0)
            def risk(p):
                return sum(probability[i] * invoke(loss, p, labels[i], role, extra=extra)[0][2] for i in range(len(labels)))
            center = mu.detach().clone().requires_grad_()
            center_loss = risk(center)
            gradient = torch.autograd.grad(center_loss, center)[0]
            assert torch.allclose(gradient, torch.zeros_like(gradient), atol=1e-14, rtol=0)
            hessian = torch.autograd.functional.hessian(risk, center)
            if extra:
                a = center.new_tensor([-1., 0., 1.] if role == 0 else [0., -1., 1.])
                expected_hessian = torch.eye(3, dtype=torch.float64) / 3. + a[:, None] * a[None]
                scored_eigenvalues = torch.linalg.eigvalsh(hessian)
                assert scored_eigenvalues.min() > .333333333333 - 1e-12
            else:
                expected_hessian = torch.diag(center.new_tensor([.5, .5, 0.]))
                scored_eigenvalues = torch.linalg.eigvalsh(hessian[:2, :2])
                assert scored_eigenvalues.min() > .499999999999
            assert torch.allclose(hessian, expected_hessian, atol=1e-14, rtol=0)
            max_decomposition_error = 0.
            for trial in range(5):
                delta = torch.randn(3, dtype=torch.float64) * .3
                increase = risk(mu + delta) - center_loss.detach()
                expected_increase = .5 * delta @ expected_hessian @ delta
                error = abs((increase - expected_increase).item())
                max_decomposition_error = max(max_decomposition_error, error)
                assert error < 1e-13 and increase >= -1e-14
            joint_results.append(dict(law=law, role=role, extra=extra, label_vectors=17, mean=mu.tolist(),
                gradient_at_mean=gradient.tolist(), hessian=hessian.tolist(), scored_hessian_eigenvalues=scored_eigenvalues.tolist(),
                decomposition_checks=5, max_decomposition_error=max_decomposition_error))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='REVISION2_PRELIMINARY_CPU_PASS', plan_approval=False, runtime_approval=False,
    torch_version=torch.__version__, cuda_initialized=False, neural_execution=False,
    paired_cases=pair_cases, paired_exact_target_cases=32, unchanged_quality_cases=32,
    tie_negative_zero_anchor_cases=special_cases, no_extra_cases=no_extra_cases,
    scalar_random_label_mean=scalar_result, joint_random_label_mean_cases=joint_results,
    common_state_assumptions=['Fix common conditioning state and observed candidate set/selection.',
        'The same joint label distribution supplies raw and pair labels, with finite second moments.',
        'Raw regression weights are positive and independent of realized labels conditional on that state.',
        'Predictions may vary freely over the common state; actual finite shared networks need not meet this condition.'],
    guarantee_excluded='No finite-network, differing-input-information, optimization, online-selection or population calibration guarantee.'), indent=2))
'''.replace('__PAYLOAD__', encoded)
(OUT / 'remote_cpu_fixture.py').write_text(remote, encoding='utf-8')
result = subprocess.run(['ssh', '2026', "env CUDA_VISIBLE_DEVICES='' /data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python -"],
    input=remote, text=True, capture_output=True)
(OUT / 'remote_cpu_stdout.json').write_text(result.stdout, encoding='utf-8')
(OUT / 'remote_cpu_stderr.txt').write_text(result.stderr, encoding='utf-8')
assert result.returncode == 0, (result.returncode, result.stderr)
checked = json.loads(result.stdout)
assert checked['status'] == 'REVISION2_PRELIMINARY_CPU_PASS'
summary = dict(status='REVISION2_PRELIMINARY_CODE_CPU_PASS', plan_approval=False, runtime_approval=False,
    source_sha256=HASHES, preserved_previous_source_sha256='a7fa276cc4a8490ed76e0ae45ee3bdd0aa82f21f77bbfeff1c84908932c802d9',
    snapshot_sha256=hashlib.sha256((OUT / 'candidate_source.py').read_bytes()).hexdigest(),
    unchanged_functions=unchanged, quality_ast_identical=True, trajectory_ast_identical_except_logging=True,
    main_ast_identical_except_description=True, inherited_source_hashes=49, no_added_teacher_or_model_calls=True,
    pair_cases=32, pair_exact_target_cases=32, unchanged_quality_cases=32,
    tie_negative_zero_anchor_cases=10, no_extra_cases=8, scalar_random_label_case=1,
    joint_random_label_cases=12, quadratic_risk_decomposition_checks=60,
    scalar_random_label_mean=checked['scalar_random_label_mean'], common_state_assumptions=checked['common_state_assumptions'],
    guarantee_excluded=checked['guarantee_excluded'])
(OUT / 'cpu_verification.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
