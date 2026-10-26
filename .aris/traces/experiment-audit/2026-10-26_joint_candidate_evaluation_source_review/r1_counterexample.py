"""SYNTHETIC ONLY: source-gate counterexamples; no NN or experimental result."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
FIXTURE = TRACE / 'synthetic_r1'
FIXTURE.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts'))
import joint_evaluation_contract as contract

loader = importlib.util.spec_from_file_location('joint_candidate_results_under_review', ROOT / 'tools/joint_candidate_results.py')
results = importlib.util.module_from_spec(loader)
loader.loader.exec_module(results)


def write(name, value):
    path = FIXTURE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    return path


plan_path = 'synthetic_plan.json'
plan = dict(candidate_id='SYNTHETIC_ONLY_NOT_A_MODEL', checkpoint_sha256='SYNTHETIC_NO_CHECKPOINT',
            max_candidates=8, frozen_selection='synthetic_selection.json',
            eligibility_decision='synthetic_eligibility.json')
write(plan_path, plan)
spec = dict(source_sha256={}, datasets={'LasHeR': {'dev': 'synthetic_manifest.json'},
            'RGBT210': {}, 'RGBT234': {}}, input_sha256={'synthetic_manifest.json': 'SYNTHETIC_MANIFEST'},
            dev_cost_candidates=[0.0, 0.05, 0.15], selection_criterion='SYNTHETIC tie-break',
            reference_decision='synthetic_reference.json')
write(spec['reference_decision'], dict(reference_dev=dict(SR=56.43314371195134)))

# The verifier result is deliberately mocked as one valid full-cost0 receipt.
# It supplies cost0 runtime configs for all three inputs. The reviewed freeze
# implementation must compare each report to that returned config itself.
timing = np.asarray([0.01, 0.01])
verified_configs = [dict(cost_weight=0.0, max_candidates=8, total_parameters=103218887)]
verified_run = (plan, spec, {}, verified_configs, timing, {}, 385.0)
reports = []
curves = dict(PR=np.full((97, 51), 0.75), NPR=np.full((97, 51), 0.71), SR=np.full((97, 21), 0.60))
for cost in spec['dev_cost_candidates']:
    name = 'synthetic_report_%s.json' % str(cost).replace('.', 'p')
    path = FIXTURE / name
    np.savez_compressed(path.with_suffix('.npz'), **curves)
    report = dict(status='SCORED_COMPLETE', candidate_id=plan['candidate_id'],
                  checkpoint_sha256=plan['checkpoint_sha256'], benchmark='LasHeR', split='dev',
                  sequences=97, frames=54646, evaluation_plan_sha256=contract.sha(FIXTURE / plan_path),
                  manifest_sha256='SYNTHETIC_MANIFEST', source_sha256={}, max_candidates=8,
                  frozen_selection_sha256=None, run_artifacts_sha256={},
                  curves_sha256=contract.sha(path.with_suffix('.npz')),
                  PR=100 * float(curves['PR'][:, 20].mean()), NPR=100 * float(curves['NPR'][:, 20].mean()),
                  SR=100 * float(curves['SR'].mean()), run_names=['SYNTHETIC_COST0_RUN'],
                  cost_weight=cost, mean_heavy_tokens=385.0, fps=len(timing) / float(timing.sum()))
    write(name, report)
    reports.append(name)
audit = dict(status='DEV_RESULT_AUDIT_PASS', blocking_issues=0,
             scope='SYNTHETIC TEST INPUT, NOT AN ACTUAL INDEPENDENT RESULT AUDIT',
             evaluation_plan_sha256=contract.sha(FIXTURE / plan_path), checkpoint_sha256=plan['checkpoint_sha256'],
             dev_reports_sha256={name: contract.sha(FIXTURE / name) for name in reports})
write('synthetic_audit.json', audit)
args = SimpleNamespace(plan=plan_path, result_review='synthetic_audit.json', dev_reports=reports)
captured = io.StringIO()
with patch.object(results, 'ROOT', FIXTURE), \
        patch.object(results, 'evaluation_plan', return_value=(plan, spec, {})), \
        patch.object(contract, 'evaluation_plan', return_value=(plan, spec, {})), \
        patch.object(results, 'verify_runs', return_value=verified_run) as verifier, \
        contextlib.redirect_stdout(captured):
    results.freeze(args)

selection = json.loads((FIXTURE / plan['frozen_selection']).read_bytes())
decision = json.loads((FIXTURE / plan['eligibility_decision']).read_bytes())
observed = dict(scope='Synthetic source-gate unit fixture only; no model, real training or accuracy/runtime validation.',
                mocked_boundaries=['evaluation_plan readiness', 'verify_runs valid cost0 receipt'],
                actual_reviewed_functions=['freeze', 'checked_dev_report', 'selected_configuration'],
                claimed_costs=spec['dev_cost_candidates'], actual_verified_costs=[0.0, 0.0, 0.0],
                verified_call_count=verifier.call_count, observed_decision=decision['status'],
                observed_selected_cost=selection['configurations']['LasHeR']['cost_weight'],
                expected='Reject claimed cost0.05/0.15 receipts whose verified runtime config is cost0.',
                result='COUNTEREXAMPLE_ACCEPTED_MISMATCH')
assert verifier.call_count == 3 and decision['status'] == 'ELIGIBLE'
assert selection['configurations']['LasHeR']['cost_weight'] == 0.15

# Independently demonstrate that threshold dimensions are not checked.
bad = copy.deepcopy(json.loads((FIXTURE / reports[0]).read_bytes()))
bad_path = FIXTURE / 'synthetic_bad_curve_shape.json'
bad_curves = dict(PR=curves['PR'][:, :21], NPR=curves['NPR'][:, :21], SR=np.full((97, 1), 0.60))
np.savez_compressed(bad_path.with_suffix('.npz'), **bad_curves)
bad['curves_sha256'] = contract.sha(bad_path.with_suffix('.npz'))
bad['SR'] = 100 * float(bad_curves['SR'].mean())
write(bad_path.name, bad)
contract.checked_dev_report(FIXTURE, plan_path, plan, spec, bad_path)
observed['malformed_curve_shape'] = dict(expected={'PR': [97, 51], 'NPR': [97, 51], 'SR': [97, 21]},
                                       accepted={k: list(v.shape) for k, v in bad_curves.items()},
                                       result='COUNTEREXAMPLE_ACCEPTED_MALFORMED_CURVES')
(TRACE / 'r1_counterexample_result.json').write_text(json.dumps(observed, indent=2) + '\n')
(TRACE / 'r1_counterexample_stdout.txt').write_text(captured.getvalue())
print(json.dumps(observed, indent=2))
