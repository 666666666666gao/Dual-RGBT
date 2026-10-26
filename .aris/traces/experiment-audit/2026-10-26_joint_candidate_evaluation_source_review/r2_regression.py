"""SYNTHETIC ONLY: regression of the two R1 source blockers; no NN."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
FIXTURE = TRACE / 'synthetic_r2'
FIXTURE.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts'))
import joint_evaluation_contract as current


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


old = module('sealed_r1_contract', TRACE / 'R1_sealed_source/joint_evaluation_contract.py')
results = module('r2_results', ROOT / 'tools/joint_candidate_results.py')


def write(name, value):
    path = FIXTURE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    return path


observed = []


def case(name, expected, call):
    try:
        call()
        actual = 'ACCEPT'
    except AssertionError:
        actual = 'REJECT_ASSERTION'
    assert actual == expected, (name, expected, actual)
    observed.append(dict(case=name, expected=expected, actual=actual, status='PASS'))


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
config_name = 'runs/SYNTHETIC_COST0_RUN/config.json'
config = dict(cost_weight=0.0, max_candidates=8, total_parameters=103218887)
write(config_name, config)
artifacts = {config_name: current.sha(FIXTURE / config_name)}
timing = np.asarray([0.01, 0.01])
verified_run = (plan, spec, {}, [config], timing, artifacts, 385.0)
reports = []
curves = dict(PR=np.full((97, 51), 0.75), NPR=np.full((97, 51), 0.71), SR=np.full((97, 21), 0.60))
for cost in spec['dev_cost_candidates']:
    name = 'synthetic_report_%s.json' % str(cost).replace('.', 'p')
    path = FIXTURE / name
    np.savez_compressed(path.with_suffix('.npz'), **curves)
    report = dict(status='SCORED_COMPLETE', candidate_id=plan['candidate_id'],
                  checkpoint_sha256=plan['checkpoint_sha256'], benchmark='LasHeR', split='dev',
                  sequences=97, frames=54646, evaluation_plan_sha256=current.sha(FIXTURE / plan_path),
                  manifest_sha256='SYNTHETIC_MANIFEST', source_sha256={}, max_candidates=8,
                  frozen_selection_sha256=None, run_artifacts_sha256=artifacts,
                  curves_sha256=current.sha(path.with_suffix('.npz')),
                  PR=100 * float(curves['PR'][:, 20].mean()), NPR=100 * float(curves['NPR'][:, 20].mean()),
                  SR=100 * float(curves['SR'].mean()), run_names=['SYNTHETIC_COST0_RUN'],
                  cost_weight=cost, mean_heavy_tokens=385.0, fps=len(timing) / float(timing.sum()))
    write(name, report)
    reports.append(name)

case('R2_valid_cost0_full_shape_report', 'ACCEPT',
     lambda: current.checked_dev_report(FIXTURE, plan_path, plan, spec, FIXTURE / reports[0]))
case('sealed_R1_cost0_runtime_report_claims_cost0p05', 'ACCEPT',
     lambda: old.checked_dev_report(FIXTURE, plan_path, plan, spec, FIXTURE / reports[1]))
for name in reports[1:]:
    case('R2_reject_runtime_cost_mismatch_' + name, 'REJECT_ASSERTION',
         lambda name=name: current.checked_dev_report(FIXTURE, plan_path, plan, spec, FIXTURE / name))

audit = dict(status='DEV_RESULT_AUDIT_PASS', blocking_issues=0,
             scope='SYNTHETIC TEST INPUT, NOT AN ACTUAL INDEPENDENT RESULT AUDIT',
             evaluation_plan_sha256=current.sha(FIXTURE / plan_path), checkpoint_sha256=plan['checkpoint_sha256'],
             dev_reports_sha256={name: current.sha(FIXTURE / name) for name in reports})
write('synthetic_audit.json', audit)
args = SimpleNamespace(plan=plan_path, result_review='synthetic_audit.json', dev_reports=reports)
with patch.object(results, 'ROOT', FIXTURE), \
        patch.object(results, 'evaluation_plan', return_value=(plan, spec, {})), \
        patch.object(current, 'evaluation_plan', return_value=(plan, spec, {})), \
        patch.object(results, 'verify_runs', return_value=verified_run):
    case('R2_freeze_rejects_three_nominal_costs_with_cost0_runtime', 'REJECT_ASSERTION',
         lambda: results.freeze(args))
assert not (FIXTURE / plan['eligibility_decision']).exists()
assert not (FIXTURE / plan['frozen_selection']).exists()

selection = dict(status='FROZEN', candidate_id=plan['candidate_id'], selection_split='LasHeR_dev97',
                 checkpoint_sha256=plan['checkpoint_sha256'], evaluation_plan_sha256=current.sha(FIXTURE / plan_path),
                 criterion=spec['selection_criterion'], result_review='synthetic_audit.json',
                 result_review_sha256=current.sha(FIXTURE / 'synthetic_audit.json'),
                 dev_receipts=[dict(path=name, sha256=current.sha(FIXTURE / name)) for name in reports],
                 configurations={name: dict(cost_weight=0.15, max_candidates=8) for name in spec['datasets']})
write(plan['frozen_selection'], selection)
with patch.object(current, 'evaluation_plan', return_value=(plan, spec, {})):
    case('R2_selected_configuration_rejects_cost_mismatch', 'REJECT_ASSERTION',
         lambda: current.selected_configuration(FIXTURE, plan_path, current.sha(FIXTURE / plan['frozen_selection'])))

bad = copy.deepcopy(json.loads((FIXTURE / reports[0]).read_bytes()))
bad_path = FIXTURE / 'synthetic_bad_curve_shape.json'
bad_curves = dict(PR=curves['PR'][:, :21], NPR=curves['NPR'][:, :21], SR=np.full((97, 1), 0.60))
np.savez_compressed(bad_path.with_suffix('.npz'), **bad_curves)
bad['curves_sha256'] = current.sha(bad_path.with_suffix('.npz'))
bad['SR'] = 100 * float(bad_curves['SR'].mean())
write(bad_path.name, bad)
case('sealed_R1_accepts_malformed_curve_shape', 'ACCEPT',
     lambda: old.checked_dev_report(FIXTURE, plan_path, plan, spec, bad_path))
case('R2_rejects_malformed_curve_shape', 'REJECT_ASSERTION',
     lambda: current.checked_dev_report(FIXTURE, plan_path, plan, spec, bad_path))

summary = dict(status='PASS', scope='Synthetic CPU-only regression of R1 B1/B2; no real training, model, accuracy, service timing or independent result audit.',
               mocked_boundaries=['evaluation_plan readiness', 'verify_runs valid cost0 receipt'],
               checks=observed, check_count=len(observed), no_rejected_freeze_decision_written=True,
               source_sha256=current.sha(ROOT / 'scripts/joint_evaluation_contract.py'))
(TRACE / 'r2_regression_result.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
