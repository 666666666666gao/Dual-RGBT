import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
TRACE = Path(__file__).resolve().parent
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()
AGENT = '/root/review_joint_quality_train_20261026'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
verification = json.loads((TRACE / 'cpu_verification.json').read_text())
detail = json.loads((TRACE / 'remote_cpu_stdout.json').read_text())
assert verification['status'] == 'REVISION2_PRELIMINARY_CODE_CPU_PASS'
for path, expected in verification['source_sha256'].items():
    assert sha(ROOT / path) == expected
previous_report = sha(TRACE.parent / 'preliminary_findings.json')
assert previous_report == '411942e7db8a48d4606954d7959d8b1fcbafc1bda19ed52d1d55f390626c7181'
source = 'scripts/train_candidate_joint_gainpair_trajectory.py'
checks = {
    'signed_gain_regression': dict(status='PASS', evidence=[source + ':100', source + ':106', source + ':109'],
        details='Each actual gain is trained by half-MSE; the two or three terms are averaged. There is no query hinge, signed margin, target-dependent weight or cost-offset label. Raw-gradient fixtures equal(prediction-target)/n. Existing inference retains STOP0 and gain-lambda*token_cost.'),
    'pairing_and_gradient': dict(status='PASS', evidence=[source + ':99', source + ':107'],
        details='Selected-extra minus same-role sampled predictions and gains are compared by half-MSE. Both prediction gradients remain connected, equal/opposite; the unrelated modality has zero pair gradient.32 role/kind combinations pass. The common primary IoU cancels algebraically; floating arithmetic still uses the actual signed gain labels.'),
    'ties_no_extra_and_anchor': dict(status='PASS', evidence=[source + ':101', source + ':110'],
        details='Tie targets drive the predicted difference to0. No-extra gives0 pair loss and exactly two candidate-gain calls. A common nonzero gain shift leaves pair loss0 but yields positive raw regression, so the pair-only offset ambiguity is removed.32 exact-target,10 tie/negative/zero/anchor and8 no-extra fixtures pass.'),
    'quality_and_existing_pipeline': dict(status='PASS', evidence=[source + ':88', source + ':114', source + ':154'],
        details='Quality AST and32 value/gradient checks are unchanged. Counts,causal batch,selected-target replay,checkpoint/freeze/probe and label writer are AST-identical. Main and trajectory loop match after removing new description/logging only. All49 prior source hashes remain exact. No additional model/teacher calls or new parameters/labels are introduced.'),
    'scalar_mean_fixture': dict(status='PASS', evidence=['remote_cpu_stdout.json#/scalar_random_label_mean'],
        details='For the preceding Bernoulli(.25) scalar example, half-MSE has derivative0 at.25 and positive derivative.0458333 at the old hinge optimum.2958333, whose new risk is higher.'),
    'joint_common_state_mean_fixture': dict(status='PASS_UNDER_STATED_ASSUMPTIONS', evidence=['remote_cpu_stdout.json#/joint_random_label_mean_cases'],
        details='Twelve correlated-joint-label cases verify zero gradient at conditional means, the analytic Hessian and60 risk decompositions. For fixed common state/candidate/selection, expected risk excess is||p-mu||^2/(2n)+.5*(a^T(p-mu))^2 when paired. Positive raw anchoring makes the scored mean vector the unique unrestricted minimizer. No independence between labels is required.'),
    'claim_limits': dict(status='QUALIFIED', evidence=['remote_cpu_stdout.json#/common_state_assumptions'],
        details='Condition on the same complete state and fixed candidate/selection event; raw and pair targets must share one joint distribution with finite second moments and positive label-independent raw weights. The proof assumes freely specifiable common-state predictions. Different per-query information, finite shared parameters, shared-CNN quality loss, changing online sampling and imperfect optimization are outside it. No finite-network/population calibration or tracking-improvement guarantee.')
}
report = dict(status='REVISION2_PRELIMINARY_CODE_REVIEW_PASS', code_blocking_issues=0, blockers=[],
    plan_approval=False, runtime_approval=False, formal_plan_binding='NOT_REVIEWED', generated_at=NOW,
    agent_id=AGENT, reviewer_model_requested='gpt-6-astra', reviewer_reasoning_requested='max',
    reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED', actual_route_attestation='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional', reviewer_family='openai',
    source_sha256=verification['source_sha256'], source_snapshot='candidate_source.py',
    snapshot_sha256=verification['snapshot_sha256'], previous_preliminary_report_sha256=previous_report,
    archived_prior_candidate_sha256=verification['preserved_previous_source_sha256'],
    checks=checks, deterministic_verification=verification,
    common_state_assumptions=detail['common_state_assumptions'], guarantee_excluded=detail['guarantee_excluded'],
    scope='Revision2 source and CPU mathematical fixtures only; no formal plan binding or runtime approval.',
    operations=dict(trainer_import=False, neural_execution=False, cuda_initialization=False,
        active_training_dev_or_resource_query=False, remote_write=False, experiment_source_edit=False,
        previous_record_edit=False, plan_edit=False, job_operation=False))
markdown = r'''# Revision2 signed-gain/difference review

**Revision2 preliminary code/CPU checks pass; 0 concrete source blockers. No formal-plan or runtime approval.** Requested reviewer: gpt-6-astra/max. Actual route: UNATTESTED. Same-family/provisional.

Source SHA: `6cede0c60dbba2cb76483ffef0e0dd31571838b6bd63fb3cb8e122d538f8357b`. Exact-byte snapshot: `candidate_source.py`. The previous utility-hinge candidate and preliminary findings are preserved unchanged; that candidate is not approved for launch.

The new gain objective is the mean half-MSE over the same two sampled gains and optional selected-extra gain, plus half-MSE of the selected-extra minus same-role sampled gain difference (trainer lines94-111). All query utility/fixed-margin hinges and target-dependent weighting are removed. Quality regression/ranking remains unchanged. Existing inference still compares signed predicted gain minus token cost against STOP0.

Both predictions in the pair retain their gradient paths. For residual vector r=p-g and a vector a with+1 at selected-extra and-1 at its sampled role, raw gradient is r/n and pair gradient is a(a^T r). The other modality receives no pair gradient. Ties target a zero difference. No-extra yields zero pair loss and the original two gain predictions. Raw regression anchors absolute gain: adding a common offset can leave the pair difference correct, but incurs positive raw loss. The target difference cancels the shared main IoU algebraically; there is no claim of bitwise equivalence to a separately rounded query-IoU subtraction.

**Conditional objective identity.** Fix a complete common state, observed candidate set and selection event. Let G be their joint random gain labels, mu=E[G], and allow the prediction vector p to vary freely. Require finite second moments and positive raw-regression weights independent of the realized labels under that conditioning. For an extra pair, n=3 and

`E[L(p,G)] - E[L(mu,G)] = ||p-mu||^2 / (2n) + (a^T(p-mu))^2 / 2`.

Its Hessian is I/3+aa^T, with eigenvalues1/3,1/3,7/3. Thus the scored conditional-mean vector is the unique unrestricted minimizer. For no-extra, the two scored predictions have Hessian I/2; the absent extra prediction is not scored. Label correlation changes the constant risk term, not this minimizer. The common-main-IoU dependence is allowed.

The prior scalar Bernoulli(.25) counterexample now has gradient0 at.25; the old hinge optimum.2958333 has gradient+.0458333 and higher squared-error risk. Twelve finite-support correlated joint-label cases verify the mean gradients, Hessians and60 quadratic risk decompositions.

This identity is not a finite-network or population calibration guarantee. Actual query inputs may expose different information, weights are shared with other predictions and the quality objective, optimization is nonconvex, and online state/selection distributions change. The proof fixes the selection event and assumes freely specifiable common-state predictions. No measured calibration, accuracy, causality or novelty improvement follows from these fixtures.

All49 inherited source hashes remain exact. Five helpers are AST-identical to the previously reviewed quality trainer: trajectory_counts, causal_batch, selected_target, save_checkpoint and write_label. The loop is identical apart from new logging; main differs only in the objective description. Existing quality/CNN paths,251-frozen/14-changed checks, optimizer groups, causal GT timing, teacher state preservation, strict reload/probes and original-B formal reset remain unchanged. No teacher/model forward call or label is added.

Validation:32 pair-gradient combinations;32 exact-target cases;32 quality value/gradient comparisons;10 tie/negative/zero/common-offset cases;8 no-extra cases;one scalar random-label case;12 joint random-label cases and60 decomposition checks. Torch1.12.1+cu113 remained CPU-only with CUDA uninitialized. No trainer import, neural execution, active training/dev/resource query, remote write, experiment-source edit, previous-record edit or job operation occurred.

This is preliminary only. The future plan and completed evaluation gate require separate binding before any runtime approval. Files: `preliminary_findings.json`, `cpu_verification.json`, `remote_cpu_stdout.json`, `verify_cpu.py`, `remote_cpu_fixture.py`.
'''
(TRACE / 'preliminary_findings.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
(TRACE / 'preliminary_findings.md').write_text(markdown, encoding='utf-8')
prompt = '''Your concrete conditional-mean counterexample is accepted. Preserve preliminary record, but that utility-margin candidate will NOT launch; exact original a7fa bytes archived as .aris/traces/experiment-audit/2026-10-26_joint_utility_source_review/preliminary_candidate_source.py and untracked working draft removed. Revised simpler successor is scripts/train_candidate_joint_gainpair_trajectory.py SHA6cede0c60dbba2cb76483ffef0e0dd31571838b6bd63fb3cb8e122d538f8357b. It removes all fixed/query utility hinges; .5MSE raw signed gains (STOP is fixed0 at inference) plus .5MSE of selected-extra minus same-role sampled gain difference, no target-dependent weighting; quality/CNN training unchanged. Raw regression remains anchored, pair target cancels common main IoU; existing inference gain-lambda*cost unchanged. No new query/labels/parameters. Please code-only CPU verify gradients, paired/tie/no-extra cases, unchanged quality/teacher/freeze, and scalar/joint random-label mean minimizers under stated common-state assumptions. Do not claim finite-network/population calibration guarantee. Save revision2 source snapshot/findings separately; no plan/runtime approval yet. All3 old dev audit is completing; final plan/gate binding follows.'''
def save(name, data):
    (TRACE / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
metadata = dict(model='UNATTESTED', requested_model='gpt-6-astra', reasoning_effort='UNATTESTED',
    requested_reasoning_effort='max', reviewer_family='openai', review_independence='same-family', acceptance_status='provisional')
save('run.meta.json', dict(skill='experiment-audit', run_id='2026-10-26_joint_utility_source_review/revision2_gainpair',
    timestamp=NOW, project_dir=str(ROOT), status=report['status'], **metadata))
save('001-revision2.request.json', dict(call_number=1, purpose='gainpair-revision2-preliminary-review',
    timestamp=NOW, tool='followup_task', prompt=prompt, **metadata))
(TRACE / '001-revision2.response.md').write_text(markdown, encoding='utf-8')
save('001-revision2.meta.json', dict(call_number=1, purpose='gainpair-revision2-preliminary-review',
    timestamp=NOW, agent_id=AGENT, status='ok', plan_approval=False, runtime_approval=False, **metadata))
seals = {p.name: sha(p) for p in TRACE.iterdir() if p.is_file() and p.name != 'output_seals.json'}
save('output_seals.json', dict(timestamp=NOW, file_sha256=seals))
event = dict(event='review_trace', skill='experiment-audit', purpose='gainpair-revision2-preliminary-review',
    agent_id=AGENT, trace_path=str(TRACE.relative_to(ROOT)).replace('\\', '/'), status='ok')
with (ROOT / '.aris/meta/events.jsonl').open('a', encoding='utf-8') as f:
    f.write(json.dumps(event) + '\n')
print(json.dumps(dict(status=report['status'], code_blocking_issues=0, plan_approval=False,
    runtime_approval=False, report_sha256=seals['preliminary_findings.json'], path=str(TRACE)), indent=2))
