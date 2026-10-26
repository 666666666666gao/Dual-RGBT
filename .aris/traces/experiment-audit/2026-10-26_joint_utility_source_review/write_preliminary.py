import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()
AGENT = '/root/review_joint_quality_train_20261026'
verification = json.loads((TRACE / 'cpu_verification.json').read_text())
remote = json.loads((TRACE / 'remote_cpu_stdout.json').read_text())
assert verification['status'] == 'PRELIMINARY_CODE_CPU_PASS'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for name, expected in verification['source_sha256'].items():
    assert sha(ROOT / name) == expected
source = 'scripts/train_candidate_joint_utility_trajectory.py'
checks = {
    'dimensions_and_costs': dict(status='PASS', evidence=[source + ':88', source + ':102', source + ':113'],
        details='Production prediction/target/cost shapes are[1]; budget[3,1] broadcasts to[3,1] and mean returns a scalar. Token cost uses the existing buffer indices kind+1. Pair costs are signed selected-minus-sampled; both positive and negative differences are valid.11 helper fixtures include vector shapes, utility ties and both cost-reversal directions.'),
    'pair_roles_and_gradient': dict(status='PASS', evidence=[source + ':104', source + ':120'],
        details='The selected-extra prediction is compared with sampled_predictions[modality] and the matching target gains. Both tensor paths remain connected, with equal/opposite pair gradients and zero gradient to the other role.32 role/sample-kind/selected-kind combinations pass. No-extra returns zero pair loss and performs only the original two gain predictions; eight cases pass.'),
    'exact_target_zero': dict(status='PASS', evidence=[source + ':90', source + ':121'],
        details='At prediction==target, estimated utility==target utility for every budget; each added hinge is zero. This also holds for the selected-minus-sampled pair if all gains are exact.10 helper batches covering80 target values acrossfloat32/float64 and32 full controller exact-target cases produce zero value/gradient.'),
    'preserved_quality_and_causal_pipeline': dict(status='PASS', evidence=[source + ':95', source + ':168', source + ':255'],
        details='Quality objective AST is identical and32 value/gradient comparisons pass. Five helpers are fully AST-identical: counts,causal batch,selected target,checkpoint/probe,labels. trajectory_stage is identical after removing two logging fields/outputs; main is identical after removing the new query-supervision description. All49 inherited source hashes still match the prior review. CNN/backbone/freeze/optimizer/reset/native admission paths are unchanged.'),
    'teacher_and_neural_call_budget': dict(status='PASS', evidence=[source + ':45', source + ':104', source + ':200'],
        details='No new tracker,teacher,model.query,preview or candidate_gains call site. Pair loss reuses the two sampled predictions and already-required selected-extra prediction. Scalar fixtures observe the same2 or3 gain calls as baseline. Added work is loss arithmetic; actual future route choices may change counts of existing conditional requests.'),
    'archived_hypothesis_arithmetic': dict(status='PASS', evidence=['tools/analyze_rankstop_utility_targets.py:25', 'reports/RANKSTOP_UTILITY_TARGET_DIAGNOSTIC_20261026.json'],
        details='Independent read-only recount matches both input hashes, all399401 labels/182182 decisions/35037 selected-extra pairs, all group counts and toy arithmetic. Pair joins require same role,state key,primary geometry/box/IoU and chosen action/rank. These remain archived evolving TRAIN diagnostics, not measured deployed error rates.'),
    'conditional_mean_calibration_claim': dict(status='LIMITATION', evidence=[source + ':88', source + ':124'],
        details='Pointwise zero does not imply a proper expected-gain/conditional-mean calibration loss. A CPU counterexample withP(T=1)=.25,T otherwise0,cost1 gives derivative-.045833334 atE[T]=.25 and an interior minimizer about.2958333 forSmoothL1+.25utility loss, without the pair term. No theorem or empirical calibration improvement is established. This is a claim restriction, not a source implementation blocker.')
}
report = dict(status='PRELIMINARY_CODE_REVIEW_PASS', code_blocking_issues=0, blockers=[],
    plan_approval=False, runtime_approval=False, formal_plan_binding='NOT_REVIEWED', generated_at=NOW,
    agent_id=AGENT, reviewer_model_requested='gpt-6-astra', reviewer_reasoning_requested='max',
    reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED', actual_route_attestation='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional', reviewer_family='openai',
    source_sha256=verification['source_sha256'],
    diagnostic_sha256=verification['diagnostic_sha256'], diagnostic_producer_sha256=verification['diagnostic_producer_sha256'],
    checks=checks, cpu_verification='cpu_verification.json', cpu_details='remote_cpu_stdout.json',
    calibration_counterexample=remote['expected_value_calibration_counterexample'],
    scope='Preliminary source/CPU review only; future formal plan binding and runtime admission were not requested and are not approved.',
    prohibited_operations_performed=dict(neural_execution=False, cuda_initialization=False,
        trainer_import=False, active_process_or_resource_query=False, active_training_or_dev_read=False,
        remote_write=False, experiment_source_edit=False, plan_edit=False, job_operation=False),
    claim_limits=['Exact per-label target consistency is supported; conditional-mean calibration is not guaranteed.',
        'Saved TRAIN objective conflicts and sampled pair opportunities are descriptive, not measured deployed wrong-call rates or final tracking improvement.',
        'No full candidate pool, trajectory-oracle, causal-benefit, efficacy or novelty conclusion.',
        'No final plan or runtime approval; leave the running quality training and previous evaluation gate unchanged.'])
markdown = '''# Preliminary joint utility source review

**Preliminary code/CPU checks pass; 0 concrete source blockers. No formal-plan or runtime approval.** Requested reviewer:gpt-6-astra/max; actual route:UNATTESTED. Same-family/provisional.

Reviewed new trainer SHA`a7fa276cc4a8490ed76e0ae45ee3bdd0aa82f21f77bbfeff1c84908932c802d9` against unchanged quality trainer SHA`a19e7ae96787250ded975e89f2aaeb87a0f9cb3ab7b631e4fe9294ba4d0c2331`. All49 inherited source hashes still match the previous review.

The new helper forms target and predicted utility using the same budgets0/.05/.15 and existing token cost. Production[1] prediction/target/cost tensors broadcast with[3,1] budgets to one scalar loss. The selected-extra pair uses selected minus sampled gain and cost for the same chosen modality. Signed cost differences correctly handle an expensive request becoming worse, or a cheaper request becoming better, as budget rises. Utility ties have zero weight. Evidence:train_candidate_joint_utility_trajectory.py:88-124.

Pair gradients reach both predictions with equal/opposite signs; the other modality has zero pair gradient. There is no detach. The no-extra branch produces zero pair loss. At exact raw targets, every STOP and pair term is exactly zero, including cost-reversal and equal-cost cases. Raw SmoothL1 remains averaged across the same two or three labels; the replacement STOP term retains weight.25 and the added pair term has weight1.

**Calibration limit:** exact per-label consistency is not conditional-mean calibration. For a synthetic targetT=1 with probability.25 andT=0 otherwise, cost1 and no pair term, the expected`SmoothL1+.25*utility_order_loss` derivative atE[T]=.25 is−.045833334. Its interior minimizer is approximately.2958333, with lower risk. CPU autograd verifies this counterexample. This does not diagnose the trained network; it rules out an unconditional expected-value calibration theorem for this objective. The source and diagnostic do not currently assert such a theorem.

Quality loss and its gradient are unchanged. Five helpers are AST-identical:trajectory_counts,causal_batch,selected_target,save_checkpoint,write_label. The trajectory loop is identical after removing new logging; main is identical after removing its objective label. Therefore existing CNN gradients,251/14 state checks,teacher/current-GT timing,state preservation,probe/reload,optimizer groups and sanity/formal reset paths are preserved. No new architecture or parameters are introduced.

No extra teacher or neural call is introduced by the loss. It reuses sampled predictions and the selected-extra prediction already computed by the preceding objective. Actual future routing can change how often the existing conditional branch runs; identical realized compute across separate training trajectories is not established by source equality.

Independent archived-CSV arithmetic exactly reproduces399401 labels,182182 update states and35037 selected-extra pairs, all diagnostic group counts and the toy values. Among these pairs,7626 have a sampled alternative with net utility better by>.01 at budget.05; the corresponding raw-gain count is5330. These are evolving TRAIN observations for one sampled alternative at the same role/state. The isolated old-loss optimum counts are hypothetical objective diagnostics, not final-network calibration or deployed wrong-call rates.

CPU validation passes11 helper cases;10 exact-target batches covering80 values across two dtypes;32 pair role/kind combinations with both gradient paths;32 controller exact-target cases;32 unchanged-quality comparisons;8 no-extra cases; and the calibration counterexample. The diagnostic recount also checks original CSV hashes and pair role/action/rank/primary joins. Torch1.12.1+cu113 stayed CPU-only with CUDA uninitialized. No trainer import,neural execution,active process/resource/training/dev read,remote write,experiment-source edit or job operation occurred.

Evidence:`cpu_verification.json`,`remote_cpu_stdout.json`,`verify_preliminary_cpu.py`,`remote_cpu_fixture.py`. This record is preliminary only. Future formal plan binding must be reviewed separately after the existing evaluation gate; this work does not alter the ongoing training.
'''
(TRACE / 'preliminary_findings.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
(TRACE / 'preliminary_findings.md').write_text(markdown, encoding='utf-8')
prompt = '''Code-only review of a possible matched-budget successor; do not launch or issue final plan approval yet. Main GPU0 training remains untouched. New local scripts/train_candidate_joint_utility_trajectory.py SHA a7fa276cc4a8490ed76e0ae45ee3bdd0aa82f21f77bbfeff1c84908932c802d9, baseline your reviewed scripts/train_candidate_joint_quality_trajectory.py unchanged a19e7ae9. New helper replaces fixed signed STOP margin with cost-matched target-magnitude margin at budgets0/.05/.15, and ranks selected-extra against already-computed same-role sampled request. No new NN/teacher queries, architecture/weights/shapes or prefix changes; only gain objective/logging. Paths for hypothesis arithmetic: tools/analyze_rankstop_utility_targets.py and reports/RANKSTOP_UTILITY_TARGET_DIAGNOSTIC_20261026.json; previous archived CSVs. Please CPU-source/gradient fixtures only (readonly SSH CPU Torch with CUDA_VISIBLE empty allowed), no GPU/active-process queries. Inspect dimensions, signs/ties/cost reversals, gradient flow across paired predictions, exact-target-zero property and no unjustified expected-value calibration theorem; ensure no added teacher compute and old quality/CNN/causal/freeze/probe paths unchanged. Save preliminary findings/CPU fixtures under .aris/traces/experiment-audit/2026-10-26_joint_utility_source_review. Do not modify code. Formal plan binding will follow ONLY after old all3dev audit/gate completes; no runtime approval now. Report concrete blockers promptly.'''
def save(name, content):
    (TRACE / name).write_text(json.dumps(content, indent=2) + '\n', encoding='utf-8')
metadata = dict(model='UNATTESTED', requested_model='gpt-6-astra', reasoning_effort='UNATTESTED',
    requested_reasoning_effort='max', reviewer_family='openai', review_independence='same-family', acceptance_status='provisional')
save('run.meta.json', dict(skill='experiment-audit', run_id='2026-10-26_joint_utility_source_review',
    timestamp=NOW, project_dir=str(ROOT), status='PRELIMINARY_CODE_REVIEW_PASS', **metadata))
save('001-preliminary.request.json', dict(call_number=1, purpose='joint-utility-preliminary-source-review',
    timestamp=NOW, tool='followup_task', prompt=prompt, **metadata))
(TRACE / '001-preliminary.response.md').write_text(markdown, encoding='utf-8')
save('001-preliminary.meta.json', dict(call_number=1, purpose='joint-utility-preliminary-source-review',
    timestamp=NOW, agent_id=AGENT, status='ok', plan_approval=False, runtime_approval=False, **metadata))
seals = {p.name: sha(p) for p in TRACE.iterdir() if p.is_file() and p.name != 'output_seals.json'}
save('output_seals.json', dict(timestamp=NOW, file_sha256=seals))
event = dict(event='review_trace', skill='experiment-audit', purpose='joint-utility-preliminary-source-review',
    agent_id=AGENT, trace_path=str(TRACE.relative_to(ROOT)).replace('\\', '/'), status='ok')
with (ROOT / '.aris/meta/events.jsonl').open('a', encoding='utf-8') as f:
    f.write(json.dumps(event) + '\n')
print(json.dumps(dict(status=report['status'], code_blocking_issues=0, plan_approval=False,
    runtime_approval=False, report_sha256=seals['preliminary_findings.json'], path=str(TRACE)), indent=2))
