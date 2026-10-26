import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
PLAN_REL = 'reports/global_candidate_joint_quality_trajectory_plan_20261026.json'
SOURCE_REL = 'scripts/train_candidate_joint_quality_trajectory.py'
REPORT_REL = 'reports/JOINT_QUALITY_TRAIN_SOURCE_REVIEW_20261026'
PLAN_SHA = '014101511170dc2742c1cca4662035bae7a2b106f0f6e38cc10310eda090d42f'
SOURCE_SHA = 'a19e7ae96787250ded975e89f2aaeb87a0f9cb3ab7b631e4fe9294ba4d0c2331'
AGENT = '/root/review_joint_quality_train_20261026'
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = json.loads((ROOT / PLAN_REL).read_text())
assert sha(ROOT / PLAN_REL) == PLAN_SHA
assert sha(ROOT / SOURCE_REL) == SOURCE_SHA
assert all(sha(ROOT / p) == v for p, v in plan['source_sha256'].items())
verification = json.loads((TRACE / 'cpu_verification.json').read_text())
remote = json.loads((TRACE / 'remote_cpu_stdout.json').read_text())
assert verification['status'] == remote['status'] == 'PASS'
checks = {
    'source_and_plan_binding': dict(status='PASS', evidence=[SOURCE_REL + ':255', SOURCE_REL + ':337'],
        details='Exact requested plan/new-source hashes; 49 local source hashes and AST parses. Original 48 source-map entries equal the approved server26 dev plan and all 48 installed remote hashes. All 23 plan inputs hash-match on server26.'),
    'scope_and_parent': dict(status='PASS', evidence=[SOURCE_REL + ':269', SOURCE_REL + ':297', SOURCE_REL + ':307'],
        details='Archived B checkpoint is candidate_b step6250 with native COMPLETE exit0, expected config and lineage; 882 TRAIN names, 97 disjoint dev names, two rounds at valid stride5 independently give 917106 tracks,182182 updates,1764 initializations. First-sequence sanity schedule gives20 tracks/4 updates/1 initialization.'),
    'relative_quality_gradient': dict(status='PASS', evidence=[SOURCE_REL + ':88'],
        details='Absolute-IoU SmoothL1 is preserved. Added abs(target_gap)*relu(abs(target_gap)-sign(target_gap)*predicted_gap) has the correct descent direction for both modalities; ties give exactly zero rank value/gradient and satisfied margins give zero. Eight CPU fixtures pass.'),
    'gain_stop_objective': dict(status='PASS', evidence=[SOURCE_REL + ':94', SOURCE_REL + ':115'],
        details='Old two-role sampled and optional selected-extra gain/STOP body is AST-identical. Margin0.05, abs(gain)>0.01 filter and weight0.25 are unchanged. Six CPU value/gradient comparisons against the old loss are exact.'),
    'preview_gradient_connection_and_freeze': dict(status='PASS', evidence=[SOURCE_REL + ':25', SOURCE_REL + ':191', SOURCE_REL + ':316', 'dual_rgbt/model.py:50', 'dual_rgbt_global/policy.py:19', 'dual_rgbt_global/candidate_policy.py:47', 'scripts/train_global_evidence_policy.py:40'],
        details='Fresh training preview is outside no_grad; shared CNN pools/descriptors reach quality and gain context, while broad CNN features reach gain through grid_sample. There is no detach along these paths. Model.eval does not disable autograd, and Preview contains only Conv2d/GELU. Exactly6 existing preview state tensors plus4 quality and4 query_gain tensors are selected; remaining251/265 remain frozen. Fullmodel state-change and nonzero-gradient execution are mandatory next sanity checks, not claimed from CPU scalar fixtures.'),
    'optimizer_and_formal_reset': dict(status='PASS', evidence=[SOURCE_REL + ':303', SOURCE_REL + ':314', SOURCE_REL + ':319', SOURCE_REL + ':118'],
        details='Formal and sanity each load original B weights strictly, seed2027 anew and create new AdamW; no optimizer/sanity-weight restore. Disjoint preview/head groups use1e-5/1e-4 and weight_decay1e-4. Four-step CPU scalar AdamW fixture confirms rates and fresh state counters. Archived B has8 optimizer states all at6250; these are not loaded into the new optimizer.'),
    'causal_gt_teacher_and_replay': dict(status='PASS', evidence=[SOURCE_REL + ':173', SOURCE_REL + ':178', SOURCE_REL + ':187', SOURCE_REL + ':190', SOURCE_REL + ':194', 'scripts/train_candidate_evidence_policy.py:49', 'dual_rgbt_global/candidate_tracker.py:19'],
        details='First-frame GT initializes; current GT is read only after track advances. Teacher and replay use copied pre-track state/templates/history and pre-update model. Selected replay retains exact route/request/kind/rank/geometry/executed-box assertions and teacher state-preservation checks. trajectory_counts,causal_batch,selected_target,write_label are AST-identical to old trainer; original teacher/tracker sources unchanged.'),
    'native_wrapper_and_sanity_gate': dict(status='PASS', evidence=[SOURCE_REL + ':261', SOURCE_REL + ':279', 'scripts/run_task.py:18'],
        details='Distinct task/run names bind status to RUNNING wrapper parent PID, task, physical GPU0 and exact argv. Actual gate passes a matching fixture and rejects five independent identity mismatches. Formal additionally requires hash-bound sanity COMPLETE, native exit0, exact plan/source/input config and strict reload/probe receipts; it still reloads B.'),
    'probe_and_architecture': dict(status='PASS', evidence=[SOURCE_REL + ':128', 'scripts/train_candidate_evidence_policy.py:86', 'dual_rgbt/model.py:14', 'dual_rgbt_global/candidate_policy.py:87'],
        details='No architecture, model module or deployment interface change. Sanity strict reload uses all12 existing probe fields and checks the max_candidates1 legacy request/benefit equivalence. The old shared CNN, quality and gain are adapted; backbone/decode/box head/local reader are frozen. GPU inference equivalence remains a runtime admission requirement.'),
    'integrity_and_claim_scope': dict(status='PASS', evidence=['reports/RANKSTOP_TRAIN_DECISION_AUDIT_20261026.json', SOURCE_REL + ':187', 'scripts/train_candidate_evidence_policy.py:57'],
        details='Supervision uses dataset TRAIN GT and geometric IoU; no prediction-statistic score normalization. Previous decision audit supports only evolving TRAIN descriptive diagnostics, not final-weight accuracy, a causal benefit or a trajectory oracle. This source review makes no efficacy, novelty, dev/test or target-attainment claim.')
}
report = dict(
    audit_skills=['run-experiment', 'experiment-audit'], status='SOURCE_PASS', verdict='PASS',
    blocking_issues=0, blockers=[], generated_at=now, date='2026-10-26',
    agent_id=AGENT, verdict_id=AGENT + ':source:1',
    reviewer_model_requested='gpt-6-astra', reviewer_reasoning_requested='max',
    reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED', actual_route_attestation='UNATTESTED',
    reviewer_family='openai', review_independence='same-family', acceptance_status='provisional',
    plan_path=PLAN_REL, plan_sha256=PLAN_SHA, source_sha256=plan['source_sha256'],
    new_source_path=SOURCE_REL, new_source_sha256=SOURCE_SHA,
    input_sha256=plan['input_sha256'], checks=checks,
    deterministic_checks=dict(status='PASS', acceptance_status='accepted-for-listed-checks-only',
        local_source_hashes=49, local_ast_parses=49, original_source_map_exact=48,
        remote_inherited_source_hashes=48, remote_archived_input_hashes=23,
        rank_gradient_cases=8, gain_value_and_gradient_cases=6, native_identity_cases=6,
        scalar_optimizer_cases=1, torch_version=remote['torch_version'], cuda_initialized=False),
    planned_training=dict(sequence_count=882, rounds=2, seed=2027, label_stride=5,
        expected_formal_counts=remote['formal_counts'], expected_sanity_counts=remote['sanity_counts'],
        parent_checkpoint=plan['checkpoint'], parent_sha256=plan['input_sha256'][plan['checkpoint']],
        parent_stage='candidate_b', parent_step=6250, native_parent_exit_code=0,
        expected_frozen_state_tensors=251, expected_changed_state_tensors=14,
        adapted_state_tensors=remote['b_parent']['adapted_state_tensors'],
        preview_learning_rate=1e-5, head_learning_rate=1e-4, fresh_optimizer=True),
    runtime_admission_remaining='Run the planned native4-update GPU sanity, including real CNN backward/AdamW,251 frozen/14 changed, selected replay and strict reload, then full formal from B6250. This source verdict does not attest runtime GPU readiness or completion.',
    launch_contract=dict(physical_gpu=0, CUDA_DEVICE_ORDER='PCI_BUS_ID', OPENCV_FOR_THREADS_NUM='1',
        note='OPENCV_FOR_THREADS_NUM is supplied by the launcher through the inherited environment, not set by the trainer.'),
    limitations=['Semantic review is same-family/provisional; actual requested model/effort route is UNATTESTED.',
        'No neural model forward/backward was executed; only scalar loss/autograd/optimizer CPU fixtures and CPU archived checkpoint inspection.',
        'Expected251 frozen/14 changed is source/state-layout verified; actual change/freeze/gradient assertions must pass the planned real sanity.',
        'Inherited48 installed bytes verified; deployment of the new trainer/plan/review is owned by the executor and must preserve these hashes.',
        'No active dev output, progress log, PID/resource query or GPU admission was performed. No final accuracy or speed conclusion.'],
    operations=dict(source_edits=False, plan_edits=False, handoff_edits=False, git_mutation=False,
        remote_writes=False, trainer_import=False, neural_execution=False, cuda_allocation=False,
        active_resource_or_pid_query=False, active_dev_query=False,
        archived_b_cpu_checkpoint_load=True, cpu_scalar_autograd=True),
    trace_path=str(TRACE.relative_to(ROOT)).replace('\\', '/'),
    verification_path=str((TRACE / 'cpu_verification.json').relative_to(ROOT)).replace('\\', '/'),
    audited_diagnostic_sha256=sha(ROOT / 'reports/RANKSTOP_TRAIN_DECISION_AUDIT_20261026.json'))
markdown = f'''# Joint preview/quality TRAIN source review

**SOURCE_PASS; 0 blocking issues.** Fresh delegated source review under run-experiment and experiment-audit. Requested reviewer: gpt-6-astra / max; actual route: **UNATTESTED**. Same-family review; semantic acceptance is provisional.

- Plan SHA-256: `{PLAN_SHA}`.
- New trainer SHA-256: `{SOURCE_SHA}`.
- All 49 local source hashes/AST parses pass. The inherited 48 exactly match the approved server26 plan and installed files. All 23 bound archived inputs pass SHA-256 verification.

The delta is limited to adapting the existing preview CNN, adding relative-quality ranking, separate preview/head learning rates, expected state checks, loss logging and native wrapper identity. No model architecture, module, deployment interface, teacher or candidate definition changes. Original model/tracker sources are unchanged.

The new rank term is `abs(d) * relu(abs(d) - sign(d) * p)`, with GT quality gap `d` and predicted gap `p`. CPU autograd verifies that descent raises the better modality, reverses correctly for TIR, gives zero rank loss/gradient for ties, and stops after the requested margin. The small-gap case is downweighted. Eight rank cases pass. The old gain/STOP body is AST-identical; six comparisons preserve exact values and gradients, including selected-extra, neutral and threshold cases. Evidence: trainer lines88-115.

The training preview at line191 is computed outside `no_grad`. Local pools and template descriptors reach quality and gain; broad CNN features reach gain through `grid_sample` (model.py:50, global/policy.py:19, candidate_policy.py:47). No detach cuts these paths. `eval()` keeps the frozen model behavior stable without disabling gradients; Preview contains Conv2d/GELU. Archived B has265 state tensors; the selected existing6 preview +4 quality +4 gain tensors leave251 frozen. Actual nonzero CNN gradients and14 changed tensors are required by the forthcoming real sanity, not inferred from scalar tests.

Both runs strictly load the same native-complete B6250 checkpoint, reseed2027 and create fresh AdamW. Sanity weights and old optimizer state are never loaded into formal. Preview/head groups are disjoint at1e-5/1e-4; the CPU scalar optimizer starts empty and reaches four updates correctly. Archived B config/lineage matches its bound plan, with8 historical optimizer states at6250. Evidence: trainer lines269-278,303-323.

The manifest independently yields **882 sequences ×2 rounds,917106 tracked frames,182182 updates,1764 initializations**. The first-sequence sanity yields20 tracked frames/four updates/one initialization. The97 dev sequence names are disjoint. These are verified planned counts, not newly executed training results.

Current-frame GT enters only after `track(image)` advances state; the teacher uses copied pre-track state/templates/history and unchanged pre-update weights. Selected replay retains route/request/kind/rank/geometry/executed-box equality; teacher preserves tracker state. The corresponding helpers are AST-identical to RankSTOP. Existing12-field strict-reload and four-request legacy probes remain in sanity. Evidence: trainer lines173-197,128-138; train_candidate_evidence_policy.py:49-109.

The native identity gate binds the running wrapper parent PID, task, GPU0 and exact argv. One valid and five rejection fixtures pass. Formal checks the exact sanity receipt, native exit0, plan/source/input hashes and strict reload before starting anew from B. Wrapper creation records RUNNING before spawning its child (run_task.py:18-28). The launcher must supply `OPENCV_FOR_THREADS_NUM=1` as specified by the plan.

The previous saved-TRAIN decision audit remains descriptive evidence only; it does not prove this intervention will improve final-weight tracking. No new accuracy, novelty or causal-benefit claim is supported by this review.

**Validation:** 49 local source/AST checks;48 inherited remote hashes;23 archived input hashes; B6250 CPU checkpoint/config/native-status inspection; complete schedule census;21 CPU scalar/native-identity fixtures. Torch1.12.1+cu113 stayed CPU-only with CUDA uninitialized. No trainer import, neural execution, GPU allocation, active dev/process/resource query, remote write, source/plan/HANDOFF edit or Git mutation occurred.

**Remaining admission:** the executor must preserve the reviewed hashes and run the planned native four-update GPU sanity, then the full formal job from B6250 after native sanity exit0. This verdict does not claim runtime admission or training completion. No source fix is required.

Detailed machine report: `{REPORT_REL}.json`. CPU evidence and reviewer trace: `{report['trace_path']}/`.
'''
with (ROOT / (REPORT_REL + '.json')).open('x', encoding='utf-8', newline='\n') as f:
    json.dump(report, f, indent=2)
    f.write('\n')
with (ROOT / (REPORT_REL + '.md')).open('x', encoding='utf-8', newline='\n') as f:
    f.write(markdown)
prompt = '''Fresh prelaunch review under run-experiment/experiment-audit for imminent full TRAIN882 job. User prioritizes starting training promptly; focus on actual delta and blockers, do not expand experiment scope. Read applicable AGENTS, C:/Users/gb/.codex/skills/run-experiment/SKILL.md, experiment-audit/SKILL.md and shared-references/local-codex-policy.md. Local root C:/Users/gb/projects/Dual-RGBT. Paths: scripts/train_candidate_joint_quality_trajectory.py; reports/global_candidate_joint_quality_trajectory_plan_20261026.json; baseline scripts/train_candidate_rankstop_trajectory.py; scripts/train_candidate_evidence_policy.py; scripts/train_global_evidence_policy.py; dual_rgbt/model.py; dual_rgbt_global/{policy.py,candidate_policy.py,candidate_tracker.py}; reports/RANKSTOP_TRAIN_DECISION_AUDIT_20261026.json. Final plan SHA014101511170dc2742c1cca4662035bae7a2b106f0f6e38cc10310eda090d42f; new source SHAa19e7ae96787250ded975e89f2aaeb87a0f9cb3ab7b631e4fe9294ba4d0c2331. Inspect plan49-source binding and original48 unchanged; full882x2/917106track/182182updates from sameB6250 parent, own native wrapper identities and sanity->formal reset; training adapts existing preview CNN+quality/gain only, no architecture/new module; target gap weighted relative-quality hinge plus old STOP margin; distinct1e-5 preview/1e-4 heads. Check gradient direction, ties, freeze251/change14, optimizer states, causal current-GT-after-track, new CNN gradients connected and teacher/state/probe replay logic not broken. CPU fixtures only, no GPU/NN launch, active resource/PID/dev queries or source edits. If needed readonly SSH2026 archived B/config/manifest and CPU-only Torch scalar-loss tests permitted with no CUDA allocation, project /data2/gb/Dual-RGBT. Do not run trainer. Write unique reports/JOINT_QUALITY_TRAIN_SOURCE_REVIEW_20261026.json/.md; PASS format status SOURCE_PASS, blocking_issues0, plan_sha256 and source_sha256 exact; record same-family/provisional requested Astra/max actual UNATTESTED. Report real blockers promptly, no hypothetical safety layers. Return verdict, tests, files. Do not change source/plan/HANDOFF/Git.'''
common = dict(model='UNATTESTED', requested_model='gpt-6-astra', reasoning_effort='UNATTESTED',
              requested_reasoning_effort='max', reviewer_family='openai',
              review_independence='same-family', acceptance_status='provisional')
def save(name, data):
    (TRACE / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
save('run.meta.json', dict(skill='experiment-audit', companion_skill='run-experiment',
    run_id='2026-10-26_joint_quality_source_review', generated_at=now, executor='codex',
    project_dir=str(ROOT), **common))
save('001-source.request.json', dict(call_number=1, purpose='joint-quality-prelaunch-source-review',
    timestamp=now, tool='spawn_agent', files_referenced=[PLAN_REL, SOURCE_REL,
    'scripts/train_candidate_rankstop_trajectory.py','scripts/train_candidate_evidence_policy.py',
    'scripts/train_global_evidence_policy.py','dual_rgbt/model.py','dual_rgbt_global/policy.py',
    'dual_rgbt_global/candidate_policy.py','dual_rgbt_global/candidate_tracker.py',
    'reports/RANKSTOP_TRAIN_DECISION_AUDIT_20261026.json'], prompt=prompt, **common))
(TRACE / '001-source.response.md').write_text(markdown, encoding='utf-8')
save('001-source.meta.json', dict(call_number=1, purpose='joint-quality-prelaunch-source-review',
    timestamp=now, agent_id=AGENT, status='ok', verdict='SOURCE_PASS', blocking_issues=0, **common))
seal_names = [REPORT_REL + '.json', REPORT_REL + '.md']
seals = {name: sha(ROOT / name) for name in seal_names}
save('output_seals.json', dict(generated_at=now, output_sha256=seals,
    verifier_sha256=sha(TRACE / 'verify_cpu.py'), remote_fixture_sha256=sha(TRACE / 'remote_cpu_fixture.py'),
    cpu_verification_sha256=sha(TRACE / 'cpu_verification.json'), remote_receipt_sha256=sha(TRACE / 'remote_cpu_stdout.json')))
event_path = ROOT / '.aris/meta/events.jsonl'
event_path.parent.mkdir(parents=True, exist_ok=True)
with event_path.open('a', encoding='utf-8') as f:
    f.write(json.dumps(dict(event='review_trace', skill='experiment-audit',
        purpose='joint-quality-prelaunch-source-review', agent_id=AGENT,
        trace_path=report['trace_path'], status='ok')) + '\n')
print(json.dumps(dict(status='SOURCE_PASS', blocking_issues=0, outputs=seals), indent=2))
