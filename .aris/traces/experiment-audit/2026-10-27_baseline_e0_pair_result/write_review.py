"""Write the local fresh-review trace; no experiment or reports mutations."""
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
TRACE = '.aris/traces/experiment-audit/2026-10-27_baseline_e0_pair_result/'
RUN = 'runs/baseline_e0_paired_latency_20261027/'
TASK = 'runs/pipeline_tasks/profile_baseline_e0_pair_latency_20261027/'
P = 'tools/profile_baseline_e0_pair.py'
d = json.loads((OUT / 'deterministic_result_checks.json').read_bytes())
complete = json.loads((ROOT / (RUN + 'COMPLETE.json')).read_bytes())
plan = json.loads((ROOT / 'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json').read_bytes())
now = datetime.datetime.now(datetime.timezone.utc).isoformat()

def check(status, evidence, details):
    return {'status': status, 'evidence': evidence, 'details': details}

checks = {
 'A_gt_provenance': check('PASS', [P+':40-45', P+':99-118', 'dual_rgbt/data.py:15-19'],
    'This is a real-input causal TRAIN latency sample, not an accuracy experiment. The hash-bound manifest selects train sequences; only boxes[0] initializes each tracker. Later frames provide RGB/TIR images without later GT boxes. Predictions are checked only for finiteness; they never become reference labels. The manifest/images are not locally available and were not reverified byte-by-byte by this reviewer; successful native execution passed the manifest hash gate. No real-GT accuracy, synthetic proxy accuracy, PR/NPR/SR or new precision result is produced.'),
 'B_score_normalization': check('PASS', [P+':108-128', RUN+'frame_seconds.csv:1-2401', TRACE+'verify_results.py'],
    'The measured metrics use positive finite raw durations. FPS is frame count divided by summed durations; mean is summed durations/count; P95 is the linear 95th percentile of raw frame durations. No model output statistic enters a metric denominator. Independent stdlib recomputation matches all 24 phase metrics within 7.106e-15; decode+service=end_to_end for every row with zero residual.'),
 'C_result_existence_identity': check('PASS', [RUN+'COMPLETE.json:1-110', TASK+'status.json:1-15', TASK+'run.log:4-16', 'reports/BASELINE_E0_PAIRED_LATENCY_FIRST_NODE_20261027.json:2-20', 'reports/BASELINE_E0_PAIRED_LATENCY_NATIVE_20261027.json:3-16'],
    'All required receipts exist. Native task profile_baseline_e0_pair_latency_20261027 completed with actual exit_code0 on physical GPU0 under wrapper526316. Launch and all three native snapshots have the exact interpreter/absolute profiler/--plan command. Decoded first-node status bytes equal the archived native status; first-node log equals run.log; all four phase JSON records and final log JSON equal COMPLETE. Profiler source lines36-39 asserted RUNNING/task/GPU/exact command and pid==os.getppid() before pause; the exact reviewed source subsequently completed natively. Parent binding is established by that runtime assertion and exit receipt, not by a new live /proc query or separately archived profiler-child PID.'),
 'D_executed_timing_path': check('PASS', [P+':94-152', TASK+'run.log:4-16', 'scripts/run_task_four_gpu.py:18-32'],
    'The four phase records, 2400 ordered raw rows, final COMPLETE and native0 show the timer/statistics path executed. There are no dead or uncalled accuracy metrics in this profiler. Per-phase auxiliary-call counters are445 and310 from the measured-frame branch; they agree across COMPLETE and log but cannot be rederived from timing-only CSV because actions are not serialized.'),
 'E_scope': check('WARN', [P+':150', 'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json:135-167', 'reports/BASELINE_E0_PAIRED_LATENCY_COMPARISON_20261027.json:63-73', TRACE+'deterministic_result_checks.json'],
    'Exactly three sequences, one fixed seed2027/order, 20 causal warmup frames per sequence, then frames21..220. There are600 unique measured frame identities reused in each phase; the pooled baseline has1200 observations of those600 frames, not1200 different frames. Same server25 GPU0 RTX3090, FP32/TF32 disabled, with only the two owned same-card trainers paused. Other GPU/CPU/host tasks continued and their exact load was not held fixed. Endpoint baseline drift is small but does not establish identical intermediate load or statistical robustness. No full-test FPS, isolated-host maximum, universal speedup/cost choice, multi-seed result or current32-training-model FPS follows.'),
 'F_evaluation_type': check('PASS', [P+':42-45', P+':99-131'],
    'real_input_latency_no_accuracy; dataset-manifest GT is used only for frame0 tracker initialization. The skill accuracy categories are not applied as an accuracy certification. This is neither a synthetic-ground-truth comparison nor a self-supervised accuracy score.'),
 'raw_sample_and_resets': check('PASS', ['reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json:135-165', P+':94-123', 'lib/test/tracker/cadtrack.py:60-74', 'dual_rgbt/tracker.py:16-28', RUN+'frame_seconds.csv:1-2401'],
    'Exact ordered tuples match baseline_before/e0_cost0/e0_cost0p05/baseline_after × bluegirl/bike2/comingboy\'shead × frames21..220. All12 groups contain200 rows; there are2400 total and zero duplicate/missing keys. Source performs frame0 initialization and warmup1..20 on each phase/sequence, resetting boxes/templates/query/history/frame state. Standard online tracking and template updates remain included.'),
 'timing_boundaries_and_initialization': check('PASS', [P+':99-131', 'dual_rgbt/data.py:15-19', 'dual_rgbt/tracker.py:30-67', 'lib/test/tracker/cadtrack.py:80-122'],
    'CUDA synchronizes before the outer timer and after tracker.track. End-to-end covers RGB/TIR image read/decode/concat plus service. Service includes crop, preprocessing, transfers, forward/head, causal state/template work and completion synchronization; it excludes image decode. All12 initialization durations are separate, include frame0 decode and are excluded from FPS/P95 along with model load/construction and warmup. The P95 of pooled baseline is recomputed from1200 raw durations, not an average of phase P95s.'),
 'plan_source_checkpoint_identity': check('PASS', [P+':28-41', P+':81-92', P+':145-150', 'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json:5-11', 'reports/PAIRED_LATENCY_E0_RESTORE_20261027.json:2-7', TRACE+'deterministic_result_checks.json'],
    'Plan SHA6c9cf09b... and CSV SHA610cb731... match COMPLETE. All119 current source hashes equal the plan and SOURCE_PASS guard, with exact plan/profiler/guard bytes preserved in source-review R2 trace. The profiler rechecks every source hash after measurement. Protected baseline cbcc96ba... and inactive historical all882 E0 ee60704e... are the only checkpoint inputs; runtime hashed both before pause, strictly loaded both, and completed natively. Restore metadata agrees. This local audit did not open any checkpoint payload. Current32 weights are absent from all profiler load paths; ownership startup reads only native/config/trajectory/process metadata. Historical E0 original training provenance is not re-audited or resolved here.'),
 'owned_pause_finally_and_resumed_updates': check('PASS', [P+':46-72', P+':133-139', 'tools/launch_server25_training.py:62-86', RUN+'ownership_before_pause.json:3-98', RUN+'resume.json:1-9', 'reports/BASELINE_E0_PAIRED_LATENCY_NATIVE_20261027.json:18-115'],
    'The bound execution metadata admits exactly two own GPU0 groups. Source checks exact parent/child commands and CUDA device, single children, same UID, parent PGID==PID, child group membership and equality to GPU0 compute-client PIDs before SIGSTOP. Only groups223759/223760 are signaled; children223766/223768 belong to them. Finally SIGCONT restored states S/S/R/R after145.89977765083313s. Same native identities then advanced15750→16200 and15550→16000, +450 updates each, with finite logged-loss receipts. This is actual post-resume progress, not merely non-stopped state. Other groups were not selected.'),
 'total_parameter_counts': check('PASS', [P+':91-93', 'dual_rgbt/model.py:14-48', 'lib/models/cadtrack/cadtrack.py:21-35', 'lib/models/cadtrack/vit_care.py:45-95', 'lib/models/cadtrack/base_backbone.py:64-90', 'lib/models/layers/head.py:7-20', 'lib/models/layers/head.py:97-123', 'dual_rgbt_evidence/policy.py:13-20', 'dual_rgbt_global/candidate_policy.py:87-90', 'reports/SERVER26_MODEL_TOTAL_PARAMETERS_20261026.json:2-6', TRACE+'deterministic_result_checks.json'],
    'Independent integer arithmetic from the source/config corroborates runtime enumeration: backbone86045184 + decode10619136 + head6474117 = baseline103138437. E0 adds preview14336 + quality4865 + gain216577 =103374215. Current candidate architecture removes old gain216577 and adds query_gain61249 =103218887; inheritance introduces no extra parameter modules. token_cost and normalization statistics are buffers. Historical model source is byte-identical after reversing only the default-checkpoint path relocation. No model was instantiated and no trainable-parameter count is certified or reported; the legacy inventory trainable-count field is excluded.'),
 'pooled_comparison_report': check('PASS', ['reports/BASELINE_E0_PAIRED_LATENCY_COMPARISON_20261027.json:2-73', TRACE+'deterministic_result_checks.json'],
    'The comparison report was treated as pending, not as evidence of acceptance. Every numeric entry was checked independently: pooled baseline count/totalP/FPS/mean/P95, copied E0 rows, bracket drift, current architecture total with FPSnull, CSV hash and scope. Pooled baseline throughput uses1200/22.00842297096824 service seconds and1200/46.097051162898424 end-to-end seconds.'),
}

warnings = [
 {'id':'W1','status':'WARN','reason':'limited_fixed_order_loaded_host_sample','details':checks['E_scope']['details']},
 {'id':'W2','status':'WARN','reason':'cost_comparison_is_aggregate_not_universal',
  'details':'Cost0.05 has4.8728493699% higher aggregate service FPS and3.4174668580% higher aggregate end-to-end FPS than cost0 in this run. On bluegirl it is slightly slower:28.0109415 vs28.1151091 service FPS,21.0729406 vs21.2142594 end-to-end FPS. No claim of improvement on every sequence or optimal accuracy/cost tradeoff is supported.'},
 {'id':'W3','status':'WARN','reason':'different_acquisition_windows_not_matched_comparators',
  'details':'The earlier unpaired baseline49.9018077 service /20.8782093 e2e FPS differs from this pooled baseline by+9.26373546%/+24.68518530%. No model/algorithm optimization attribution follows; use only this bracketed acquisition for the reported pair, and do not combine with historical full-test FPS.'},
 {'id':'W4','status':'WARN','reason':'local_audit_identity_limit',
  'details':'No SSH/GPU/model/checkpoint access was performed. Manifest/images are absent locally. Their identity and historical checkpoint identity rely on the exact reviewed runtime hash/load gates plus native0 receipts. Original E0 training provenance is outside this result audit; source-review R2 already records conflicting historical provenance descriptions.'},
]
claims = [
 {'id':'C1','claim':'Baseline/E0 total parameter inventories103138437/103374215','impact':'supported'},
 {'id':'C2','claim':'Current architecture total103218887','impact':'supported_as_source_inventory_only_no_FPS_or_accuracy'},
 {'id':'C3','claim':'All four measured phase FPS/mean/P95 and pooled baseline1200-observation statistics','impact':'supported_with_exact_sample_device_timing_scope'},
 {'id':'C4','claim':'Historical E0 cost0 and cost0.05 are slower than pooled baseline in this acquisition','impact':'supported_for_this_sample'},
 {'id':'C5','claim':'E0cost0.05 has higher aggregate FPS than E0cost0 in this acquisition','impact':'supported_for_this_sample_not_every_sequence'},
 {'id':'C6','claim':'Full-test FPS, isolated-host maximum, universal speed/cost superiority or current32 speed','impact':'unsupported'},
 {'id':'C7','claim':'New accuracy, PR/NPR/SR, accuracy retention, or favorable precision/speed tradeoff','impact':'unsupported_no_accuracy_evaluation'},
]
guard = {
 'audit_skill':'experiment-audit', 'status':'RESULT_PASS', 'verdict':'WARN', 'overall_verdict':'WARN',
 'integrity_status':'pass_with_scope_qualifiers', 'blocking_issues':0, 'blockers':[], 'issues':warnings,
 'summary':'Raw timing and lifecycle evidence pass; publication is limited to the specified historical-E0 TRAIN latency sample.',
 'generated_at':now, 'date':'2026-10-27', 'agent_id':'/root/audit_baseline_e0_pair_result_20261027',
 'reviewer_model_requested':'gpt-6-astra','reviewer_reasoning_requested':'max',
 'actual_model':'UNATTESTED','reviewer_model':'UNATTESTED','reviewer_reasoning_actual':'UNATTESTED',
 'reviewer_family':'openai','review_independence':'same-family','acceptance_status':'provisional',
 'deterministic_checks':{'status':'PASS','review_independence':'deterministic','acceptance_status':'accepted',
                         'artifact':TRACE+'deterministic_result_checks.json'},
 'plan_sha256':d['audited_input_sha256']['reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json'],
 'source_sha256':plan['source_sha256'], 'frame_csv_sha256':d['audited_input_sha256'][RUN+'frame_seconds.csv'],
 'complete_sha256':d['audited_input_sha256'][RUN+'COMPLETE.json'],
 'native_status_sha256':d['audited_input_sha256'][TASK+'status.json'],
 'audited_input_sha256':d['audited_input_sha256'], 'checks':checks,
 'evaluation_type':'real_input_latency_no_accuracy', 'claims':claims,
 'recomputed_phase_statistics':d['phase_statistics'], 'pooled_baseline':d['pooled_baseline'],
 'baseline_bracket_drift':d['bracket_drift'], 'comparisons':d['comparisons'],
 'resumed_owner_progress':d['resumed_progress'],
 'total_parameter_source_arithmetic':d['total_parameter_source_arithmetic'],
 'claim_ceiling':'One fixed-order server25 RTX3090 FP32/TF32-disabled paired TRAIN timing sample:3 sequences,20 warm frames each,600 measured frames per phase,initialization separate,other-host-load continuing. Historical E0 only. No new/full-test accuracy or FPS,current32-model FPS,universal speedup or trainable-parameter claim.',
 'trace_path':TRACE, 'execution_performed':{'local_source_and_receipt_reads':True,'raw_csv_recomputation':True,
 'source_hashing':True,'source_integer_arithmetic':True,'ssh':False,'remote_query':False,'gpu':False,
 'model':False,'project_module_import':False,'checkpoint_payload_read':False,'active_weight_read':False,
 'reports_write':False,'source_or_plan_edit':False},
 'execution_note':'Initial PATH python launcher E:/Scripts/python.exe failed before script execution (No pyvenv.cfg file). Installed local CPython3.13.12 then ran stdlib-only checks successfully. No environment changes.',
}

text = '''# Baseline / historical E0 paired-latency result audit

**Result gate: RESULT_PASS; overall verdict: WARN; blockers: 0.** The raw latency, identity and training-resume evidence pass. The warnings constrain interpretation to this sample.

Requested reviewer: gpt-6-astra / max. Actual model and reasoning are UNATTESTED. Fresh same-family review; semantic acceptance is provisional. Deterministic checks are separately PASS. All work was local and CPU-only; no SSH, GPU, model import/instantiation, checkpoint payload or active-weight read occurred.

The CSV contains exactly2,400 positive finite rows: baseline-before / E0cost0 / E0cost0.05 / baseline-after, each over bluegirl, bike2 and comingboy'shead,200 measured frames21–220 after20 causal warmup frames. All12 groups, frame order and unique keys match the plan. `decode+service=end_to_end` has zero residual. All phase metrics match independent recomputation within7.106e-15. Source/timing evidence: `tools/profile_baseline_e0_pair.py:99-140`, `runs/baseline_e0_paired_latency_20261027/frame_seconds.csv:1-2401`.

| Phase | Frames | Service FPS | Service mean/P95 ms | End-to-end FPS | End-to-end mean/P95 ms |
|---|---:|---:|---:|---:|---:|
'''
for p in complete['phases']:
    s,e=p['service'],p['end_to_end']
    text += f"| {p['name']} |600|{s['fps']:.9f}|{s['mean_ms']:.6f} / {s['p95_ms']:.6f}|{e['fps']:.9f}|{e['mean_ms']:.6f} / {e['p95_ms']:.6f}|\n"
b=d['pooled_baseline'];s,e=b['service'],b['end_to_end']
text += f"| Baseline pooled |1200|{s['fps']:.9f}|{s['mean_ms']:.6f} / {s['p95_ms']:.6f}|{e['fps']:.9f}|{e['mean_ms']:.6f} / {e['p95_ms']:.6f}|\n"
text += '''
The pooled baseline is **1200/22.00842297096824 =54.5245791387663 service FPS** and **1200/46.097051162898424 =26.032033931181903 end-to-end FPS**. Its P95 values20.04553640181257 and47.49676149876905 ms use the combined raw durations. Neither FPS nor P95 is an arithmetic mean of phase summaries. These are1200 observations of the same600 unique frame identities.

Service includes crop/preprocessing/transfers/forward/head/state/template updates and final CUDA synchronization, while end-to-end additionally includes both image decodes and concatenation. Initialization includes frame0 decode and is separately recorded; model loading and warmup are excluded (`tools/profile_baseline_e0_pair.py:99-131`). No first-frame inference is hidden in a claimed full-sequence FPS.

| Phase | Initialization seconds, one per sequence |
|---|---|
'''
for p in complete['phases']:
    text += '| '+p['name']+' | '+', '.join(f'{v:.9f}' for v in p['initialization_seconds'])+' |\n'
text += '''
The baseline-after FPS exceeds baseline-before by0.2032605956% service and0.2057559965% end-to-end. This supports reporting a small endpoint drift, not identical global load throughout the run. Relative to pooled baseline, E0cost0 mean service latency is2.2678692114× and end-to-end is1.6082487955×; E0cost0.05 is2.1624941299× and1.5551036439×. E0cost0.05 versus cost0 has4.8728493699% more aggregate service FPS and3.4174668580% more aggregate end-to-end FPS. It is slightly slower on bluegirl, so the aggregate improvement is not universal even within these three sequences.

The prior unpaired baseline49.9018077 service/20.8782093 end-to-end FPS is a different acquisition window. The new pooled values differ by+9.26373546%/+24.68518530%; this is not an optimization gain. Do not combine either window with historical full-test E0 FPS to claim a matched speedup (`runs/cadtrack_baseline_latency_r2_20261027/COMPLETE.json:16-36`).

## Identity and lifecycle

All119 current source hashes match the plan and source guard. The R2 plan/profiler/guard are byte-identical to the preserved source-review trace. Plan SHA2566c9cf09b2d843c78bde967cb02f4c735cb5dadae6ced44a462d247292e665909 and CSV SHA256610cb731ebe54b6b2d2b2fb9a1f603643f70fd08a4419b82a150dd9189eafe4e match COMPLETE. The additional comparison JSON was checked directly against the CSV; all numeric fields match.

Native wrapper526316 completed the exact registered profile command on physical GPU0 with exit0. Launch/status/first-node/native JSON identities agree, decoded first-node status bytes match the native file, and run.log reproduces all phase and COMPLETE JSON exactly. Runtime source requires `pid==os.getppid()`, RUNNING, exact task/GPU/command before pausing (`tools/profile_baseline_e0_pair.py:36-39`; `scripts/run_task_four_gpu.py:18-32`; native `status.json:1-15`). This is successful runtime admission evidence, not a new live-process query.

Only groups223759/223760 with children223766/223768 were admitted and paused. The gate checks exact wrapper/child commands, UID/PGID, GPU0 clients and source-bound training metadata (`tools/profile_baseline_e0_pair.py:46-72`; `tools/launch_server25_training.py:62-86`). Finally resumed all four to S/S/R/R after145.89977765083313 seconds (`resume.json:1-9`). The same identities subsequently progressed15750→16200 and15550→16000 updates, with finite-loss receipts (`ownership_before_pause.json:3-98`; `reports/BASELINE_E0_PAIRED_LATENCY_NATIVE_20261027.json:18-115`). Resume is supported by actual update progress, not state alone.

The only checkpoint loads are protected baseline cbcc96ba... and historical inactive E0 ee60704e..., both runtime-hashed and strictly loaded before successful completion (`tools/profile_baseline_e0_pair.py:40-41,81-92`; restore receipt:2-7). The current32 training weights are not loaded or profiled. This local review did not open checkpoint payloads, and the manifest/images are unavailable locally; their identities rely on the exact runtime hash gates and native receipt. Original E0 training provenance remains outside this audit.

## Parameter and claim scope

Source arithmetic gives baseline86,045,184 backbone +10,619,136 decoder +6,474,117 head =**103,138,437 total parameters**. Historical E0 adds14,336 preview +4,865 quality +216,577 gain =**103,374,215**, a235,778 (+0.2286034%) increase. Current candidate architecture replaces gain216,577 with query_gain61,249 =**103,218,887**. This matches the supplied inventory, not a measurement of current training weights or speed. Buffers are excluded; no trainable count is reported or certified. Evidence: `dual_rgbt/model.py:14-48`, `dual_rgbt_evidence/policy.py:13-20`, `lib/models/cadtrack/cadtrack.py:21-35`, `deterministic_result_checks.json`.

Only frame0 dataset-manifest boxes initialize the causal trackers; no later GT, accuracy scorer, optimizer or fake reference is used. Evaluation type is **real-input latency, no accuracy evaluation** (`tools/profile_baseline_e0_pair.py:99-131`). The per-phase auxiliary-call counters445/310 are source/log-backed; timing-only CSV cannot independently reconstruct actions.

Supported: these four phase statistics, pooled baseline statistics, source/runtime total parameter inventories, and the observed relative speeds with explicit sample/device/timer scope. Unsupported: full-test FPS, isolated-host maximum, universal speed or cost superiority, new accuracy or accuracy retention, a precision/speed tradeoff, or any FPS/accuracy for the current32 models. The fixed order, one run, three TRAIN sequences and continuing other GPU/CPU tasks must accompany the timing comparison. Small baseline endpoint drift does not remove these limits.

## A–F integrity checks

'''
for k in list(checks)[:6]:
    c=checks[k]
    text += f"- **{k}: {c['status']}** — {c['details']} Evidence: "+'; '.join(c['evidence'])+'.\n'
text += '\nNo experiment correction is required for the qualified timing claim. Keep all scope warnings with downstream reporting. `result_guard_payload.json` contains the complete machine guard; `deterministic_result_checks.json` contains per-phase/per-sequence/pooled recomputation and hashes.\n'
for old, new in {
    'exactly2,400':'exactly 2,400', 'each600':'each 600', 'all600':'all 600', 'same600':'same 600',
    ',200 measured':', 200 measured', 'after20':'after 20', 'All12':'All 12',
    'within7.106':'within 7.106', 'baseline1200':'baseline 1200', 'by0.203':'by 0.203',
    'and0.205':'and 0.205', 'is2.267':'is 2.267', 'is1.608':'is 1.608',
    'is2.162':'is 2.162', 'and1.555':'and 1.555', 'has4.872':'has 4.872',
    'and3.417':'and 3.417', 'baseline49.901':'baseline 49.901', 'by+9.263':'by +9.263',
    'All119':'All 119', 'SHA2566c9':'SHA256 6c9', 'SHA256610':'SHA256 610',
    'wrapper526316':'wrapper 526316', 'groups223759':'groups 223759',
    'children223766':'children 223766', 'after145.899':'after 145.899',
    'progressed15750':'progressed 15750', 'and15550':'and 15550', 'baseline86,045':'baseline 86,045',
    '+10,619':'+ 10,619', '+6,474':'+ 6,474', 'adds14,336':'adds 14,336',
    '+4,865':'+ 4,865', '+216,577':'+ 216,577', 'a235,778':'a 235,778',
    'gain216,577':'gain 216,577', 'query_gain61,249':'query_gain 61,249',
    'counters445/310':'counters 445/310', 'counts445':'counts 445', 'seed2027':'seed 2027',
    'order,20':'order, 20', 'frames21':'frames 21', 'warmup1':'warmup 1',
    'frames1':'frames 1', 'contains200':'contains 200', 'contain200':'contain 200',
    'are2400':'are 2400', 'row2400':'row 2400', 'exactly2':'exactly 2',
    'of1200':'of 1200', 'from1200':'from 1200', 'are1200':'are 1200',
    'values20.045':'values 20.045', 'and47.496':'and 47.496',
    'E0cost0.05':'E0 cost 0.05', 'E0cost0':'E0 cost 0', 'versus cost0':'versus cost 0',
    'current32':'current 32', 'physical GPU0':'physical GPU 0', 'Same server25 GPU0':'Same server 25 GPU 0',
    'RTX3090':'RTX 3090', 'exit0':'exit 0', 'exit_code0':'exit_code 0',
    'first20':'first 20', 'all119':'all 119',
}.items():
    # Only human report text, never machine paths, hashes, or input artifacts.
    text = text.replace(old, new)

prompt = "Fresh experiment result integrity audit. Local only, no SSH/GPU/model/activecheckpoint access. Read direct C:/Users/gb/projects/Dual-RGBT paths: tools/profile_baseline_e0_pair.py; reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json; reports/BASELINE_E0_PAIRED_LATENCY_SOURCE_REVIEW_20261027.json; reports/BASELINE_E0_PAIRED_LATENCY_LAUNCH_20261027.json; reports/BASELINE_E0_PAIRED_LATENCY_FIRST_NODE_20261027.json; reports/BASELINE_E0_PAIRED_LATENCY_NATIVE_20261027.json; reports/PAIRED_LATENCY_E0_RESTORE_20261027.json; runs/baseline_e0_paired_latency_20261027/{COMPLETE.json,frame_seconds.csv,resume.json,ownership_before_pause.json}; runs/pipeline_tasks/profile_baseline_e0_pair_latency_20261027/{status.json,run.log}; reports/SERVER26_MODEL_TOTAL_PARAMETERS_20261026.json; reports/model_parameter_counts.json; runs/cadtrack_baseline_latency_r2_20261027/COMPLETE.json; source review trace .aris/traces/experiment-audit/2026-10-27_baseline_e0_pair_latency/. Check rawCSV exact4phases/3seq/200frames after20warm/total2400; recompute allFPS meanP95; decode+service=endtoend; checkpoint/plan/source/currentidentity actualnative0 taskgpu/command/ppid; ownedonly2 groups paused/finallyresumed with actual resumed updateprogress. Totalparams vsreal source model definitions, no trainableP claim. Initialization separate noaccuracyGT comparison; baselinebrackets drift and globalload scope; current32trainingweights neverread/profile onlyoldE0. Assess which pair comparisons supported, no fulltestFPS or universal speed/precision claim. Compute exact combined baseline frames/summedservice1200 and e2e, P95 based combinedCSV; bracket drift; ratios appropriate. Do not infer newaccuracy or use fakeGT. Write audit only .aris/traces/experiment-audit/2026-10-27_baseline_e0_pair_result/ with PASS/WARN/FAIL/blockers/evidence. Provide result-guard payload in trace ifappropriate, no reports/* overwrites. RequestedAstra/max actualUNATTESTED samefamily/provisional. CPU deterministic/source checks only no broadwholeprojectscan."
request = {'call_number':1, 'purpose':'fresh-result-integrity-review', 'tool':'spawn_agent',
           'model_requested':'gpt-6-astra','reasoning_effort_requested':'max','actual_model':'UNATTESTED',
           'files_referenced':list(d['audited_input_sha256']), 'prompt':prompt,
           'followup':'Additional primary computation artifact path only: reports/BASELINE_E0_PAIRED_LATENCY_COMPARISON_20261027.json. Please verify againstrawCSV as partaudit; don\'t trustsummary status pending. No source/plan/nativeartifact edits occurred.'}
meta = {k:guard[k] for k in ('agent_id','reviewer_model_requested','reviewer_reasoning_requested','actual_model',
                           'reviewer_family','review_independence','acceptance_status')}
meta.update(call_number=1,purpose='fresh-result-integrity-review',timestamp=now,status='ok',substantive_status='RESULT_PASS',overall_verdict='WARN')
runmeta = dict(meta,skill='experiment-audit',run_id='2026-10-27_baseline_e0_pair_result',project_dir=str(ROOT),trace_path=TRACE,
               writes_restricted_to_trace=True,external_event_log_not_written='Explicit trace-only task scope')
for name, obj in [('EXPERIMENT_AUDIT.json',guard),('result_guard_payload.json',guard),
                  ('run.meta.json',runmeta),('001-result-review.request.json',request),('001-result-review.meta.json',meta)]:
    (OUT / name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
for name in ('EXPERIMENT_AUDIT.md','001-result-review.response.md'):
    (OUT / name).write_text(text,encoding='utf-8')
for name, expected in d['audited_input_sha256'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
print(json.dumps({'status':guard['status'],'overall_verdict':guard['overall_verdict'],'blocking_issues':0,
                  'trace_path':TRACE,'written_files':[p.name for p in OUT.iterdir() if p.is_file()]},indent=2))
