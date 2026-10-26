# Rank/STOP completed DEV result and diagnostic audit

Date: 2026-10-26T16:42:14.787197+00:00
Auditor: fresh Codex reviewer `/root/audit_rankstop_complete_dev_20261026`. Requested `gpt-6-astra` / `max`; actual routing **UNATTESTED**. Same-family, provisional.

**Verdict: WARN, 0 blocking findings.** Deterministic evidence checks pass. The registered decision **REJECT_REPAIR_NO_TEST** is supported. The original TRAIN process exit code remains unknown.

| Cost | PR (%) | NPR (%) | SR (%) | FPS | Mean ms | P95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 74.42093473 | 68.42535457 | 54.64837060 | 7.488519 | 133.537754 | 188.009027 |
| 0.05 | 74.02754237 | 68.05727549 | 54.51145918 | 7.702367 | 129.830225 | 183.889286 |
| 0.15 | 71.34527100 | 65.72693325 | 52.83381330 | 9.426419 | 106.084824 | 153.115124 |

Each row covers all97 DEV sequences/54,646 frames with54,549 track calls. Best cost0 SR54.64837060096125 is **1.7847731109900877pp below** the registered original E1 DEV SR56.43314371195134. Reference JSON/hash/NPZ means agree. These are DEV results, not test accuracy.

All runs use the same checkpoint SHA256 `527a3bc0f56f6a26e31adcc5e656f971b6bd34bbcd220251151d7db1cc228753`, with **103,218,887 total parameters** on RTX3090. Input sizes: template128, main search256, local query128, wide query256, main preview64, global preview256.

## Evidence checks

### gt_provenance: PASS

All 97 DEV annotations/54,646 frames match saved GT, migrated benchmark manifest, migrated/original split manifests, and original dataset init.txt arrays at float32 precision. TRAIN882 and DEV97 are disjoint. All 882 original TRAIN init.txt arrays also match the training manifest. Independent FP32 arithmetic reproduces all 399,401 main IoU, query IoU and gain labels exactly. Stored boxes are model outputs, but their targets are dataset annotations. Image bytes were not exhaustively rehashed; no neural replay was performed.

Evidence: `scripts/run_rankstop_dev.py:104`; `scripts/run_rankstop_dev.py:129`; `tools/score_rankstop_dev.py:49`; `tools/migrate_server26_inputs.py:12`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/remote_verification.json`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/train_geometry_verification.json`.

### score_normalization: PASS

No metric divides by prediction maxima/statistics. PR and NPR use toolkit curve index20; NPR scales coordinate errors by GT width/height. SR averages all21 success thresholds, then all97 sequences equally. Toolkit rounds GT and predictions to float32 integer-valued boxes, substitutes GT for frame0, and tests any rounded GT component<=0. Those534 invalid rows are set to error -1: toolkit PR/NPR count them as successes, SR as failures, with all rows retained in denominators. The raw-IoU diagnostic excludes260 rows with nonpositive size/nonfinite GT and uses unrounded coordinates; it is not official SR. The unchanged protocol quirks must remain explicit.

Evidence: `scripts/score_tracking.py:43`; `scripts/score_tracking.py:58`; `scripts/score_tracking.py:70`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/dataset/basedataset.py:22`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:246`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:290`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:332`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/utils.py:118`; `tools/analyze_rankstop_completed_dev.py:18`.

### result_existence: PASS

Three complete DEV runs, each with exactly97 prediction/GT/trace files,54,646 frame times and54,549 consecutive trace rows. Independently ran original toolkit classes on all raw predictions; all35,793 PR/NPR/SR curve cells and873 per-sequence scalars match saved NPZ/JSON exactly. All action/modality progress counts, frame timings, cost/plan/checkpoint bindings and native commands agree; all3 native run_task exit codes are0. 938 common local/remote artifact hashes match. Summary hashes/metrics and both2-arm/3-arm diagnostics match.

Evidence: `scripts/run_task.py:28`; `tools/score_rankstop_dev.py:32`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/local_verification.json`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/remote_comparison.json`.

### dead_code: PASS

score_rankstop_dev.main invokes unchanged score_tracking.main, which calls PR_LasHeR/NPR/SR_LasHeR and saves their arrays and per-sequence summaries. Independent invocation reproduces those outputs. Diagnostic counts were recomputed separately rather than merely rerunning the author diagnostic. No phantom/dead metric was needed for any audited claim.

Evidence: `tools/score_rankstop_dev.py:55`; `scripts/score_tracking.py:58`; `scripts/score_tracking.py:78`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/verify_local.py`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/verify_remote.py`.

### scope: WARN

One final seed2027 checkpoint evaluated over three registered cost weights on the same DEV97 split. This is not three seeds, TEST accuracy, target achievement, a paired speedup or a causal query/update ablation. TRAIN labels evolve during optimization; selected-extra comparisons use one sampled candidate from the same role/state, not a complete candidate-pool oracle. Semantic reviewer is fresh same-family/provisional; requested Astra/max routing is UNATTESTED.

Evidence: `reports/rankstop_dev_plan_server26_20261026.json:119`; `reports/RANKSTOP_COMPLETE_DEV_RESULT_20261026.json:54`; `tools/analyze_rankstop_completed_dev.py:127`; `tools/analyze_rankstop_utility_targets.py:91`.

### eval_type: PASS

Official DEV evaluation and saved TRAIN utility-label arithmetic use real dataset GT. The fixed-margin isolated optimum is a mathematical toy, not another tracking experiment.

### source_checkpoint_plan: PASS

All48 registered DEV source hashes match both local and SSH2026 bytes, all22 remote plan input hashes match, all7 toolkit hashes match, protected pretrained checkpoint hash matches. Final checkpoint SHA527a3bc0... is exact; CPU tensor inspection verifies265 finite states,257 bit-equal to original B checkpoint, only8 quality/gain states changed, and8 optimizer steps all182182. Current TRAIN source differs from recorded historical source in6 files; exact historical bytes were recovered from Git and hash-verified (RankSTOP entry needs recorded CRLF serialization). Diffs are checkpoint/host/path changes; loss, label writer, selected-target, trajectory and candidate math functions are AST-identical. Original/migrated manifest hashes differ because paths changed, while boxes/order/splits are identical.

Evidence: `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/remote_verification.json`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/source_history_verified.json`; `scripts/train_candidate_rankstop_trajectory.py:86`; `scripts/run_rankstop_dev.py:35`.

### training_terminal_provenance: WARN

Original C882 training actual process exit code remains unknown/null. COMPLETE.json, exact labels, finite tensors and optimizer steps support completion of archived artifacts, not recovery of a native training terminal receipt or replay of training-time teacher assertions. The warning remains in each of3 config,3 COMPLETE and3 scored metric files. DEV native exit0 must not be substituted for TRAIN exit0.

Evidence: `reports/RANKSTOP_C882_COMPLETION_AUDIT_20261026.json:36`; `scripts/run_rankstop_dev.py:56`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/extra_verification.json`.

### parameters_and_timing: WARN

Total103,218,887 parameters confirmed from checkpoint state excluding named BN/token-cost buffers, corroborating original model-count receipt and all three DEV configs. FPS=54,646/sum(frame_seconds), with initialization, image decoding, crop/policy/query/template work and synchronization included. Model construction/loading, result-file I/O and hash checks are outside the frame service-time denominator. Mean/P95 independently match. Hardware RTX3090; template128, main256, local128, wide256, main-preview64, global-preview256. Native cost0/.05 receipt intervals overlap3298.434100627899s; cost.15 launch/startup request OpenCV threads1, but native DEV configs do not stamp actual cv2.getNumThreads(). Prior80/1 profile settings do not establish exact DEV runtime thread counts. Report observed service FPS only; no matched speedup attribution.

Evidence: `scripts/run_rankstop_dev.py:85`; `scripts/run_rankstop_dev.py:105`; `scripts/run_rankstop_dev.py:114`; `scripts/run_rankstop_dev.py:147`; `dual_rgbt_global/candidate_tracker.py:21`; `dual_rgbt_global/candidate_tracker.py:65`; `reports/RANKSTOP_COST15_LAUNCH_20261026.json`; `reports/RANKSTOP_COST15_STARTUP_20261026.json`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/extra_verification.json`.

### dev_diagnostics: PASS

All fields of existing2-arm and new3-arm reports reproduce, including per-sequence/action raw-IoU statistics, triggers and streaks. Trigger totals8580/7864/8001; low-IoU triggers1782/1226/1681; after>=20 preceding low-IoU frames1354/946/1177. These are source-implied conditions frame%5==0 and confidence>=.65, not observed tensor writes or evidence values changed. Low post-query IoU cannot establish harmful query gain because pre-query primary boxes are absent, and separate cost trajectories have different histories.

Evidence: `dual_rgbt_global/tracker.py:30`; `dual_rgbt_global/candidate_tracker.py:52`; `tools/analyze_rankstop_completed_dev.py:65`; `tools/analyze_rankstop_completed_dev.py:82`; `tools/analyze_rankstop_completed_dev.py:130`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/local_verification.json`.

### train_pairs_and_scalar_target: PASS

All182,182 supervision keys follow registered2x882 stride5 sequence order;399,401 labels=364,364 sampled+35,037 selected-extra. All35,037 comparisons use the chosen role and identical main geometry, raw/executed primary boxes and primary IoU. Of these,33,267 are different kinds and1,770 same kind/other rank; sampled net utility is better by>0.01 in7,626, and selected utility<0<sampled in2,094. Exactly61,543 labels with0.01<abs(g)<0.05 violate fixed margin even at true gain. The isolated-label optimum gives11,680 apparent wrong-call flags at cost.05. The toy g=.02, cost=.05*193/385 has true utility-0.00506493635; scalar optimum p=.05 implies+0.02493506484. All reported counts/toy floats match; numeric grid minimum agrees. These are conditional arithmetic/objective diagnostics, not final-model calibration or deployed error rates.

Evidence: `scripts/train_candidate_rankstop_trajectory.py:86`; `scripts/train_candidate_rankstop_trajectory.py:172`; `scripts/train_candidate_rankstop_trajectory.py:199`; `scripts/train_candidate_evidence_policy.py:50`; `tools/analyze_rankstop_utility_targets.py:43`; `tools/analyze_rankstop_utility_targets.py:69`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/remote_verification.json`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/train_geometry_verification.json`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/extra_verification.json`.

### registered_gate: PASS

Reference PR/NPR/SR76.5577172393856/71.36599823369417/56.43314371195134 matches prior selected E1 JSON, its hash in original selection receipt, and its NPZ means. Registered max(SR,PR,NPR,-mean_ms) selects cost0:74.42093473363607/68.42535456840841/54.64837060096125. SR delta is-1.7847731109900877pp, so strict SR-improvement gate fails. REJECT_REPAIR_NO_TEST is supported. No test was launched by the auditor; the supplied plan/results contain no RankSTOP test execution. This is not an exhaustive assertion about unregistered attempts.

Evidence: `reports/rankstop_dev_plan_server26_20261026.json:119`; `reports/RANKSTOP_COMPLETE_DEV_RESULT_20261026.json:44`; `reports/E1_THREE_DEV_SELECTION_20261017.json:31`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/local_verification.json`; `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/extra_verification.json`.

## Diagnostic claim limits

The scalar toy is valid: for active gain0.02, the isolated loss prefers prediction0.05; at token cost0.05×193/385 this flips negative true utility into positive predicted utility. Exactly61,543 saved labels violate the fixed margin even at exact gain, and11,680 labels meet the isolated cost0.05 wrong-call condition. Those counts are not measured final-network wrong calls. The actual objective has separate gain/margin averaging and shared parameters, and SmoothL1 generally targets a Huber location functional rather than an exact conditional expectation.

The35,037 same-role pairs contain7,626 sampled requests with net utility better by>0.01 and2,094 cases with negative selected utility and positive sampled utility. This is one sampled alternative per selected-extra request at an evolving TRAIN state. It does not establish a full-pool oracle, final-model calibration, or the causal source of the DEV regression.

| Cost | Inferred update-rule triggers | Triggers at raw IoU<0.1 | Such triggers after >=20 prior bad frames |
|---:|---:|---:|---:|
| 0.0 | 8580 | 1782 | 1354 |
| 0.05 | 7864 | 1226 | 946 |
| 0.15 | 8001 | 1681 | 1177 |

These counts follow source conditions and saved confidence/frame values. No tensor-write instrumentation or saved pre-query primary boxes exist in these DEV traces.

## Preserved warnings and action

- Retain the original TRAIN exit-code warning; the three new DEV native exit0 receipts do not repair it.
- Retain the reject/no-test decision. No new TEST, target attainment, robustness or final calibration claim is supported.
- Treat FPS as observed frame service time; concurrent intervals and unstamped runtime OpenCV thread counts preclude a paired speedup claim. The cost0.15 launch/startup records request one OpenCV thread.
- Keep raw IoU, official toolkit metrics, inferred update conditions, sampled TRAIN pairs and the scalar toy separately labeled.

Audit execution: CPU-only local scoring and read-only SSH2026 archived-artifact/annotation inspection; two CPU checkpoint loads, zero model construction/forward/backward/CUDA calls, zero active-joint-training/process/log/resource reads and zero job operations. Only this report and its trace were written; source, raw results, handoff and Git were not modified.

Full input hashes and structured checks: `reports/RANKSTOP_COMPLETE_DEV_RESULT_AUDIT_20261026.json`. Trace: `.aris/traces/experiment-audit/2026-10-26_rankstop_complete_dev/`.
