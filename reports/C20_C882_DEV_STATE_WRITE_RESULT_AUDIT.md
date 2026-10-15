# C20 / C882 dev state-write result audit

**Date:** 2026-10-15  
**Overall verdict: WARN. Blocking issues: 0. Deterministic verification: PASS.**

Fresh native reviewer `/root/audit_c20_c882_state_write_20261015`. Requested `gpt-6-astra` / reasoning `max`; actual routing **UNATTESTED**. Review independence: **same-family**. Acceptance status: **provisional**. No cross-family assurance is claimed.

Both archived arm objects, all 194 per-sequence rows and all 2,087 event records match independent CPU recomputation exactly. A fresh unchanged producer run exited 0 and reproduced report, events, inventory, stdout and stderr byte for byte. No blocking defect was found in the stated descriptive comparison.

## A. Ground truth provenance: PASS

The private original manifest SHA256 is 389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc. Its annotation source is LasHeR trainingset init.txt, with disjoint train/dev/test name sets of 882/97/245 and seed 2027. The dev benchmark manifest binds that same hash.

Direct checks match all 97 dev names and order, RGB/TIR filename order and source paths, frame counts, GT arrays and initial boxes across the original manifest, dev manifest and both run configs. All 194 saved GT files reproduce byte for byte from manifest boxes cast to float32 and serialized with %.6f comma-separated LF: 109292 stored rows, representing the same 54646 unique frames in each arm.

All 194 prediction initialization rows equal their GT; none of the 109098 postinitial prediction rows is identical to its GT row. Source initializes only frame zero from GT and advances search/state with predictions. Equality checks are supporting consistency evidence, not proof of historical execution.

The previous-prediction crop in analyse_sequence is a prediction-conditioned diagnostic reference, explicitly approximate after six-decimal serialization. It is not the localization target. This comparison uses saved real GT for IoU. Original remote images/init.txt bytes and checkpoint bytes were not reacquired or loaded; provenance is bounded by the immutable supplied snapshot and archived run metadata.

**Evidence:** `scripts/prepare_data.py:7`; `scripts/prepare_data.py:11`; `scripts/prepare_data.py:15`; `scripts/prepare_data.py:17`; `scripts/prepare_data.py:24`; `scripts/run_tracking.py:25`; `scripts/run_tracking.py:81`; `scripts/run_tracking.py:83`; `scripts/run_tracking.py:98`; `scripts/run_tracking.py:112`; `scripts/run_tracking.py:113`; `dual_rgbt/tracker.py:34`; `dual_rgbt/tracker.py:42`; `scripts/analyze_full_dev_coverage.py:51`; `data/benchmarks/lasher_dev.json:1`; `C:/Users/gb/.codex_tmp/dual_rgbt_c20_c882_state_write_audit_20261015/gt_serialization.json:1`.

## B. Geometry, normalization and denominators: PASS

Independent scalar continuous-xywh arithmetic uses float32 coordinates, intersection, areas and division, then stores IoU as float64 for reductions. Every valid union is positive; GT and predictions are finite, predictions have positive size, and valid GT requires positive width and height. No IoU denominator comes from a model score or its max/mean. Mean response confidence is a model-response diagnostic, not a localization probability.

Each arm excludes 97 initialization frames and has 54549 tracked frames, 54289 valid and 260 invalid. The bad-write fraction divides simultaneous valid-GT IoU<0.1 events by valid-GT inferred write events: 1053/6430 and 1034/6584. Token means divide by all tracked frames, including invalid-GT frames. A write event counts once; inferred modality writes count one for STOP and two for non-STOP.

Inference is frame%5==0 and confidence>=0.65; low is IoU<0.1, good/predecessor crossing is IoU>=0.5. Streaks count consecutive prior low frames excluding the anchor and reset on any non-low or invalid GT. Episodes are maximal contiguous low runs. Windows are t+1 through t+H for H in 5,20,50, within the same sequence; insufficient remainder is excluded before invalid-GT checking. Recovery is the first threshold crossing, not sustained recovery.

Window means first average valid future frames, then anchors within sequence, then weight sequence means by complete-window count. Exact declared-order recomputation passes; flattening the six comparison window means changes at most 1.3877787807814457e-17 from addition order. No tolerance, adjustment, clipping or threshold change was needed.

The existing toolkit differs: it rounds boxes, includes the replaced initialization prediction, and uses equal sequence weighting. Its special rule marks GT with any nonpositive coordinate as -1; this counts as PR/NPR success but SR failure. There are 534 such stored rows per arm. The raw diagnostics instead exclude 260 postinitial nonpositive-size GT frames. These conventions must not be merged or labeled the same metric.

**Evidence:** `scripts/analyze_full_dev_coverage.py:30`; `scripts/analyze_full_dev_coverage.py:60`; `scripts/analyze_full_dev_coverage.py:62`; `scripts/analyze_full_dev_coverage.py:66`; `scripts/analyze_full_dev_coverage.py:70`; `scripts/compare_dev_state_writes.py:69`; `scripts/compare_dev_state_writes.py:94`; `scripts/compare_dev_state_writes.py:113`; `scripts/compare_dev_state_writes.py:118`; `scripts/compare_dev_state_writes.py:120`; `scripts/analyze_state_write_trajectories.py:17`; `scripts/analyze_state_write_trajectories.py:21`; `scripts/analyze_state_write_trajectories.py:28`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/dataset/basedataset.py:22`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:253`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:267`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:310`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:315`; `scripts/score_tracking.py:70`.

## C. Existence, arithmetic and actual CPU reproducibility: PASS

All 595 producer-inventory hashes match before and after. Both directories contain exactly 97 prediction, 97 GT and 97 trace files plus config/COMPLETE/progress/frame_seconds, with contiguous trace frame IDs 1..N-1. The full reviewer input seal additionally covers listed artifacts, frozen sources and historical receipts.

The fresh independent verifier imports no project metric/analysis functions. Its 13195 checks include 8501 exact numeric leaf comparisons across both complete arm objects, all 194 per-sequence rows, every count/mean/fraction/action/predecessor/streak/episode/window field, all deltas, prior C20 temporal results and receipt projections. All 2087 private comparison event rows reproduce byte for byte. It also reconstructs progress receipts and frame_seconds/COMPLETE arithmetic.

All 11 numeric comparison Markdown table rows and numeric narrative claims are checked. Independently recomputed legacy cost0.05 PR/NPR/SR and all 194 legacy per-sequence rows match exactly, supporting the cited SR direction. The previous temporal audit tables and all 18 earlier document table rows also agree with the fresh temporal calculation. Historical preservation and runtime numbers are checked against retained bytes/receipts, not relabeled as new observations.

Fresh actual CPU producer execution exited 0, with empty stderr. JSON report, event CSV, inventory JSON, stdout and stderr are byte-identical to R3. The unchanged producer runtime was 2.588900899980217 seconds using the existing Python3.12.12/NumPy1.26.4 environment.

Original R1 failure remains preserved. Its NumPy int32 serialization failure is consistent with its source; R2 changes only that int cast, R3 only relocates root to the repository. Their input inventories differ only at producer identity, and all R1/R2/R3 event bytes agree. R2/R3 recorded exit0 and stdout/report hashes agree. These original exits are receipt-only evidence.

One reviewer supplemental attempt failed two assertions because it assumed training manifest-prefix order. The frozen trainer actually uses RandomState(seed).permutation before slicing. The failed script, evidence, stdout, stderr and exit1 transport were preserved; a separate correction checked only those two failed assertions against the real rule and exited0. No input or production change was made; zero unresolved mismatches remain.

**Evidence:** `scripts/compare_dev_state_writes.py:45`; `scripts/compare_dev_state_writes.py:47`; `scripts/compare_dev_state_writes.py:91`; `scripts/compare_dev_state_writes.py:126`; `scripts/compare_dev_state_writes.py:130`; `reports/C20_C882_DEV_STATE_WRITE_COMPARISON.json:33`; `reports/C20_C882_DEV_STATE_WRITE_COMPARISON.json:103`; `reports/C20_C882_DEV_STATE_WRITE_COMPARISON.json:1760`; `reports/C20_C882_DEV_STATE_WRITE_COMPARISON.json:1830`; `reports/C20_C882_DEV_STATE_WRITE_COMPARISON.json:3483`; `reports/C20_C882_DEV_STATE_WRITE_COMPARISON.md:7`; `reports/C20_C882_DEV_STATE_WRITE_COMPARISON.md:19`; `reports/C20_C882_DEV_STATE_WRITE_COMPARISON.md:23`; `C:/Users/gb/.codex_tmp/dual_rgbt_c20_c882_state_write_20261015/R1_producer.py:99`; `C:/Users/gb/.codex_tmp/dual_rgbt_c20_c882_state_write_20261015/R1_failure.json:3`; `C:/Users/gb/.codex_tmp/dual_rgbt_c20_c882_state_write_20261015/R2.transport.json:1`; `C:/Users/gb/.codex_tmp/dual_rgbt_c20_c882_state_write_20261015/R3.transport.json:1`; `scripts/train_trajectory_policy.py:43`; `C:/Users/gb/.codex_tmp/dual_rgbt_c20_c882_state_write_audit_20261015/independent_evidence.json:1`; `C:/Users/gb/.codex_tmp/dual_rgbt_c20_c882_state_write_audit_20261015/fresh_transport.json:1`; `C:/Users/gb/.codex_tmp/dual_rgbt_c20_c882_state_write_audit_20261015/supplemental_correction.json:1`.

## D. Execution flow and frozen source bindings: PASS

Read the comparison entry and imported helper functions line by line. It calls analyse_sequence for every sequence, uses its float32-derived IoU and trace arrays, calls future_summary for every horizon, then writes the computed counts/windows/deltas and event rows. The helper summarize is executed inside analyse_sequence but its unused coverage aggregates are not advertised as new results. Separate helper CLI main functions are not invoked by this comparison.

Hard-coded totals are assertions against calculations, not substituted observations; the independent computation confirms them. A fresh actual producer rerun establishes that this path is callable and flows to the stored output.

All 64 e1_paired_latency_plan source bindings and all 68 scale_candidate_training_plan source bindings match actual local bytes before and after. The added comparison CPU analyzer is outside those NN path inventories; it does not change the frozen NN source. The plans themselves retain their declared future/waiting scope. This is a result audit, not a new NN-source admission, plan approval or training-completion certificate.

**Evidence:** `scripts/compare_dev_state_writes.py:13`; `scripts/compare_dev_state_writes.py:14`; `scripts/compare_dev_state_writes.py:66`; `scripts/compare_dev_state_writes.py:108`; `scripts/compare_dev_state_writes.py:109`; `scripts/compare_dev_state_writes.py:124`; `scripts/compare_dev_state_writes.py:153`; `scripts/analyze_full_dev_coverage.py:19`; `scripts/analyze_full_dev_coverage.py:85`; `scripts/analyze_state_write_trajectories.py:13`; `reports/e1_paired_latency_plan.json:4`; `reports/e1_paired_latency_plan.json:123`; `reports/scale_candidate_training_plan.json:10`; `reports/scale_candidate_training_plan.json:127`; `reports/scale_candidate_training_plan.json:128`; `C:/Users/gb/.codex_tmp/dual_rgbt_c20_c882_state_write_audit_20261015/frozen_source_bindings.json:1`.

## E. Scope, dependency and interpretation: WARN

The comparison is one archived dev run per C20/C882 checkpoint, original E0, cost0.05, the same97 dev sequences. C20 checkpoint is 94be6239f7f6864cc193fff7c21d67c9479924c9432039c6ff16cd0d1223f15f; C882 is ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c. Source seed2027 and both training-config seeds agree. Seeded training name order is independently reproduced. Training configs/COMPLETE report20 versus882 sequences,2rounds,3504 versus182182 updates and17708 versus917106 track calls; frame exposure is checked against manifest lengths. Checkpoint-byte contents and historical model execution remain archive attestations.

Write status is inferred conditional on the frozen tracker rule, not a logged runtime write. Archive run configs bind checkpoint metadata but contain no full historical inference-source hash. Matching current NN plan hashes does not independently prove which source bytes the historical process loaded. This limits any literal historical-write assertion; retain the word inferred.

Same cost coefficient is not same executed work: mean recorded heavy tokens differ549.2207739830244 versus529.6723312984656. Actions and states are trajectory-dependent; training exposure and training sequence range both differ. This is not a matched-state intervention, matched-compute comparison, seed uncertainty estimate or isolated causal training-range ablation.

Predecessors and streaks show that most bad inferred writes occur during existing low-IoU runs. Future windows overlap, events within a sequence depend on one another, and valid/full-window selection introduces horizon-specific censoring. A crossing of0.5 is not stable recovery. No randomized/paired write suppression, template rollback, main-only box or rejected-query counterfactual is saved, so neither causal template harm nor negative auxiliary gain follows.

The current C882 all-train frozen selection is cost0, while this diagnostic uses cost0.05. This audit supplies no current test/full-benchmark/E1 results, speedup, target attainment or evidence that more quality/gain training solved the write problem. Existing comparison prose retains these limits. The Markdown line32 assertion about two currently-running test shards is an executor operational statement and was not independently queried in this no-SSH audit.

Semantic assurance is fresh-context, same-family and provisional. Requested routing is gpt-6-astra with reasoning max; actual model/effort routing is UNATTESTED. No cross-family assurance or unavailable reviewer metadata is claimed.

**Evidence:** `reports/full_pipeline/dev_complete_c_cost0p05/config.json:2`; `reports/full_pipeline/dev_complete_c_cost0p05/config.json:12`; `reports/full_pipeline/dev_all882_cost0p05/config.json:2`; `reports/full_pipeline/dev_all882_cost0p05/config.json:12`; `reports/full_pipeline/stage_c_complete_20261013/config.json:4`; `reports/full_pipeline/stage_c_complete_20261013/config.json:8`; `reports/full_pipeline/stage_c_complete_20261013/COMPLETE.json:1`; `reports/full_pipeline/stage_c_all882_20261013/config.json:4`; `reports/full_pipeline/stage_c_all882_20261013/COMPLETE.json:1`; `scripts/train_trajectory_policy.py:43`; `scripts/run_tracking.py:54`; `scripts/run_tracking.py:65`; `dual_rgbt/tracker.py:44`; `dual_rgbt/tracker.py:47`; `dual_rgbt/tracker.py:50`; `dual_rgbt/tracker.py:51`; `reports/all_train_frozen_test_selection.json:2`; `reports/all_train_frozen_test_selection.json:3`; `scripts/compare_dev_state_writes.py:145`; `reports/C20_C882_DEV_STATE_WRITE_COMPARISON.md:19`; `reports/C20_C882_DEV_STATE_WRITE_COMPARISON.md:20`; `reports/C20_C882_DEV_STATE_WRITE_COMPARISON.md:21`; `reports/C20_C882_DEV_STATE_WRITE_COMPARISON.md:31`; `reports/C20_C882_DEV_STATE_WRITE_COMPARISON.md:32`.

## F. Evaluation type: PASS

The localization outcomes and temporal comparisons are real_gt post-hoc observational diagnostics: target boxes come from the dataset manifest and the saved GT serialization, independently checked. They are not synthetic_proxy, self_supervised_proxy, simulation_only or human_eval.

Prediction-derived prior crop geometry and rule-inferred write status are conditioning variables, not model-derived GT. The prior C20 temporal record and the referenced legacy .05 scoring are also real-GT comparisons under their respective declared protocols. This classification does not grant official benchmark equivalence or causal validity.

**Evidence:** `scripts/prepare_data.py:17`; `scripts/run_tracking.py:81`; `scripts/run_tracking.py:113`; `scripts/analyze_full_dev_coverage.py:32`; `scripts/analyze_full_dev_coverage.py:62`; `scripts/analyze_full_dev_coverage.py:66`; `scripts/compare_dev_state_writes.py:70`; `scripts/compare_dev_state_writes.py:71`; `data/benchmarks/lasher_dev.json:1`.

## Independently verified comparison

| Quantity | C20 | C882 |
|---|---:|---:|
| frames | 54646 | 54646 |
| tracked_frames | 54549 | 54549 |
| valid_frames | 54289 | 54289 |
| invalid_frames | 260 | 260 |
| inferred_write_events_all_gt | 6445 | 6597 |
| inferred_write_events_valid_gt | 6430 | 6584 |
| inferred_modality_writes_valid_gt | 11124 | 10190 |
| low_iou_frames | 12249 | 12445 |
| low_iou_write_events | 1053 | 1034 |
| bad_write_auxiliary_selected | 869 | 625 |
| bad_write_after20_low_frames | 851 | 809 |
| low_iou_episodes | 413 | 427 |
| low_iou_episodes_with_write | 88 | 106 |
| low_iou_episodes_at_least20 | 83 | 78 |
| Low-IoU event fraction (%) | 16.376360808709176 | 15.704738760631836 |
| Mean bad-write response | 0.8183547577627024 | 0.805025220205521 |
| Mean heavy tokens per tracked frame | 549.2207739830244 | 529.6723312984656 |
| Bad-event action counts (STOP / PRIMARY / SECONDARY / PREVIEW / EXPAND) | 184 / 485 / 252 / 124 / 8 | 409 / 261 / 149 / 195 / 20 |
| Bad-event predecessor (low / intermediate / good / invalid-unobserved) | 1044 / 7 / 1 / 1 | 1020 / 11 / 2 / 1 |
| Prior low streak (0 / 1-4 / 5-19 / at least20) | 9 / 48 / 145 / 851 | 14 / 59 / 152 / 809 |

| Arm | H | Anchors | Complete | End excluded | Invalid excluded | Entirely low | Reaches0.5 | Recovery offset sum | Mean future IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C20 | 5 | 1053 | 1042 | 9 | 2 | 991 | 19 | 84 | 0.012088255817249711 |
| C20 | 20 | 1053 | 1011 | 38 | 4 | 860 | 100 | 1158 | 0.038970547538932546 |
| C20 | 50 | 1053 | 954 | 94 | 5 | 688 | 200 | 4683 | 0.07180850226811995 |
| C882 | 5 | 1034 | 1026 | 7 | 1 | 956 | 17 | 76 | 0.014575967914467386 |
| C882 | 20 | 1034 | 993 | 39 | 2 | 819 | 89 | 981 | 0.04142012036984972 |
| C882 | 50 | 1034 | 932 | 100 | 2 | 654 | 195 | 4609 | 0.07395805485334868 |

C882 minus C20 is -19 bad inferred events, -0.6716220480773405 percentage points in bad-write fraction, +196 raw low-IoU frames and +18 low-IoU episodes containing a write. All other deltas match the JSON. These are descriptive counts and associations.

## Execution evidence, failed attempts and preservation

New execution observations: producer wrapper native chunk `43dbf5` exit0; independent raw verifier `f2d3f2` exit0; supplemental R1 session `82733` closed at chunk `2c7f03` exit1; its narrow correction `e38d73` exit0; historical-receipt consistency `079cae` exit0; main input after-seal `e758ac` exit0. There is no pending native verification session. The corrected supplemental failure belongs to the reviewer assumption, not to the experiment data.

Original R1/R2/R3 and the previous C20 temporal audit executions are **receipt-only history**. This reviewer checked their retained artifacts and numeric consistency but did not witness those original processes. Newly executed verification is recorded separately above and in the companion JSON.

All 595 producer inputs match. The main before/after seal contains 692 files; including extra legacy-scoring and preserved historical evidence, 1076 audited input paths (256328659 bytes) have matching hashes. Full hashes are in the companion JSON. Final unchanged-input seal: `C:/Users/gb/.codex_tmp/dual_rgbt_c20_c882_state_write_audit_20261015/final_unchanged_input_seal.json`. The raw files remain unchanged.

| Artifact | SHA256 |
|---|---|
| Compared report | 50d6a6c936beea9990680a74e7579a56c633b7987dfa80e916a99965f52fc515 |
| Event CSV | 18885169f8f88d9c5ef3a0644743440d04d4c3b209c5a43e0a26a30c70c3abb5 |
| Producer input inventory | 509acbf23207752efa98394353a2fc1c7cfd07c46f5fde6ed4aee84dc3fde317 |
| Current producer source | 1be3c43dcde5d85b61f702c5bb35857a7cd4f29540ecc8bdb36d098029d355db |
| Independent verifier source | 9a1be524c1abba4b8fd512adc8a930f51e49590c0ae5882769297d656729937f |

## Claim impact and action items

The reported persistent bad-write counts, predecessor/streak prevalence, STOP/query split and small C20-to-C882 descriptive change are supported with the stated scope. Causal template harm, negative auxiliary gain, solved-write behavior, current full-test/E1 success, matched-compute superiority, speedup and target attainment are not established. No producer or numerical-claim edit is required for this descriptive report. Preserve the existing qualifiers; treat Markdown line32 live-operation language as unverified by this audit.

Private verification scripts, all failed/successful logs and receipts: `C:/Users/gb/.codex_tmp/dual_rgbt_c20_c882_state_write_audit_20261015/`. Full trace: `.aris/traces/experiment-audit/2026-10-15_c20_c882_state_write/response.txt`. No NN model, GPU query, SSH/network access, delegation, threshold tuning or existing-input edits were performed.
