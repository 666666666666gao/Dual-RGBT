# C20 state-write temporal result audit

**Date:** 2026-10-15  
**Verdict: WARN — 0 blocking issues.** Deterministic archived arithmetic: **PASS**.  
**Review attribution:** fresh native Codex reviewer `/root/audit_c20_state_write_temporal_20261015`; requested `gpt-6-astra` / `max`, actual routing **UNATTESTED**, same-family / provisional. This is a result audit, not a source admission or training-completion certificate.

All 97 saved GT files match the original manifest serialization byte for byte. Independent raw recomputation matches all reported metric sections and all 1053 private event records. A fresh native CPU rerun exits 0 and reproduces the JSON report, event CSV and stdout byte for byte. No blocking arithmetic or implementation defect was found. WARN retains the observational scope and reviewer-attribution limits.

## A–F findings

### A ground truth provenance: PASS

- All 97 sequence names, frame counts, image-path order and GT arrays were checked against the original private manifest, not reconstructed from predictions.
- Original manifest SHA256 is 389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc; it identifies init.txt annotations and a disjoint 882/97/245 split. The comparison dev manifest names this same hash and matches the dev data.
- For all 54646 GT rows, manifest boxes cast to float32 and serialized with %.6f comma-separated LF reproduce every saved GT byte. Reloaded saved GT has maximum float32 difference 0.0 from the manifest conversion.
- All 97 prediction initialization rows equal GT; none of the 54549 subsequent prediction rows is identical to its GT row. Code passes GT only to initialization and advances state from model predictions.
- Original remote init.txt files and the historical checkpoint were not opened or reloaded in this CPU-only archive audit; lineage is bounded by the supplied immutable snapshot and saved artifacts.

**Evidence:** `scripts/prepare_data.py:11-25`; `scripts/run_tracking.py:25-28`; `scripts/run_tracking.py:81-86`; `scripts/run_tracking.py:98-113`; `dual_rgbt/tracker.py:34-43`; `data/benchmarks/lasher_dev.json:1`.

### B raw geometry and denominators: PASS

- Prediction/GT coordinates and intersection/union geometry use float32; the IoU array and subsequent reductions use float64. Scalar independent continuous-xywh arithmetic matches all reported values under the declared reduction order.
- IoU denominator is geometric union area, not prediction-score max/min/mean. Event and window denominators are explicit raw counts. All GT/predictions are finite; prediction sizes are positive; valid GT requires positive width and height; every valid scalar union checked positive.
- The 97 initialization frames are excluded, leaving 54549 tracked frames, 54289 valid and 260 invalid. Invalid GT splits low-IoU episodes and streaks. Future windows exclude insufficient sequence remainder before checking any invalid future GT.
- For each horizon, complete-window means are weighted by number of complete windows across sequences. Recomputing this specified order is exact. A separate flat reduction differs by at most 1.1102230246251565e-16 from floating-point addition order; no tolerance, data correction or threshold adjustment was used.
- Six-decimal saved predictions only approximate original search-crop geometry. This affects the reconstructed target-centre-inside diagnostic, not conversion into official PR/NPR/SR; no official metric is claimed.

**Evidence:** `scripts/analyze_full_dev_coverage.py:30-49`; `scripts/analyze_full_dev_coverage.py:51-75`; `scripts/analyze_state_write_trajectories.py:13-36`; `scripts/analyze_state_write_trajectories.py:135-148`; `dual_rgbt/data.py:22-38`; `docs/C20_STATE_WRITE_TEMPORAL.md:3-5`.

### C result files and recomputed numbers: PASS

- All 291 raw trajectory files exist, with exactly 97 prediction, 97 groundtruth and 97 trace files. All trace frame numbers are exactly 1..N-1, and config/COMPLETE and per-file reference hashes agree.
- Without importing either production analysis function, independently recomputed all 11 reported metric sections, every one of 97 per-sequence rows, all 4 confidence bins, predecessor/streak counts, low-IoU episodes, and all 12 group/horizon rows including exclusions, recovery counts, recovery-frame sums and means; exact match.
- All 1053 independent event records, including frame/action/modality/confidence/raw IoU/prior streak/target-centre-inside, serialize byte-exactly to the archived private events.csv.
- All 18 numeric Markdown table rows match independently recomputed integers and specified display rounding.
- Archived original transport records exit_code=0, elapsed 1.855964183807373 seconds and the current producer source hash. Archived stdout matches the report projection and stderr is empty. This is historical recorded evidence, not a new observation of that original process.
- A fresh actual native CPU subprocess reran the unchanged producer with NumPy 1.26.4 and new private outputs: exit 0, empty stderr, byte-identical report, events and stdout. No torch import or neural execution was used.

**Evidence:** `reports/c20_dev_state_write_temporal_20261015.json:16-198`; `reports/c20_dev_state_write_temporal_20261015.json:199-1558`; `docs/C20_STATE_WRITE_TEMPORAL.md:9-45`; `C:/Users/gb/.codex_tmp/dual_rgbt_state_write_20261015/transport.json:1`; `C:/Users/gb/.codex_tmp/dual_rgbt_c20_state_write_audit_20261015/independent_evidence.json:1`.

### D analysis execution: PASS

- The __main__ entry calls main; main calls analyse_sequence for all names and future_summary for each group/horizon. Their outputs flow to report and CSV, reproduced by the actual native subprocess.
- The hard-coded legacy totals 97/54646/6430/1053 are archive assertions checked against computations. Independent raw recomputation establishes the same totals; they are not substituted phantom observations.
- analyze_full_dev_coverage.main is a separate legacy CLI and is not claimed as a second new run. Its analyse_sequence/summarize path is called by the temporal producer.
- Analysis/helper/tracker source hashes agree with the report; all 46 reference-completion sealed source files match before and after the audit.

**Evidence:** `scripts/analyze_state_write_trajectories.py:72-74`; `scripts/analyze_state_write_trajectories.py:117-119`; `scripts/analyze_state_write_trajectories.py:151-183`; `scripts/analyze_full_dev_coverage.py:29-89`.

### E scope and semantic claims: WARN

- Scope is one archived C20-identified checkpoint hash 94be6239f7f6864cc193fff7c21d67c9479924c9432039c6ff16cd0d1223f15f, cost0.05, full97 dev and one recorded tracking run. No new E1, full882 C, test, cross-dataset, target-attainment or multi-seed claim is supported or made.
- 6430 denotes valid-GT inferred write events, including 1053 with simultaneous raw IoU<0.1. Across all GT statuses there are 6445 inferred events; 15 occur at invalid GT. The corresponding inferred modality writes are 11124 valid-GT and 11153 all-status because STOP writes one modality and non-STOP writes two. Events are not modality-write counts.
- Windows start at t+1 and end at t+H within the same sequence; no off-by-one or cross-sequence inclusion was found. The anchor is excluded. Initialization and invalid GT cannot silently seed observed low-IoU streaks.
- 99.15% of low-IoU writes follow an already-low frame and 80.82% follow at least20 consecutive low frames. This supports temporal prevalence within existing drift, not a cause of drift, precise drift onset, or a prevention benefit.
- High-confidence writes and low-confidence eligible nonwrites are not randomized/matched; current IoU severity, visibility, target identity, motion and prior state can differ. Good/intermediate/current-low groups also condition on present IoU. Overlapping windows and sequence dependence prevent independent-trial/error-probability interpretation.
- Reaches IoU0.5 means at least one future frame meets the threshold, not stable recovery. Existing diagnostics lack main-only counterfactuals, rejected-query outputs and template rollback. The future-control suggestions are hypotheses, not established efficacy.
- The document explicitly retains these limits and does not deploy the 0.95 confidence bin as a tuned threshold. No mandatory claim correction is required within the stated descriptive scope.

**Evidence:** `dual_rgbt/tracker.py:44-54`; `scripts/analyze_state_write_trajectories.py:76-84`; `scripts/analyze_state_write_trajectories.py:168-177`; `docs/C20_STATE_WRITE_TEMPORAL.md:3-5`; `docs/C20_STATE_WRITE_TEMPORAL.md:32-53`.

### F evaluation type: PASS

- Temporal outcome IoU uses dataset-provided boxes verified against the original manifest; it is real_gt post-hoc observational evidence, not synthetic_proxy, self-supervised proxy or simulation.
- Search-crop geometry is prediction-conditioned and update status is inferred from the frozen rule. These do not turn the GT into a prediction-derived reference and do not establish a causal intervention.
- The earlier clipped-target report contains expressly hypothetical clipping diagnostics of real dev GT. This audit checks its used metadata/hashes and write/low-IoU event references; it is not a fresh full audit of every unrelated clipping stratum.

**Evidence:** `scripts/run_tracking.py:81-113`; `scripts/analyze_full_dev_coverage.py:30-75`; `docs/C20_STATE_WRITE_TEMPORAL.md:3-5`; `docs/CLIPPED_TARGET_DIAGNOSTICS.md:19-23`.

## Independent counts

| Quantity | Recomputed |
|---|---:|
| Sequences / total frames | 97 / 54646 |
| Initialization / tracked / valid / invalid frames | 97 / 54549 / 54289 / 260 |
| Inferred valid-GT write events / low-IoU events | 6430 / 1053 |
| All-status write events / invalid-GT events | 6445 / 15 |
| Inferred modality writes, valid-GT / all-status | 11124 / 11153 |
| Predecessor: low / good / intermediate / invalid-unobserved | 1044 / 1 / 7 / 1 |
| Prior low streak: zero / 1–4 / 5–19 / at least20 | 9 / 48 / 145 / 851 |
| Low episodes / low frames / episodes with writes / length at least20 | 413 / 12249 / 88 / 83 |
| Episodes ending at sequence end / before invalid GT | 21 / 7 |

| Confidence bin | Events | Low IoU | IoU sum |
|---|---:|---:|---:|
| [0.65, 0.75) | 1351 | 237 | 798.678809673991 |
| [0.75, 0.85) | 2214 | 398 | 1318.1293923994526 |
| [0.85, 0.95) | 2745 | 417 | 1833.656979508698 |
| [0.95, 1.00] | 120 | 1 | 100.96843236684799 |

All 12 horizon/group rows below were independently enumerated from the raw frame files. Counts and the specified sequence-weighted floating-point means match exactly. `Recovery sum` adds first qualifying future-frame offsets only among windows that reach IoU≥0.5.

| Group | H | Anchors | Complete | End excluded | Invalid excluded | Entirely low | Reaches0.5 | Recovery sum | Mean future IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low_iou_update | 5 | 1053 | 1042 | 9 | 2 | 991 | 19 | 84 | 0.012088255817249711 |
| low_iou_update | 20 | 1053 | 1011 | 38 | 4 | 860 | 100 | 1158 | 0.038970547538932546 |
| low_iou_update | 50 | 1053 | 954 | 94 | 5 | 688 | 200 | 4683 | 0.07180850226811995 |
| low_iou_no_update_eligible | 5 | 1401 | 1387 | 11 | 3 | 1193 | 102 | 305 | 0.04369609085792607 |
| low_iou_no_update_eligible | 20 | 1401 | 1339 | 50 | 12 | 952 | 284 | 2461 | 0.09247302996771292 |
| low_iou_no_update_eligible | 50 | 1401 | 1235 | 141 | 25 | 759 | 382 | 6209 | 0.12631820981319775 |
| good_iou_update | 5 | 4909 | 4850 | 54 | 5 | 0 | 4839 | 4892 | 0.787165392366574 |
| good_iou_update | 20 | 4909 | 4675 | 210 | 24 | 0 | 4669 | 4755 | 0.7722161478773183 |
| good_iou_update | 50 | 4909 | 4334 | 515 | 60 | 0 | 4332 | 4560 | 0.7504619963487871 |
| intermediate_iou_update | 5 | 468 | 466 | 1 | 1 | 4 | 83 | 205 | 0.3051376123846681 |
| intermediate_iou_update | 20 | 468 | 457 | 8 | 3 | 3 | 163 | 1203 | 0.308615933755972 |
| intermediate_iou_update | 50 | 468 | 428 | 30 | 10 | 1 | 217 | 3380 | 0.3038066255886006 |

## Preservation and runtime evidence

All **356 inputs (122435664 bytes)** were copied to private R1 preservation storage and checked against SHA256 before and after the audit and against preserved bytes. Every input remains unchanged. All 46 source hashes in `reports/reference_completion_plan.json` match; the plan and launch references also remain unchanged.

The companion JSON records every input hash, per-check evidence, all 12 window summaries, and hashes of the private independent script, recomputed results, event CSV, GT serialization receipts, native reproduction transport/stdout/stderr and before/after manifests. Private evidence directory: `C:/Users/gb/.codex_tmp/dual_rgbt_c20_state_write_audit_20261015/`. Raw GT, predictions, events and the original manifest remain private.

Original CPU run: archived transport records exit 0 and 1.855964183807373 seconds; stdout agrees with the report and stderr is empty. Independent producer rerun: actual native subprocess exit 0 with byte-identical artifacts. The independent recomputation imports no production analyzer and no torch. Its script uses the existing NumPy1.26.4 CPU runtime. A different final summation order produces at most 1.1102230246251565e-16 difference; the declared order matches exactly and no tolerance or correction was introduced.

## Claim limits and action items

- Fresh native reviewer, requested gpt-6-astra/max; actual model/effort routing is UNATTESTED. Same-family semantic judgment is provisional, not cross-family acceptance.
- Deterministic acceptance is limited to archived input identity, exact GT serialization, independently recomputed arithmetic and native CPU reproduction.
- Remote original init.txt bytes, historical checkpoint bytes/reload and historical model execution were not reacquired. Historical CPU exit 0 is recorded transport evidence; the audit directly witnessed its own reproduction exit 0.
- One model/run at cost0.05 on dev97; no official PR/NPR/SR, current training state, final test, full882 C, new E1, or target-attainment judgment.
- Inferred update events are not a direct runtime write log or per-modality write count; reconstructed search geometry uses six-decimal archived predictions.
- Current-state confounding, sequence dependence, overlapping future windows and horizon-specific censoring prohibit causal or independent-error-rate claims.
- No saved main-only counterfactual, accepted/rejected alternate query output or rollback trajectory; no proof that an update caused, prevented or worsened drift.
- No producer/source/result/claim edits, threshold tuning, model run, SSH, GPU, torch, process/CIM check, live status or current-log intake occurred.

No producer or documentation fix is required for the current descriptive claim. Retain the existing qualifiers in downstream writing. Any proposal to change update thresholds, disable writes or claim causal benefit remains a future experiment and is not authorized or executed by this audit.
