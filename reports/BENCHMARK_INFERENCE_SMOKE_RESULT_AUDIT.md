# Benchmark inference smoke result audit

**WARN — 0 blocking issues.** The two authentic LasHeR-train inference runs are consistent with their registered source, manifests, native task records and collected outputs. This supports an execution-admission claim only. Fresh-context review is **same-family / provisional**, with requested `gpt-6-astra` / `max` routing **UNATTESTED**.

- Plan SHA-256: `fa9bd08a922978aa79a6d597a115f7654aaf84996e64baba8850eb59ce07ed25`.
- Smoke summary SHA-256: `e87e1960f1c5083946512077e4136bf3f71203b1ed743ce0d74100eadcd00b8e`.
- Launcher SHA-256: `52367bfe6ffc83a1ceefb51504d63d15a7e4eaa719ef62d5e6faa47d201823d0`.
- The independent local CPU verifier completed **182 checks, 0 failures, actual exit 0**, with **118 input files** retained and hashed. No torch import, NN execution, SSH, GPU query, old-E0 progress query or task restart occurred.

## Native execution and artifact identity

Both launches bind the exact wrapper/child argument arrays, task name, wrapper PID, physical GPU1 and config. Their native records report `COMPLETE` and the actual child return code **0**; the wrapper records `subprocess.run(...).returncode`, rather than inferring success from a launcher. All eight copy/sync/launch/observation/collection transports returned **0**, with empty stderr. Raw launch/observation stdout parses identically to the retained JSON, and each native record matches the collected `native_task.json`. The runs were sequential; E0 completed and its collection finished before E1 launched.

The plan, all **44 source files**, exact launcher, both SOURCE-review bindings, nine source-archive members and every locally available registered input match their expected hashes. All **18 run files** match the two complete summary SHA maps: config, native record, progress, SMOKE, SANITY, timing array, GT, predictions and trace per run. The old38 source seal and original R1/R2 review files remain byte-identical. The public JSON lists each input/evidence hash. Source evidence: `scripts/run_task.py:21`, `scripts/run_task.py:28`, `scripts/run_benchmark_tracking.py:153`, `scripts/run_benchmark_tracking.py:160`, `scripts/run_benchmark_tracking.py:270`.

| Fact | E0 | E1 |
|---|---:|---:|
| Native elapsed seconds, including setup | 35.602550 | 36.091916 |
| Saved frames / tracking decisions | 11 / 10 | 11 / 10 |
| Native child exit | 0 | 0 |
| Primary modality counts, RGB / thermal | 10 / 0 | 10 / 0 |
| Mean heavy tokens per tracking decision | 558.7 | 404.3 |
| Parameter count, native assertion evidence | 103,374,215 | 103,218,887 |

E0 uses `runs/stage_b_refreshed_20261013/latest.pth` (`ba49b506cd4bdcdfb355e8741feaf6c5c61e2edf9cd0a5c91459c3e5957c80ed`). E1 uses `runs/global_candidate_policy_formal_20261014/latest.pth` (`b84d00454717cae56048753e0fe54c1d6b92cfcd2820c01fa65296508488ccb3`), admitted as non-sanity candidate-B step6250. Both construct through the protected CAD checkpoint; E1 strictly loads the registered E0 factory before the candidate checkpoint. These are smoke checkpoint paths, not final-C weights. The checkpoint bytes are unavailable locally; their SHA and load facts are supported by the executed source-bound native gates, not an independent local checkpoint load. Sources: `dual_rgbt/model.py:34`, `dual_rgbt_global/candidate_policy.py:87`, `scripts/run_benchmark_tracking.py:45`, `scripts/run_benchmark_tracking.py:100`, `scripts/run_benchmark_tracking.py:192`.

## Ground truth, causality and frame integrity

**A — GT provenance: PASS (`real_gt`, train inference smoke only).** Each run covers frames0–10 of `2up`, a 176-frame sequence in the original train882 partition. The entire smoke manifest row—RGB/TIR filenames and boxes—matches the first train row of the original manifest snapshot; its SHA equals the plan's `data/manifest.json` hash. Train882/dev97/test245 names are disjoint. Both saved 11-row GT files match the registered raw-float32 GT serialization exactly. Original `init.txt` and image bytes were not downloaded, so the local provenance check ends at the exact original-manifest snapshot. Sources: `scripts/prepare_benchmark_inputs.py:26`, `scripts/prepare_benchmark_inputs.py:31`, `scripts/benchmark_inputs.py:102`.

Both prediction and GT arrays are finite11×4 with positive widths/heights, the first prediction equals frame0 GT, and subsequent predictions are not a copied GT table. Trace rows cover exactly frames1–10 with finite fields, consistent positive timing, route ranges and token accounting. Progress counts and means match the actual trace. The runner initializes only once, tracks from images and prior state/history/templates, and writes subsequent GT after completing predictions. E0 retains its original `Tracker`, with the admitted six-channel crop binding applied before its import; E1 uses `CandidateEvidenceTracker`. No later GT enters either tracker. Sources: `scripts/run_benchmark_tracking.py:21`, `scripts/run_benchmark_tracking.py:32`, `scripts/run_benchmark_tracking.py:216`, `scripts/run_benchmark_tracking.py:237`, `scripts/run_benchmark_tracking.py:247`, `dual_rgbt/tracker.py:34`, `dual_rgbt_global/candidate_tracker.py:23`.

**B — normalization: PASS. C — result existence: PASS. D — executed causal path: PASS.** No PR/NPR/SR or other benchmark accuracy metric was produced or recomputed. Internal score-sum normalization in policy entropy features does not normalize a reported benchmark result. The separate scorer is intentionally absent from this smoke path. No artificial CPU fixture is counted as a neural result.

## Actual route coverage and tensor limits

E0 action counts in `(STOP, PRIMARY, SECONDARY, PREVIEW, EXPAND)` order are **[1,4,5,0,0]**. E1 counts in `(STOP, LOCAL_MAIN, LOCAL_ALTERNATE, GLOBAL_LOCAL, GLOBAL_WIDE)` order are **[9,1,0,0,0]**. All20 decisions use RGB as primary. E1's trace reports8 global candidate positions on every frame; the only non-STOP request is local-main index0 with rank−1. No global candidate is executed. Thermal-primary, E0 PREVIEW/EXPAND, E1 LOCAL_ALTERNATE/GLOBAL_LOCAL/GLOBAL_WIDE, and short/exhausted candidate-pool paths remain uncovered. Sources: the two private `traces/2up.csv:2` files; `dual_rgbt/model.py:182`, `dual_rgbt_global/candidate_tracker.py:34`, `dual_rgbt_global/candidate_policy.py:24`.

**Tensor evidence: qualified native assertions.** The executed source freezes model parameters, applies no-grad, compares all265 state tensors before/after inference, and asserts no parameter gradients. Strict reload restores the saved pre-frame10 bbox/templates/history/window/frame/query-age state and compares the repeated box, route fields excluding seconds, templates and history. It replays **only frame10**: E0 SECONDARY and E1 STOP. Source plus trace implies template-update conditions at E0 frame10 and E1 frames5/10; those tensor changes were not independently inspected locally. No checkpoint/tensor snapshots/replay tensors are present in the downloaded evidence. The zero optimizer/backward fields are written constants supported by the executed no-grad/freeze path and absence of optimizer/backward calls, rather than instrumented call counters. Sources: `scripts/run_benchmark_tracking.py:40`, `scripts/run_benchmark_tracking.py:105`, `scripts/run_benchmark_tracking.py:128`, `scripts/run_benchmark_tracking.py:233`, `scripts/run_benchmark_tracking.py:263`, `scripts/train_global_evidence_policy.py:29`.

## Timing, memory and claim ceiling

| Recomputed 11-frame service statistic | E0 | E1 |
|---|---:|---:|
| Total seconds | 4.529721135 | 4.731850632 |
| Mean milliseconds | 411.792830 | 430.168239 |
| P95 milliseconds, linear quantile | 1924.824619 | 1968.959553 |
| Frames / service second | 2.428406 | 2.324672 |
| Peak allocated bytes, including reload clone | 961,906,176 | 962,530,304 |
| Peak reserved bytes, including reload clone | 1,056,964,608 | 1,054,867,456 |

All11 timing entries match initialization plus the10 trace service times exactly; sums, means, p95 and FPS match SMOKE and summary exactly. Service includes image decoding, first-frame initialization and synchronized tracking. It also includes smoke's last-frame state snapshot copy. Model loading, output-file writes and reload verification are outside the service timer; `wall_seconds` ends before reload verification. These are cold short-run measurements: the first tracked frame takes3.751712/3.809488 seconds. They do not establish steady-state throughput or an E0/E1 speed comparison. Sources: `scripts/run_benchmark_tracking.py:211`, `scripts/run_benchmark_tracking.py:217`, `scripts/run_benchmark_tracking.py:230`, `scripts/run_benchmark_tracking.py:262`, `scripts/run_benchmark_tracking.py:275`.

Peak memory is read **after** `reload_probe` builds and runs a second CUDA model. Although that clone is out of scope on return, the peak counter still includes its allocations; reserved memory can retain them. These values cannot be represented as the peak for one production model. Sources: `scripts/run_benchmark_tracking.py:107`, `scripts/run_benchmark_tracking.py:267`, `scripts/run_benchmark_tracking.py:280`.

**E — scope: WARN. F — classification: real_gt / train inference smoke only.** One shared train sequence, one seed2027 and11 frames per variant do not establish full882 training, dev97 selection, any full test set, three-benchmark performance, module effectiveness, accuracy targets or whole-system speed gains. Full-C identities, frozen full-dev selection and authentic full-test scoring remain separate obligations. This audit does not update their status or authorize their completion claims.

The preserved local-runtime incident states that the first driver invocation's output/exit was not retained. Missing NumPy and absence of the private directory were later recorded; imports precede directory creation/SSH in the launcher. The original exit is **unknown** and has not been invented. The later native execution/transport chain was verified directly.

Private audit source, raw CPU stdout/stderr, actual exit receipt, full input snapshots, exact request and report response are retained at `C:/Users/gb/.codex_tmp/dual_rgbt_benchmark_inference_audit_20261015_76d32f`. Public artifacts contain no raw GT coordinates, prediction arrays or command arrays. The machine report binds every audited input and the independent witness.
