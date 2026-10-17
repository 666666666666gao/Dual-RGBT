# E1 Three-Benchmark Test Experiment Audit

**Verdict: WARN — result integrity PASS WITH WARNINGS; blocking issues: 0.**  
Generated: `2026-10-17T16:13:48.794180Z` (UTC). Reviewer requested: `gpt-6-astra` / `max`; actual executor: `UNATTESTED`; authorization: `same-family/provisional`.

## Scope and method

This is a fresh CPU-only deterministic audit of saved E1 test outputs for LasHeR, RGBT210 and RGBT234. Raw scripts, manifests, run receipts, prediction/GT files, timing arrays, traces and toolkit inputs were copied into `C:\Users\gb\.codex_tmp\dual_rgbt_e1_three_benchmark_audit_20261017`. No GPU query, model forward, training, optimizer, live-job query, source/input/HANDOFF edit or commit/push was performed. Complete raw machine-readable evidence is in `recompute_summary.json`, `identity_checks.json`, `gt_toolkit_checks.json`, `scope_checks.json`, and `input_hashes.json`.

## A–F checklist

| Check | Status | Audit result | Claim impact |
|---|---|---|---|
| A. Ground truth | **PASS** | LasHeR 245/220703, RGBT210 210/104706, RGBT234 234/116650; raw frame recounts, saved GT arrays and toolkit annotations match; static route/source checks show test GT is materialized after prediction and absent from candidate tracker/policy. LasHeR toolkit order differs but set and all arrays match. | Real-GT evaluation supported; frame-0 initialization is GT by declared protocol. |
| B. Score arithmetic | **PASS** | Independent CPU recomputation exactly matches stored curves/aggregates; finite shapes and per-sequence comparisons pass. | Reported metrics are arithmetically reproducible from saved outputs. |
| C. Identity/existence | **WARN** | Six shards are native COMPLETE/exit 0, non-smoke, zero optimizer/backward calls, exact task/PID/GPU/config/checkpoint/frozen-selection bindings, partition/frame/trace/timing checks pass. Local formal checkpoint tensor and native training receipt are absent; requested freeze script is missing. | Test receipt identity is supported by local evidence; checkpoint provenance remains declared/remote-only. |
| D. Scope | **WARN** | E1 test is separated from dev selection, E0 LasHeR reference and user baseline. One seed (2027); timing is concurrent-shard service time, no paired baseline. | No robustness, speedup, novelty or cross-protocol superiority claim. |
| E. Route diagnostics | **PASS** | Independently counted action/modality/candidate traces for all benchmarks. | Diagnostics only; no causal ablation or module attribution. |
| F. Evaluation type | **WARN** | All three are `real_gt`; supplied CADTrack legacy toolkit and NumPy-compatibility/rounding semantics are recorded. | Metrics should carry toolkit/provenance qualification. |

## Independently recomputed test metrics

| Benchmark | Sequences / frames | Metrics (calculated = stored) | Mean ms | P95 ms | FPS |
|---|---:|---|---:|---:|---:|
| LasHeR | 245 / 220703 | PR=67.612356655655873, NPR=62.545796990316227, SR=50.842354251863554 | 138.345142431968 | 211.687612172682 | 7.228298604642 |
| RGBT210 | 210 / 104706 | PR=84.429456587253910, SR=54.041235042676270 | 100.591786104382 | 141.950828314293 | 9.941169540049 |
| RGBT234 | 234 / 116650 | MPR=87.774863758118499, MSR=57.523560082146993 | 91.506525177640 | 133.601841976633 | 10.928182422606 |

Stored-vs-calculated metric differences are exactly zero for every aggregate. Curve shapes and exact equality: LasHeR PR/NPR `(245,51)`, SR `(245,21)`; RGBT210 PR `(210,51)`, SR `(210,21)`; RGBT234 MPR `(234,51)`, MSR `(234,21)`. Every curve is finite; per-sequence mismatch count is zero.

## Coverage, identity and route evidence

Each benchmark is covered by two shards with exact sequence and frame totals: LasHeR 123+122 sequences and 107,898+112,805 frames; RGBT210 105+105 and 48,925+55,781; RGBT234 117+117 and 54,292+62,358. All six native receipts report COMPLETE and exit code 0, `smoke_frames=0`, `optimizer_updates=0`, `backward_calls=0`; no missing, extra or duplicate sequence/trace files were found.

- **LasHeR:** trace rows `220458`, trace files `245`, action counts `[205632, 7023, 6443, 371, 989]`, modality counts `[105483, 114975]`, candidate-count histogram `{"1": 534, "2": 1186, "3": 2171, "4": 2986, "5": 3471, "6": 4117, "7": 2686, "8": 203307}`.
- **RGBT210:** trace rows `104496`, trace files `210`, action counts `[100009, 3501, 954, 6, 26]`, modality counts `[46575, 57921]`, candidate-count histogram `{"2": 13, "3": 47, "4": 297, "5": 295, "6": 332, "7": 571, "8": 102941}`.
- **RGBT234:** trace rows `116416`, trace files `234`, action counts `[111634, 3892, 837, 12, 41]`, modality counts `[56873, 59543]`, candidate-count histogram `{"2": 14, "3": 41, "4": 73, "5": 169, "6": 270, "7": 523, "8": 115326}`.

These are route diagnostics only. This audit did not run or infer causal module ablations.

## Limitations and claim boundaries

- The formal checkpoint path is declared as `runs/global_candidate_trajectory_formal_20261015/latest.pth` with SHA256 `d4454775f085e1af65c61f53d4d21f284706289b6ff3aeb1167f73d79d31a362`, but the tensor and native training receipt are not present locally; they cannot be rehashed here.
- `scripts/freeze_benchmark_selection.py` was explicitly requested but is missing. The frozen selection JSON, dev receipts, selection SHA, and run config bindings are present and hash-consistent.
- The LasHeR legacy scorer uses invalid GT/error rows as successful at thresholds (1,260 invalid rows across 52 sequences in the raw manifest). The supplied toolkit is not independently authenticated as an official upstream release.
- Host NumPy is 2.3.5. The recomputation uses explicit float64 scalar promotion where needed to reproduce the registered legacy NumPy-1.x arithmetic; toolkit portability should be treated as a provenance limitation.
- One seed is audited. The evidence supports final E1 test metrics and deterministic arithmetic/coverage only; it does not establish robustness, speedup, novelty or causal component effects.

## Exact hashes and preserved evidence

- Input seal: `C:\Users\gb\.codex_tmp\dual_rgbt_e1_three_benchmark_audit_20261017\input_hashes.json`; SHA256 `7f08bbdfb543d045d3be5ffca15db1ecdb094eeee9db5c33d1d43d456dfe823e`; `3255` sealed file records. Missing requested path: `scripts/freeze_benchmark_selection.py`.
- The JSON report contains exact SHA256 records for every explicitly listed source/report/manifest and linked metrics file. The full per-run artifact and toolkit/annotation seal is preserved in `input_hashes.json`.
- Full raw evidence and this response are preserved under `C:\Users\gb\.codex_tmp\dual_rgbt_e1_three_benchmark_audit_20261017`.

## Report files

- `C:\Users\gb\projects\Dual-RGBT\reports\e1_three_benchmark_test_20261017\EXPERIMENT_AUDIT.json`
- `C:\Users\gb\projects\Dual-RGBT\reports\e1_three_benchmark_test_20261017\EXPERIMENT_AUDIT.md`

