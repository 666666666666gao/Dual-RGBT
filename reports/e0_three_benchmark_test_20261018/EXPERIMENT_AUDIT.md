# Experiment Audit Report: Reference-E0 Three Benchmarks

**Date**: 2026-10-18
**Auditor**: gpt-6-astra max, fresh same-family reviewer (provisional)
**Project**: Dual-RGBT
**Input seal**: `6180c09858ae1be42ccc7d969af52e2db800930522e94e4cadd71a460eb86374` for deterministic recomputation record

## Overall Verdict: WARN

## Integrity Status: RESULT_INTEGRITY_PASS_WITH_WARNINGS

### A. Ground Truth Provenance: PASS

The runner builds ground truth from the dataset manifest. The benchmark input preparation validates LasHeR, RGBT210, and RGBT234 boxes, finite values, shapes, and initial boxes. The scorer checks saved ground truth against the manifest and the RGBT toolkit data. No model-generated ground truth was found.

Evidence: `scripts/run_benchmark_tracking.py:247-254`; `tools/score_benchmark_tracking.py:94-107`; `scripts/benchmark_inputs.py:102-123`.

### B. Score Normalization: PASS

The reported values are the unchanged toolkit metrics multiplied by 100: PR/MPR at column 20 and SR/MSR as the curve mean; LasHeR NPR uses the toolkit's official GT geometry normalization. No denominator derived from model predictions was found.

Evidence: `tools/score_benchmark_tracking.py:113-136`; `scripts/score_tracking.py:58-78`; `toolkit/rgbt/metrics/metrics.py:28-44,62-80,177-197,212-233,290-320,332-361`.

### C. Result Existence and Arithmetic: PASS with provenance warning

All three JSON/NPZ result pairs parse and contain finite arrays. Shapes are LasHeR PR/NPR `(245,51)` and SR `(245,21)`, RGBT210 PR `(210,51)` and SR `(210,21)`, and RGBT234 MPR/MSR `(234,51)` and `(234,21)`. Independent CPU recomputation matches each aggregate and every per-sequence value. All six test shards are native COMPLETE/exit 0 and cover 245/220703, 210/104706, and 234/116650.

The warning is provenance only: the formal checkpoint file and complete native training receipt are not present locally for a fresh byte-level rehash. The remote execution records and checkpoint SHA are preserved in the result/config receipts.

Evidence: `reports/reference_complete_adopted_20261016/e0_*_full_metrics.json/.npz`; `reports/reference_completion_adopted_status_20261016.json:1-4,249-335`; `reports/e0_benchmark_frozen_selection.json:2-36`; deterministic record `C:/Users/gb/.codex_tmp/dual_rgbt_e0_deterministic_check_20261018.json`.

### D. Dead Code Detection: PASS

The RGBT210/234 metric paths are called by the benchmark scorer; the LasHeR path delegates to the existing LasHeR scorer. No unused metric path affecting these results was found.

Evidence: `tools/score_benchmark_tracking.py:68-72,113-123`; `scripts/score_tracking.py:126-130`.

### E. Scope Assessment: PASS with claim limits

The evaluation covers exactly one frozen E0 checkpoint, one seed, one frozen cost configuration selected on LasHeR dev97/54646, and three complete test benchmarks. It supports reporting these three benchmark results. It does not support multi-seed robustness, causal module attribution, or a SOTA claim.

Evidence: `reports/e0_benchmark_frozen_selection.json:2-36`; `reports/benchmark_evaluation_plan.json:3,77-116,118-126,128-153`.

### F. Evaluation Type: real_gt

All three evaluations use dataset-provided ground truth through the manifest/toolkit chain.

## Metrics

| Dataset | Coverage | Metrics |
|---|---:|---|
| LasHeR | 245 / 220703 | PR 72.524939 / NPR 69.068894 / SR 57.732435 |
| RGBT210 | 210 / 104706 | PR 88.061759 / SR 65.741170 |
| RGBT234 | 234 / 116650 | MPR 90.806676 / MSR 68.910136 |

## Action Items

- Keep the checkpoint-origin warning attached to the result record.
- Do not describe this single-seed reference evaluation as a multi-seed or causal result.
- Preserve the user-supplied LasHeR baseline as an external reference; it was not re-run.

## Claim Impact

- Complete three-benchmark metric record: **supported with integrity warning**.
- Direct comparison to the user-supplied LasHeR baseline: **supported as a descriptive comparison**.
- Speedup or lower-cost claim: **unsupported by this audit alone**; timing is unpaired.
- Three-module causal contribution: **unsupported by this result alone**.

## Reviewer Metadata

- reviewer model: `gpt-6-astra`
- reasoning effort: `max`
- review independence: `same-family`
- acceptance status: `provisional`
- reviewer task: `/root/audit_e0_three_benchmark_20261018`
