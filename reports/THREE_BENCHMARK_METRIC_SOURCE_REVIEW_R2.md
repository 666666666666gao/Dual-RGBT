# Three-benchmark metric/source review — R2

**SOURCE_FAIL — 2 remaining blocking findings.** The four R1 scorer failures are fixed. All three full registered test paths and all three full-dev cost paths run successfully on private artificial CPU fixtures. Frozen dev provenance and exact shard assignment still need correction.

- Fresh-context original reviewer, follow-up R2; `same-family`, `provisional`, requested `gpt-6-astra` / `max`; actual routing `UNATTESTED`.
- Plan SHA-256: `85757fd0ddb23855ed9861eb81cf90a4db863e7cea32f255c124804d43d7b498`.
- Scorer SHA-256: `7dcf4957b6ce572c4d5cd889ad9c8021c6801b858679988f10a264d7af00ab12`.
- Inputs/helper SHA-256: `f430bfe18903ffee3109ac445a89e05d6e345d3d00ef5002f359b0b598cadf72`.
- Freezer SHA-256: `4e11502ed7291e6c8bf32ad56cf4afcd7b9f787033cf36bb42bf0e82b78f6067`.
- All 44 exact source files and the plan were copied and verified before execution. Current live revisions are excluded from this verdict. R1 reports remain byte-exact.

## Remaining blockers

### R2-M1 — frozen dev provenance remains incomplete

The shared guard (`scripts/benchmark_inputs.py:43`) accepts any nonempty artifact map whose listed files hash correctly. Starting with valid full-dev reports, deleting one prediction hash from each map and updating the receipt hashes still passes. Complete config/COMPLETE/native/GT/prediction/timing evidence is not required by the validator.

The separate NN-source reviewer also executed the exact R2 freezer with three reports that merely declare dev97/54,646 and bind one unrelated marker file each. The freezer wrote a selection and validation accepted it while no run directories existed. This review directly read the test code, result, stdout, empty stderr and exit-0 receipt, and matched all 44 source hashes and the plan.

Required: validate exact complete run artifacts and identities, recompute the three full-dev score reports from those artifacts, compare the report contents, then write the frozen selection. Hashing selected arbitrary files alone does not establish full-dev origin.

### R2-M2 — shard number is not tied to sequence assignment

Swapping the two configured shard ids and their matching canonical argv, while keeping each original sequence list and raw files, is accepted. Aggregate coverage remains complete, so this counterexample does not change the numerical aggregate; it invalidates the claimed executed shard identity.

Source: `tools/score_benchmark_tracking.py:53`, `:57`, `:82`. Require the sequence list to equal the registered manifest slice `[shard::shards]`, matching the runner.

## Verified fixes and complete positive paths

The original unregistered/shortened manifest, unrelated command, inconsistent parameter count and unknown variant cases now all exit nonzero. Nine command mutations (variant, checkpoint hash, plan, run, task, shard, shards, cost, candidates) also reject, as do invalid GPU7, wrong formal-checkpoint path, missing shard, out-of-grid cost and wrong candidate budget. Test selection command SHA and changed selection bytes reject.

| Artificial CPU protocol path | Sequences | Frames | Result |
|---|---:|---:|---|
| LasHeR dev, cost 0.0 | 97 | 54,646 | PASS |
| LasHeR dev, cost 0.05 | 97 | 54,646 | PASS |
| LasHeR dev, cost 0.15 | 97 | 54,646 | PASS |
| LasHeR test | 245 | 220,703 | PASS |
| RGBT210 test | 210 | 104,706 | PASS |
| RGBT234 test | 234 | 116,650 | PASS |

These six paths used unchanged registered manifests and GT, with manually defined artificial predictions, timing and native metadata. Every output artifact map matches the private fixture files. The real R2 freezer CLI also accepted the three valid full-dev outputs and produced identical selected configurations for all three benchmarks.

Scorer suite: 27 cases = 6 valid accepts, 20 correct rejects, 1 incorrect accept. Frozen-selection helper: 5 correct rejects and 1 incomplete-hash-map accept. Missing cost receipt, inconsistent three-benchmark settings, wrong dev frame count, wrong checkpoint and empty artifact map reject.

## Metrics and GT retained

Rechecked 933 original toolkit code/GT/list file hashes with no differences. All five registered manifest hashes match. `groundtruth`, `load_manifest` and `sha` retain identical ASTs; the old LasHeR scorer is byte-identical. Therefore the seven independent R1 metric goldens were reused without unnecessary reruns.

The R1 findings remain applicable: RGBT234 uses per-frame min-CLE/max-IoU; RGBT210 uses original PR/SR and first-frame semantics; LasHeR preserves PR/NPR/SR and special-GT behavior. No prediction-derived denominator or synthetic ground truth enters the planned benchmark path.

**All executed scores in this review are CPU protocol fixtures, not benchmark results.** No torch/model import, neural inference, GPU query, SSH or original E0 progress query occurred. This review does not establish actual neural completion, checkpoint contents, model capacity, latency, target attainment or a new baseline.

## Evidence

- Private exact R2 source, fixtures and results: `C:\Users\gb\.codex_tmp\dual_rgbt_metric_source_review_r2_20261015_d834a9f2`.
- Primary script/result: `test_r2.py`, `protocol_results.json`.
- Reused metric/GT proof: `reused_evidence_verification.json`; R1 originals remain in their previous private directory.
- Independently read marker-only proof: `C:\Users\gb\.codex_tmp\dual_rgbt_benchmark_source_review_r2_20261015_37ac92e1`.
- Machine report contains exact source/plan/input hashes, 27 scorer exit records, 6 selection-guard outcomes, primary evidence hashes and peer attribution.
