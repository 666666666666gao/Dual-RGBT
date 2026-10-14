# Three-benchmark metric/source review — R1

**Verdict: SOURCE_FAIL — 4 blocking scorer admission findings.**

The original metric arithmetic and dataset GT mapping pass the checks below. The unchanged scorer accepts four invalid controlled inputs, so this review does not authorize benchmark aggregation. No neural inference or training was executed.

- Review: fresh native Codex sub-agent; `same-family`, `provisional`.
- Requested reviewer route: `gpt-6-astra`, `max`; actual routing: `UNATTESTED`.
- Scope: CPU metric/source protocol only. No torch/model import, GPU, SSH, original E0 progress query, benchmark run, or public metric CSV.
- Plan SHA-256: `f4313e78413811251f09b6d824313c838d2cebd3defb5615aaf3e1480385773c`.
- Scorer SHA-256: `a01b381fb8ee217231da90d552d16cbf172b46702e9a29cfd808712bd76fa448`.
- Original LasHeR scorer SHA-256: `22f0cb04de31714f70019f25c8b6b3d48b0384a7aff812d25be367394b3b4ac4`.
- All 43 R1 project-source hashes and 7 toolkit-code hashes match the reviewed snapshots. Full exact hashes are in the JSON report; original source copies are retained privately. Root edits began after the CPU tests, so changed live files are excluded from this R1 verdict.

## Blocking findings

### M1: Unregistered and shortened manifest admitted by scorer

With the unchanged authoritative plan, the scorer accepted 97 artificial sequences / 291 frames under LasHeR dev and wrote scores, exit 0. The registered dev manifest has 54,646 frames and a different hash.

Cause: Supplied manifest is compared only with run config declarations. The scorer never binds it to plan.datasets, plan.input_sha256, or plan.split_sizes.

Required correction: Bind manifest bytes, benchmark/split, counts, and full sequence coverage to the selected registered plan dataset before scoring.

Source: `tools/score_benchmark_tracking.py:23`, `tools/score_benchmark_tracking.py:39`, `tools/score_benchmark_tracking.py:45`, `reports/benchmark_evaluation_plan.json:61`, `reports/benchmark_evaluation_plan.json:97`.

Private CPU case: `unregistered_partial_manifest`; output SHA-256 `2b802a47494df2cbcb3b2514f5823fd649d4991c54340c86ccb87ed759dff2a1`.

### M2: Borrowed native command admitted

Both config.native_command and native.command were set to ['python','OTHER_PROJECT_NOT_EXECUTED.py']; unchanged scorer accepted and wrote scores, exit 0. No such command was executed; the CPU test invoked only the scorer.

Cause: Native task/GPU/command are compared with mutable copied config fields; command executable/script and CLI bindings to run, task, variant, checkpoint, plan, benchmark, split, cost, candidate budget and shard are not reconstructed.

Required correction: Validate the exact registered evaluation command and its arguments against each run/config, and the matching terminal native record. Do not accept a mutually consistent pair of unrelated command copies.

Source: `tools/score_benchmark_tracking.py:70`, `tools/score_benchmark_tracking.py:72`.

Private CPU case: `borrowed_wrongtask_command`; output SHA-256 `2b802a47494df2cbcb3b2514f5823fd649d4991c54340c86ccb87ed759dff2a1`.

### M3: Inconsistent parameter counts silently reported from first shard

Two shards declared total_parameters 123 and 987. The scorer accepted and reported only 123, exit 0.

Cause: total_parameters is absent from shard invariants while the first value is exported as a model property.

Required correction: Require parameter-count agreement across all shards before reporting the shared model cost.

Source: `tools/score_benchmark_tracking.py:35`, `tools/score_benchmark_tracking.py:125`.

Private CPU case: `inconsistent_parameter_counts`; output SHA-256 `2b802a47494df2cbcb3b2514f5823fd649d4991c54340c86ccb87ed759dff2a1`.

### M4: Unregistered model variant admitted

All shards used variant='not_registered'. The dev scorer accepted it and exported that variant, exit 0.

Cause: The dev path requires only cross-shard equality. It never validates variant membership in registered models or binds model configuration to the recorded command.

Required correction: Require a registered model variant and its variant-specific cost/candidate/checkpoint command binding before aggregation.

Source: `tools/score_benchmark_tracking.py:35`, `tools/score_benchmark_tracking.py:40`, `tools/score_benchmark_tracking.py:48`, `tools/score_benchmark_tracking.py:123`.

Private CPU case: `unknown_variant_all_shards`; output SHA-256 `2e95a48578d7d780eae2197336dd22034664ae6ebd8067f4c1f52996eb74a878`.

## Dataset GT and coverage

| Input | Sequences | Frames | Registered manifest SHA-256 |
|---|---:|---:|---|
| lasher_train_smoke | 1 | 176 | `21a8af46d97868fcf2882f06c808453b50a36e81a056b3aa4c9f5220e316f644` |
| lasher_dev | 97 | 54646 | `86f123d3386909dbc2749d95f0de6d76da2f241d84744a08208a35a0191b1911` |
| lasher_test | 245 | 220703 | `9cac4f49890bbb0c179b9d1d0717178fe3a8ad96264878774436a7b87e6d9beb` |
| rgbt210_test | 210 | 104706 | `e221a059ce389ab58fe957d787281cc55ab027cda96cbdd83e36117f9cd3bc77` |
| rgbt234_test | 234 | 116650 | `fd0c7163e5a83f1bdccfb2f6eeb12e2f21f2ef81e6fd61171c48b453f313e8b2` |

All manifest name sets are unique; RGB/TIR/GT lengths agree; every initial box equals frame 0 raw float32 visible/common GT and has positive width/height. Full paths within each sequence contain no duplicates.

Independently loaded 210 RGBT210 files (418,824 coordinates) and 468 RGBT234 modality files (933,200 coordinates). Every float32 value matches both the exported annotation JSON and the full private manifest; all 680 source annotation/list hashes match. The current provided values happen to be integral; fractional synthetic goldens independently verify that the code preserves unrounded dual GT.

All 245 LasHeR test sequences / 220,703 frames match toolkit GT both raw and rounded. Its 1,260 frames with any GT coordinate <=0 remain present. The old toolkit makes those frames pass PR/NPR and fail SR while retaining them in the denominator; this existing behavior is preserved, not silently corrected.

Nine saved transport receipts have exit code 0, every saved stderr is empty, and the remote preparation stdout equals both its private parsed record and the published preparation report. The archive binds the exact two preparation sources and exported annotations. These are saved CPU preparation receipts; this review did not access remote hosts or decode image pixels.

RGBT210 now has the recorded independent image root `/data/wangwj/dataset/RGBT210testingset`. The older data-status document and M61/M63 text still say it is missing and need a factual update. RGBT234 uses toolkit annotations deliberately; `afterrain/visible.txt` is known to differ numerically from the image-directory version. Matching sequence/frame counts do not erase that version difference.

## Metric protocol checks

| Check | Result |
|---|---|
| RGBT234 MPR/MSR | PASS: per-frame minimum CLE / maximum IoU, then per-sequence curves and macro mean. |
| RGBT210 PR/SR | PASS: original first-frame GT replacement, PR <= threshold, SR > threshold, macro mean. |
| LasHeR PR/NPR/SR | PASS: unchanged original scorer delegates to the same original toolkit functions and special-GT behavior. |
| Prediction / GT semantics | PASS: original TrackerResult rounds float32 predictions with NumPy round(0); modality GT reader preserves fractional values. |
| Score normalization | PASS: fixed frame counts and GT width/height for NPR; no prediction maximum, minimum or mean denominator. |
| Partial/malformed admission | 20 tested rejection cases pass; four provenance/metadata cases above fail. |

Independent scalar calculations matched every threshold of MPR, MSR, PR, SR, PR_LasHeR, NPR, and SR_LasHeR. Alternating modality winners distinguish per-frame maximum from a sequence-level choice, unequal sequence lengths distinguish macro averaging from frame weighting, and fractional GT distinguishes raw annotations from rounded annotations.

**All executed fixture scores are artificial CPU protocol tests, not benchmark results.** The wrapper fixture uses 97 clearly named `CPU_FIXTURE_*` sequences / 291 artificial frames, an explicitly synthetic annotation label, and no image decoding or model output. The unchanged original plan was supplied deliberately to test rejection. All data and outputs remain in the unique private review directory.

Correctly rejected: inconsistent checkpoint hash, cost, candidate count or variant between shards; wrong source map; smoke config/completion; incomplete or mismatched native task/GPU/command copies; missing/duplicate shard indices; backward calls; frame-count/service-time mismatch; NaN/nonpositive/truncated predictions; and altered GT.

## Claim limits and frozen selection

No actual E0/E1 full benchmark predictions, completed performance result, or trained-checkpoint origin is established. `SOURCE_PASS` would only certify the reviewed source protocol. Full neural execution, full-C completion, checkpoint identity and test admission require their own evidence.

The scorer test gate (`tools/score_benchmark_tracking.py:47`) checks declarative frozen-selection fields and hashes but does not authenticate full-dev result artifacts or pre-test selection origin. This is WARN/unestablished in this metric review; the separate NN SOURCE review owns that gate. No existing selection document is treated as a completed validated experiment here.

Image preparation covers all filenames/counts and first/last pairs; interior images were not exhaustively decoded. Local checkpoints and the original large LasHeR manifest were unavailable, so no claim is made that this review hashed those remote files.

## Evidence

- Machine report: `reports/THREE_BENCHMARK_METRIC_SOURCE_REVIEW.json`.
- Private source snapshot and trace: `C:\Users\gb\.codex_tmp\dual_rgbt_metric_source_review_20261015_9d3c74b1`.
- Primary scripts: `verify_sources.py`, `test_protocol.py`.
- Primary results: `source_verification.json`, `arithmetic_goldens.json`, `wrapper_protocol_tests.json`.
- Exact subprocess stdout/stderr and synthetic outputs: `SYNTHETIC_CPU_PROTOCOL_ONLY_R2/`.
- All primary artifact hashes, all real GT/list hashes, every test exit code, and individual case output hashes are retained in the JSON report.

The first private metric fixture used Python lists where LasHeR expects NumPy row arrays and failed with TypeError before wrapper execution. test_protocol_r1.py and its separate fixture directory remain preserved. Only private fixture row types/directory were corrected for R2; all production hashes stayed equal. This is a reviewer fixture correction, not a project-source failure.
