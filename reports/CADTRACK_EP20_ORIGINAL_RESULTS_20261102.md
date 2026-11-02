# Original CADTrack ep20 Reproduction: Full Results

## Scope

This is a fresh full-test reproduction of the supplied checkpoint
`/data2/gaob/Dual-RGBT/checkpoints/CADTrack_ep0020.pth.tar`
(SHA-256 `d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`)
using the original stripped-CADTrack tracker, not the selective E0 inference
path and not a new finetune. The isolated P25 snapshot is
`/home/gaob/Dual-RGBT/snapshots/cadtrack_ep20_original_r2_20261102`.

R2 evaluator/scorer source review is `SOURCE_PASS`, zero blockers, and binds
the exact 71-file source map. All three benchmarks completed full TEST
inference with native exit 0 and full sequence/frame coverage. The official
scorer was run once per benchmark after the missing dependency was
approved by the additive R3 scoring-dependency review
(`SOURCE_PASS`, zero blockers). R2 sources, plans, run configs, receipts,
native statuses and predictions were not modified.

## Results

| Benchmark | Sequences / frames | Metrics | Project target | Difference |
|---|---:|---|---|---|
| LasHeR | 245 / 220,703 | PR 74.2413, NPR 70.6388, SR 59.2021 | 75.8 / 71.9 / 60.2 | -1.5587 / -1.2612 / -0.9979 pp |
| RGBT210 | 210 / 104,706 | PR 89.0879, SR 64.9922 | 88.7 / 62.9 | +0.3879 / +2.0922 pp |
| RGBT234 | 234 / 116,650 | MPR 90.7308, MSR 67.1104 | 90.9 / 65.6 | -0.1692 / +1.5104 pp |

The user-reported DropMoE-derived stripped-CADTrack baseline is LasHeR
75.6365/71.7728/60.1257 and RGBT234 90.6771/66.8452. This reproduction's
LasHeR deltas are -1.3952/-1.1340/-0.9236 percentage points; RGBT234 deltas
are +0.0537/+0.2652. The close RGBT234 match is a reproduction sanity check,
not a two-point backbone gain. The LasHeR decrease is the measured result,
but the supplied baseline has not been independently rerun under the same
evaluator, so this is not a controlled causal backbone comparison. This is
also the original dense CADTrack inference path, not the selective E0 route
whose retained policy heads see a changed executor after finetuning. Backbone
pretraining strength alone does not guarantee end-task metric gains; the
initialization, fine-tune path, inference configuration and tracker behavior
jointly determine the measured outcome.

The current results are exploratory TEST outputs
(`exploratory_test=true`, `test_eligible=false`). They must not be used to
select checkpoints, tune parameters or claim independent final validation.

## Runtime and Artifacts

Parameter count is 103,138,437. Measured FPS is 20.9869/25.4281/25.5134 on
LasHeR/RGBT210/RGBT234; mean latency is 47.6488/39.3265/39.1951 ms and P95
is 71.6551/49.0423/49.1691 ms. Timing includes decode and initialization, excludes model
loading, and was collected under concurrent host load; it is not isolated
throughput. MACs have not yet been measured.

Score JSON, curve NPZ, native status, completion receipt and supplemental
scoring provenance are preserved under
`reports/cadtrack_ep20_original_r3_scoring_dependency/`.
The provenance records the exact scoring command, full consumed-input
inventory, pre/post hash equality, exit status, and output hashes.
