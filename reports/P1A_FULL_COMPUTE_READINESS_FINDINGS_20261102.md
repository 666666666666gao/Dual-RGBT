# P1A Full Compute Dependencies Ready

Both actual parent CPU-scoring environments completed the new
source/profile/real-smoke binding check on 2026-11-03 UTC.
Result: `CPU_COMPUTE_BINDING_READY` for DropMoE and SOT.
The unchanged aggregation tool is now present in both active snapshots.
The measured profile was preserved on25 and made available on26.
Existing collector/evaluator sources and their running tasks were unchanged.

| Actual Check | P25 DropMoE | P26 SOT |
| --- | ---: | ---: |
| Evaluation source hashes verified | 49 | 49 |
| Shared profile/evaluation source hashes matched | 25 | 25 |
| Strictly loaded actual smoke parameters | 103,374,215 | 103,374,215 |
| Actual TRAIN smoke native COMPLETE/exit0 | Yes | Yes |
| New GPU work or scorer rerun | No | No |

This enables later use of fixed-shape E0 route arithmetic in both
branches. Only the P25 profile was executed; SOT weights, accuracy,
FPS and route frequencies were not transferred. Current common model
sources, input shapes and real smoke parameter counts agree.
Actual full-route trace weighting remains necessary and has not run.

## Exact Evidence

- Review stage `P1A_fixed_epoch20_compute_readiness`, SOURCE_PASS0:
  `78b76f02a757700b928dce6679809b4149ecd3d9458d4d3ba8202c6897531583`
- Unchanged `tools/summarize_e0_full_compute.py`:
  `efd27f1d4ab4de69e77c91e2033eb644a7b52904c319428c108dce2cc54aa77e`
- Local transport `work/deploy_p1a_compute.py`:
  `0118629df49b7a617467c52f60cfd4d3634b8893df4fc88afd88a6e129d2f1f3`
- Exact measured profile:
  `9d63719a1d16d83310b7100939a64183dcf5c010c96b7d4522c1602651e8542a`
- Raw remote DropMoE readiness JSON:
  `ed7e99caeeda1f2a17e73dce21c032b6a99b2410e2187fa744d91b8c372198ba`
- Raw remote SOT readiness JSON:
  `3d8892e4a21fd020fc0a987799c5ca8f1dedf83f4f0bb52863f66d21416db1d7`

Draft, exact raw remote readiness files and execution stdout/stderr:
`reports/p1a_compute_r1_20261102/`.
The reviewer independently ran three existing CPU tests and embedded
readiness positive/negative fixtures; these are engineering checks,
not full tracking results. Attribution remains
UNATTESTED/same-family/provisional, scientific acceptance false.

## Full Result Procedure

Use the already bound official scorer once per completed full P1A
run, with commands/outputs from its branch evaluation review.
After official scoring, run the unchanged summary tool using the
matching CPU-scoring interpreter, snapshot root, actual full run,
existing score path/hash and exact profile path/hash.

Profile path relative to either snapshot:
`reports/P1B_E0_CURRENT_ROUTE_MAC_CPU_CROSSCHECK_P25_20261102.json`.
The P25 name identifies the actual profile origin on both hosts.
Write the summary beside the score as `<score-stem>_compute.json`,
outside the immutable run artifact tree. Require a fresh output.

The original tool still requires native COMPLETE0, full benchmark
sequence/frame counts, current source and parameter agreement, exact
score/profile hashes, all saved artifact hashes and every tracked
frame/route/token trace. It does not rerun the scorer.

Report official PR/NPR/SR or MPR/MSR, full-run FPS and mean/P95 time,
total parameters and mean tracked-frame GMACs together. MAC weighting
excludes one initialization frame per sequence; FPS includes decode
and initialization under concurrent load. No isolated-speed or
whole-system-FLOPs claim. All TEST outputs remain exploratory,
test-ineligible and unusable for checkpoint/recipe selection.

At the subsequent direct observation, all six full controls remained
RUNNING with fresh traces. P25 completed8/57/45 sequences for
LasHeR/RGBT210/RGBT234; P26 completed8/43/27. These are progress counts,
not partial accuracy metrics. Original eight collectors continued;
P26's long6981-frame sequence had completed normally.
Full acceptance and the separately admitted B512 five-epoch P1B FIT
remain the training critical path. No benchmark target is proven yet.
