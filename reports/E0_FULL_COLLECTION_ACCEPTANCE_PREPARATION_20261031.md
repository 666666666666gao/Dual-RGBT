# Full TRAIN Acceptance Preparation

## Actual Completed Work

- Added `tools/verify_e0_paired_train_collection.py` without modifying the
  reviewed collector, M1 trainer, E0 model or tracker. It requires complete
  eight-shard native receipts, reuses the reviewed M1 loader, verifies the
  collector-stage parent review and its exact source/input draft bindings,
  checks every full float32 trajectory and reproduces each saved TRAIN IoU
  using the collector's original mixed-precision operation order.
- A separate acceptance report and final M1 plan are written only after all
  checks pass. The existing first-recipe draft stays `launch_allowed=false`.
  The tool requires a fresh supplemental engineering review of its exact
  eight-source map and the exact training draft.
- Two focused CPU tests passed locally and on actual26. They include nine
  refreshed-hash corruption cases and rejection of unreviewed finalization.
  These are synthetic tests, not real M1 training.
- A partial check on actual26 at 05:37:36 UTC passed for 167 completed real
  sequences, 73,607 frames and 14,577 sampled labels. All sampled IoUs
  reproduced exactly. This used current per-file hashes from live caches;
  final native COMPLETE receipts do not yet exist. It does not admit training.
- Supplemental review was requested in the existing human-authorized chat,
  with an explicit new output path and instruction to preserve all historical
  reports. Actual model attribution remains UNATTESTED, same-family,
  provisional; scientific result acceptance remains false.

## Server25 Same-Path Mirror

The existing `/data2/gb/Dual-RGBT` mirror on server25 has the same manifest byte
hash as canonical26, unlike server25's `/home/gaob/Dual-RGBT` manifest.
All 918,870 TRAIN image paths exist on25. An initial read-only audit found 120
of 123 collector sources and 188 of 190 inputs already byte-identical, with
only three missing sources, the protected checkpoint and an input plan.

The missing files were copied into that existing mirror. The checkpoint was
copied from25's existing exact protected checkpoint onto its data disk.
Subsequent audit verified all 123 sources and all 190 inputs byte-identical,
with no missing or different entries. Related standalone M1 sources, tests,
plans and review reports were also deployed. The six reviewed M1 CPU tests
and the two new verifier tests passed on actual25. No GPU job or cache-transfer admission follows from this readiness
check alone. The mirror can avoid introducing different absolute root paths
when preparing future new-route work.

Before training from transferred26 caches on25, bind the actual completed
source-host native evidence and verify every transferred artifact byte hash;
do not relabel those native collection jobs as executions on25. The existing
eight26 collections should finish without interruption or duplicate collection.

## Supplemental Review And Automatic Handoff

The returned verifier report
`E0_M1_FULL_COLLECTION_VERIFIER_SOURCE_REVIEW_20261031.json`, SHA256
`1f00d3680f2f3bbcf57ca6f441b9b799162072a0e04b79851765cb3f0ee1e483`,
records SOURCE_PASS with zero blockers and the exact eight-source/draft
bindings. The reviewer exercised the real verifier/loader on full-size
synthetic data and rejected16 additional counterexamples. These are
engineering tests, not completed real collection or scientific acceptance.

At05:52 UTC the operational one-shot controller was verified live on26 as
PID78951, state WAITING_COLLECTION, with240-second polling. Its archived source
is `reports/source_snapshots/e0_collection_handoff_20261031.py`, SHA256
`2907858457a2f9a3a75dba8fda5ddfd7d7af0eda21259c7188d54a30b80d3b70`.
Status and logs use `reports/E0_M1_COLLECTION_HANDOFF_CONTROLLER_20261031`.
It waits for all eight native COMPLETE/exit0 records, runs the reviewed verifier,
then launches the existing recipe through the native GPU wrapper only after
successful full acceptance. It stops on failure and never restarts collection.
Training uses the least-used26GPU when below14,000MiB and writes
`runs/protected_e0_offline_m1_20261031`. Do not duplicate these operations while
the controller is live.

## Current Dependencies

At 05:43:49 UTC all eight real26 collector children were live with fresh current
files. Complete progress was 201/882 sequences, 93,847/459,435 frames and
18,604/91,091 saved states. The eleven pre-existing inference children were
also verified live at 05:33 UTC; no new completion, failure or score was found.
Four25 GPUs remain idle pending meaningful new-route workload admission.

The verifier review has passed. Complete eight-shard real collection and
successful full acceptance remain required before finalizing the first M1 plan
and real training. M2 still needs actual
candidate identity data; M1 observations cannot train M2. M3 is unimplemented.
Original benchmark targets remain unmet, with no new tracking/FPS claim.

Weight inventories found46 numbered files on25 and22 in the old26 snapshot,
with no run retaining more than five numbered epochs. No proven-useless
weights were identified or deleted; preserve best/latest and audit dependencies.
