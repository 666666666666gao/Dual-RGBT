# Protected-E0 M1 Offline Training Entry

## 2026-10-31 Source Review And Provenance Fix

The authorized supplemental report found one real admission gap, M1-001.
Its CPU reproduction loaded a synthetic complete cache despite contradictory
collection-plan/source metadata and an equal but unrelated native command.
This is not observed corruption of real data; no complete real cache existed.

The loader now requires the actual collection plan path/digest in the M1 plan,
checks its TRAIN/zero-training mode and stage-specific reviewed source map,
rehashes those source files, compares each embedded plan and both config/receipt
plan digests to that pinned file, and reconstructs the complete native collector
command. It also binds native task identity and physical GPU to the declared job.
The existing full-shard fixture rejects six concrete rehashed metadata variants:
config hash, receipt hash, collector digest, unrelated command, task and GPU.
All six CPU tests pass locally and on actual26 Python3.7 with CUDA disabled.

Trainer SHAea9efd94b51b471e4c436f9c3d8d92972b9db67ba79cbbd3cedd4fb9e69a63f9;
tests SHAaed05ea8a924f8dcff835a2da1a8e0db0852c1bb457d018ba73663e7bf0a2e8a.
The narrow M1 fix review returned SOURCE_PASS/0 blockers, reportSHA
ad8c413cd22da0e6149aed7b72f0a6474ae4d4a4321ec1c82060aca0e1153a9f.
It binds six sources; both six-test CPU suites passed, and ten additional
reviewer provenance counterexamples were rejected. This remains engineering-only,
UNATTESTED/same-family/provisional, with scientific acceptance false.
Historical source hashes below are retained as history. Collector static
engineering review passed separately and real full paired TRAIN collection has
started. M1 remains untrained and requires all eight verified real cache receipts,
actual class coverage and a final bound training plan. Existing user authorization
covers that later launch. No additional human GPU approval is required.

Draft R1 SHA144cf77d... declares the first real head recipe: seed2027,12epochs,
batch4096,lr0.001,weight_decay0.0001,CUDA. launch_allowed remains false and all
receipt/native hashes remain null until the full real collection is complete.
No occupancy padding or test-driven recipe selection is involved.

## Steering And State

The latest user instruction prioritizes the new protected-E0 route because
the old route's metrics are too low. Do not launch additional old gain-only
training/evaluation, loss/K/cost grids or size-tail expansion. Existing live
workers were not terminated; their outputs remain exploratory/ineligible.
Newly available capacity belongs to the new route after its review gates.

Implemented `scripts/train_e0_reliability.py`, not launched on real data.
The standalone 914->128->3 head has 117507 parameters. Original E0 is never
loaded by this trainer, and its online tracker remains unchanged.

## Actual Contract

- Input: eight completed paired E0 TRAIN shards covering exactly 882 sequences,
  with disjoint 706-fit/176-component-holdout partitions.
- Verifies supplied receipt/native hashes, native COMPLETE/exit0/command,
  protected checkpoint, frozen 265 tensors, zero collection updates, online-state
  equality and hashed feature/label/config/progress artifacts.
- Rejects prefix-only caches, duplicated shards/sequences, invalid or nonfinite
  features/overlaps, partition mismatch and missing reliability classes.
- Fits channel mean/std on fit only; constant fit channels are centered without
  rescaling. Stores these statistics with each checkpoint for later deployment.
- Trains the new head only with localization-overlap three-class cross entropy.
  Neither GT nor overlap is a forward input. No class balancing, tuned threshold
  or recovery trigger has been introduced.
- Reports unweighted CE, confusion matrix, macro recall, lost recall and reliable
  interruption fraction on fit and holdout. Best checkpoint uses component
  holdout macro recall, not test scores. No threshold fitting is performed.
- Keeps latest five numbered epochs and independent best/latest files within
  its newly created output directory. Existing weight directories are untouched.
- Requires launch_allowed=true, exact source hashes and a fresh matching
  SOURCE_PASS source review. There is no admitted plan or real-data launch.

Four focused CPU tests pass locally and on canonical26 Python3.7 with CUDA
disabled. They cover fit-only normalization including a constant channel,
confusion/recall reporting, absent-class rejection and prefix-input rejection.
Existing ten E0 tests pass locally. These are engineering checks, not an
end-to-end completed collection/training run or independent source review.

## 03:51 UTC Engineering Follow-Up

Six CPU checks now pass locally and on canonical26 Python3.7, CUDA disabled.
Added a synthetic eight-shard/882-sequence fixture and six-epoch head-only
training fixture. Neither uses real TRAIN images/features or a tracker.

The loader previously accepted an empty synthetic progress file despite a
complete receipt. It now verifies the hashed TRAIN manifest, exact shard order,
dev/test disjointness, each complete sequence length and matched frame count,
every expected fifth valid sample, per-sequence feature count and receipt frame
totals. A fixture with rehashed empty progress is rejected; file tampering is
also rejected. This closes an observed engineering coverage gap, not a detected
corruption in any actual collection (none exists yet).

The training fixture performs 24 optimizer updates in six epochs, checks loss
decrease, actual changed head parameters, fit-only preprocessing despite shifted
holdout features, strict checkpoint reload, correct first-best selection and
epochs2-6 retention. Production trainer now also checks latest checkpoint reload
for exact probe outputs before writing COMPLETE. No inference policy is changed.

Source SHA256:
- Trainer:9e159b7ef27b4787f465058d5d16455df31141850bcaebb89007a0384b82a7cd
- Tests:8576bb6bfc07109d0ebb0846e40f78a5c87507616c2e4a3db6640ce9189431f1

These checks do not exercise real CLI source-review admission, real paired
collection, GPU head training, failure-recovery efficacy or benchmark gain.
New source hashes require fresh review; no old approval is reused.

## Critical Next Action

Fresh review of `scripts/collect_e0_reliability_train.py` remains unavailable.
The old observer review predates that collector and cannot approve it.
Messaging the existing review chat requires human authorization; requested again
after the user changed priorities, but no message has been sent. The new trainer
also needs fresh review before its own admission. Do not downgrade the requested
model route or claim a self-review is independent.

Native spawn discovery now exposes a subagent schema, but its listed models
still omit the mandated gpt-6-astra route. No reviewer was spawned or model
substitution attempted. Existing-chat messaging authorization remains pending.

After review: collect all eight complete paired TRAIN shards, inspect actual
class/sequence coverage, then admit a declared head-only training recipe. The
existing prefix contains 89 reliable/1 uncertain/0 lost states and is unsuitable.
Do not train it, substitute the old21e07 cache, or report a tracking improvement.

The upstream E0 saw the component holdout. This is not independent generalization.
M1 currently describes localization, not target identity or recoverability.
M2 candidate-local identity recovery and M3 separate memory commit are still
unimplemented/unvalidated, and all benchmark targets remain unmet.

## Last Runtime Evidence

At03:51:18/19UTC,15inferencechildren remain live, freshtraces<=10seconds.
25GPU0/GPU3idle2MiB;utilityE6/E7/E9/E10=240/224/235/215seq.
26matchedparent105seq,all882E8=148seq. No new-route GPU launch.

At03:36:31/32UTC, old utilityE8 completed normally with exit0 and all245seq/
220703frames.25GPU0 andGPU3 now both idle2MiB;15children remain live (25four,
26eleven),freshtraces<=10.6s. Scoring was already attached; do not duplicate.
No new old-route follow-up or new-route admission was launched.

At2026-10-31 03:24:00/21UTC,16 inference children were live with fresh traces.
25 GPU0 was genuinely idle after old refreshed E12 completed normally;25
remaining GPU counts were0/2/2/1,26 counts3/3/3/2. No new-route GPU job launched.
Do not describe all eight GPUs as occupied.

Old E12 closure only:245sequences/220703frames, PR66.272251/NPR61.538472/
SR50.243233. Recorded concurrent FPS12.813566,mean78.042287ms/P95111.844277ms,
103218887parameters. CPU analytic MAC estimate42.337089GMAC/tracked frame,
42.290092GMAC including reused initialization; not a fresh GPU profile.
No more old-route diagnostic experiments are scheduled.
