# Relative M1-Gated M2 Runtime Preparation

## Measured Problem And Existing Integration Gap

The accepted original TRAIN replay found326/2899reliable states harmed by
replacement and179/800lost states recovered. The relative head therefore
scores current E0 as action zero instead of learning only an unconditional
no-match scalar. The fixed full TRAIN recipe and acceptance gates remain
unchanged.

The existing `E0M1M2Tracker.track` invokes the original head with three
arguments (anchor, candidates, coverage), while `E0RelativeIdentity.forward`
requires five, including current E0 descriptors and coverage.
The old constructor also loads the original epoch9checkpoint. Reusing
that path would neither load nor exercise the newly trained relative head.
No existing admitted runtime file was changed.

## Added Runtime Core

`dual_rgbt/e0_m1_relative_m2_online.py` provides
`E0M1RelativeM2Tracker`. It reuses the original initialization, sampling and
observer cleanup, and constructs the existing protected E0 tracker and
M1template guard. Its caller supplies an admitted qualifying relative head;
construction is not checkpoint, source or run admission.

Only a predicted M1lost state at the existing sampled cadence invokes M2.
The current descriptor uses the preceding-state crop, fixed initial
templates and existing current-proposal evidence helper, matching the
supplement collection geometry. Existing candidate generation/order is
unchanged. Relative action zero keeps the actual guarded E0 result, and
uncovered-current masking/exact-tie KEEP remain in the existing head.
Replacement changes position/history only; it neither adds template writes
nor changes frame/since-query counters.

Diagnostics distinguish the two extra current-evidence encoder calls from
the24candidate encoder calls. Initialization also retains the existing two
anchor encoder calls. A future admitted runner must time the entire frame
and aggregate the actual executed routes, rather than treating the base E0
route's seconds or head-only MACs as integrated recovery cost.

## Verification Scope

Five local CPU tests passed for sequence diagnostic reset, non-lost or
unsampled bypass, KEEP state preservation, replacement state/history
updates without a template write, and uncovered-current KEEP using the
actual relative head. The first test invocation exposed Python3.7
`unittest.mock` call-inspection syntax in the new tests; using the existing
repository's tuple-style call inspection fixed the tests. No runtime
behavior was changed to satisfy that test harness issue.

Independent engineering review returned CORE_SOURCE_PASS/0blockers:
`E0_M2_RELATIVE_RUNTIME_CORE_SOURCE_REVIEW_R1_20261102.json`, SHA256
`aeec8061961b1d2c084dedf1a4f93b75aa4fda044aa7593e64ba7e5d7d347c17`.
The report's stage, permission boundaries, two exact source hashes and all
read-only dependency hashes were verified locally. The reviewer ran22
additional independent CPU scenarios, including actual inherited
initialization/M1guard/current helper/24candidate helper/head selection,
last candidate indexing, tied KEEP, fractional coverage and state commits.
The local13test regression also includes the existing M1guard/old M1M2.
Attribution remains UNATTESTED/same-family/provisional/scientificfalse.
The local Anaconda environment's installed `scripts` package required
explicit repository module binding inside the independent CPU harness;
future runner imports are not admitted or verified by this core review.
These tests are not real GPU integration, trained-head acceptance,
closed-loop recovery measurements or benchmark evidence.

## Execution Boundary

This new core is local only. There is no new remote deployment, launcher,
GPU task or TEST invocation. All154active full-training/collection source
bindings and the fixed recipe are untouched.

Next prerequisites are complete882sequence collection acceptance, actual
12epoch full TRAIN completion and terminal verification. If no epoch meets
the predeclared protection/recovery gates, retain E0 and do not integrate a
rejected head. A qualifying head still needs a separately reviewed, bound
paired TRAIN run with actual state/trajectory/encoder evidence before any
full benchmark execution. The core review alone cannot grant that permission.

At07:01UTC all8collector children were live,616/882sequences and297405/459435
completed frames. Per-shard lifetime throughput estimates ranged from
10to128minutes remaining; these exclude work inside unfinished sequences
and are planning estimates, not completion promises. P25all10evaluation
children and both automatic CPU controllers were live. No new full score
or full relative training acceptance existed at that check.

At07:11UTC shard4 finished COMPLETE/exit0,110sequences/44942frames/1841sets.
Native/config/receipt plan and command identities, all113recorded artifacts
and139collector source hashes passed a byte audit. Exact terminal evidence
is archived under`e0_current_full_r1_20261102/completed_shards/s4/`.
The remaining seven children were live; totals were670sequences/319748frames.
This is one completed shard, not full collection acceptance.
