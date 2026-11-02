# Full Current-E0 TRAIN Supplement

Previous goal turn made concrete progress: the real93-frame supplement
smoke completed with exit0 and independent CPU acceptance. This turn
extends that exact computation to the complete original TRAIN collection.
The original three-benchmark goal remains unmet.

## Bound Data and Execution

Actual host audit:
`reports/E0_M2_CURRENT_FULL_HOST_AUDIT_R1_20261102.json`.
It rechecked all882 original NPZ hashes and all918,870 TRAIN image paths
with no missing file, reran the reviewed smoke verifier, and confirmed all
eight proposed task/run paths were absent on host26. File presence is not
a claim that every image pixel has been hashed.

Disabled draft:
`reports/e0_current_full_r1_20261102/draft.json`,
SHA256 `203420f6ffe1802d99536d5f6b9f8850f71b1e8e06c330ac1ea3d20cc279c4be`.
Its139 sources contain all134 unchanged R2 smoke dependencies plus five
new full-collection files. The original706fit/176component-held split,
459,435frames,18,711 sample keys and eight fixed stride shards are bound
to the original accepted collection. Even shards useGPU0, odd shardsGPU1.
No new host25 or host26GPU2/3 work is included.

`scripts/collect_e0_current_full.py` preserves the original smoke's neural
computation: original crop binding, two paired trackers and one observer,
fixed initial templates, cached predictions and paired states checked
every frame, with two supplemental encodes only on sampled frames.
It changes sequence length, shard selection, progress and receipt counts.
Original24 candidate descriptors and anchors remain immutable and reused.
It does not compute GT-driven actions, train E0 or commit an intervention.

`tools/full_current_e0_supplement_contract.py` binds the original acceptance,
entire cache map, exact source/input/review/draft identities and actual
completed smoke plan/receipt/native/acceptance. It verifies every native
shard and artifact, every full sequence array and frame join, and all
dataset totals before full acceptance. `training_allowed=false` remains;
the subsequent relative trainer needs its own reviewed recipe and real
update/reload smoke.

Local `tests/test_e0_current_full.py` passed10 tests: three new full
partition/native/100-frame cases plus the seven existing smoke checks.
Later sample misjoins, corrupt final predictions, duplicate tasks, wrong
GPU, shard names, totals and fit/holdout changes are rejected.
Fresh exact-source review was requested from the authorized existing chat.
No full collector has been deployed or launched at this draft stage.

## Training Data Interface Prepared Separately

`tools/relative_e0_train_data.py` is outside this139-source collection map,
and has not been deployed, reviewed for training, or used for real updates.
It joins exact sequence/frame keys only after full supplement acceptance,
reuses accepted anchors/candidates, reconstructs original stored IoUs and
uses common float64 coordinates for relative comparisons.

Ambiguous original candidate labels remain outside the relative target
and loss denominator. Inference retains actual measured coverage, and
metrics separately count unknown-candidate selections rather than claiming
they are supervised negatives. Current E0 without covered evidence is
excluded from the loss; its executable scorer keeps E0.

Four local CPU tests passed for unknown-label loss gradients, current
coverage exclusion, reliable harm/lost recovery/sequence-macro accounting,
and the predeclared acceptance. These are synthetic math checks.
The acceptance stays fixed: among the same component-held TRAIN states,
zero reliable-to-below0.5 crossings, nonnegative sequence-macro reliable
IoU change, and more than179 recoveries among800 lost states. The upstream
E0 has seen the component holdout; it is not independent final validation.
No threshold or checkpoint is selected using TEST.

## Existing Evaluation Evidence

At04:23UTC all four original evaluation children were live:
726624,1140162,1147373,3795734. Subsequent traces were fresh and completed
counts reached epoch40LasHeR114seq/108,091frames,
epoch39LasHeR40/30,871, epoch39RGBT21078/40,492,
and original E0+M2LasHeR225/213,486.
No task was terminal and no scorer or next queue job was started.

Only real COMPLETE/exit0 and full count verification admit a new official
score. Existing epoch40RGBT210/234 and R15RGBT210/234 scores stay archived,
with no repeated scoring or TEST-based checkpoint selection.

## Actual Full Launch And First Progress

Final full source review completed SOURCE_PASS/0 with38 independent
negative probes and10 passing CPU tests. Final report SHA256:
`77eb87e9856fc3e8b061f761a20efb989aa477b2d9a18c35b13ac95bb4ef0c66`.
Final plan SHA256:
`61f3dc02d5727093740547666cbf2af95765efa683736f3fa074e7cc7167feef`.
Only the five new files, report, draft and plan were deployed; all10 CPU
tests and canonical full admission passed on actual host26.

The first operational capacity check stopped before any launch because
available RAM99.22GiB was below its initially chosen100GiB reservation.
Existing matching-model evaluation RSS was3.17/5.32/5.37GiB; eight at the
highest observed RSS are approximately43GiB. The operational reservation
was changed to64GiB, preserving margin above that evidence. No reviewed
source/draft/data/smoke condition was changed. The second check passed
with99.54GiB available and GPU0/1 free24,104/17,313MiB.

All eight native wrappers launched at2026-11-02 04:51:25UTC:
1270944/1270949/1270953/1270955/1270956/1270957/1270960/1270962.
Their actual children are1270958/1270959/1270961/1270963/1270964/
1270965/1270967/1270968. At04:54:12UTC all eight were live, each had
an actual NVIDIA compute context, and progress/current files were updating
within5.90seconds. Combined17sequences and3,148frames completed.
GPU0 used9,095MiB andGPU1 used15,883MiB; GPU2/3 workloads were unchanged.
Initial concurrent source/cache checks took time before model loading;
no static startup count or observation timeout caused a restart.

Actual launch evidence:
`reports/e0_current_full_r1_20261102/launch.json`;
first process/native/progress/GPU audit:
`reports/E0_M2_CURRENT_FULL_LAUNCH_AUDIT_R1_20261102.json`.
This is genuine full collection in progress, not completed acceptance.
After all eight native statuses are COMPLETE/exit0, run the reviewed
`tools/verify_e0_current_full.py` with the bound plan and a new acceptance
output. Preserve failed/partial evidence and do not reuse failed task IDs.
Prepare the separate relative trainer and real update smoke while collection
runs; the fixed recipe draft is
`reports/E0_M2_RELATIVE_TRAIN_RECIPE_DRAFT_20261102.md`.

The old R15 LasHeR evaluation has now completed and been scored once:
PR/NPR/SR68.165608/64.880209/54.202170, FPS6.268035,
parameters103,490,697, mean tracked-frame103.284795GMAC.
It regressed versus matched E0 and is not promoted. Complete evidence and
comparison are in `E0_M2_R15_LASHER_FULL_FINDINGS_20261102.md`.
