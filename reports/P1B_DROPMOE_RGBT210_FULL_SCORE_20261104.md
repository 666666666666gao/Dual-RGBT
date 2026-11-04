# Fixed P1B DropMoE RGBT210 Full Score

## Completed Evidence

Native task `eval_p1b_e005_rgbt210_dropmoe_p25_r1_20261104` completed
with exit code 0 at 2026-11-04 10:40:22.837997 UTC. The fixed policy
checkpoint is epoch005, SHA256
`6f29659e3f03b5d972296059d3c2d2f00ca93cb743e155b7ad5a71a53e63f0f4`,
with cost weight 0.05. Full coverage is 210 sequences / 104,706 frames.
Completion records unchanged model state, zero optimizer updates and zero
backward calls. Precision flags are all false.

The reviewed helper `work/score_p1b_rgbt_completed.py`, SHA256
`2d2e97fc97616d2c3a1ae85ce7fcf2dce16338c9322749190f27346f76f5d4da`,
ran once after native/source/config/checkpoint/precision/GT/trace/count gates.
It used the unchanged supplied toolkit on CPU. The nine-file archive is
`reports/p1b_rgbt_completed_scores_r1_20261104/dropmoe/rgbt210/`.
All eight raw-file hashes in `audit.json` were verified locally after retrieval.
Do not repeat the scorer or archive helper for this result.

The authorized reviewer independently checked the local archive, all eight
hashes, native/config/receipt/review bindings, 58-source maps, score/curve
consistency and P1A manifest/toolkit/protocol/GT pins: PASS, zero blockers.
The reviewer did not reread remote raw artifacts, weights or images, and did
not execute the scorer. Attribution is `actual_model=UNATTESTED`,
same-family/provisional; this is an evidence audit, not scientific acceptance.

## Accuracy

All numbers are percentage points.

| Fixed Configuration | PR | SR |
|---|---:|---:|
| P1A epoch20 before policy refresh | 88.98710359061050 | 64.63427333580411 |
| P1B policy epoch005 | 88.94166124922366 | 64.85677623306427 |
| P1B minus P1A | -0.04544234138685 | +0.22250289726016 |
| Registered stronger ep20 target | 91.2 | 65.4 |
| Remaining gap to stronger ep20 target | 2.25833875077635 | 0.54322376693574 |

The P1A comparison comes from
`reports/p1a_completed_scores_r1_20261103/dropmoe/rgbt210/score.json`.
This is mixed evidence for policy refresh: SR increased slightly and PR
decreased slightly. It does not establish consistent improvement, statistical
significance, C2 performance, or success against the corresponding stronger
ep20 target. Numerically exceeding the lower SOT target 88.7/62.9 does not
satisfy the ep20 comparison.

This is exploratory TEST evidence: `exploratory_test=true`,
`test_eligible=false`. It must not select a checkpoint, branch, threshold,
recipe or architecture, and is not an independent final validation.

## Runtime Boundary

RTX 3090; search256/template128/two templates; total parameters 103,374,215.
Observed FPS 20.560067144976227, mean 48.637973453522804 ms and P95
66.1090210051043 ms. Timing includes decode and initialization, excludes
model loading, and was measured under concurrent system load. No isolated
throughput, speed advantage or new MAC measurement is claimed.

## Capacity And Next Gates

At 10:46 UTC the completed wrapper2085914/child2085916 had exited and P25
GPU2 was idle. P25 GPU0 remained on P1B LasHeR, GPU1 on C2-zero policy FIT,
and GPU3 on P1B RGBT234. P26 GPUs0/1 remained on P1B LasHeR/RGBT210;
GPUs2/3 remained on C2 FiLM/zero replay verification. The seven actual
children were live. P25 stayed at 250 W and below 85 C.

At the 10:48 UTC progress read, P25 LasHeR was 106/245 sequences and
106,964/220,703 frames; P25 RGBT234 was 204/234 and 104,638/116,650.
P26 LasHeR was 92/245 and 75,284/220,703; P26 RGBT210 was 138/210 and
65,968/104,706. C2-zero FIT had 100 flushed updates in epoch1; neither
P26 replay had produced its final acceptance. These are progress only.

Keep the idle card available until a meaningful task has its own exact
admission. SOT210 COMPLETE0 precedes the separately reviewed SOT234 R2
follow-up. Matching C2 replay PASS precedes its own policy FIT; actual
five-epoch/890-update FIT COMPLETE0 and fixed epoch005 precede C2 evaluation
plan review, CPU reconstruction, TRAIN smoke and full benchmark execution.
P25 FiLM remains blocked by missing exact original s0/s2 reviewed plan bytes.
No active job, recipe, snapshot, environment or failed evidence was changed.
The research goal remains active and unmet.
