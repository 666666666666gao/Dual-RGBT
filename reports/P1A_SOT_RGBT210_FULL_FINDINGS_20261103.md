# Fixed Epoch20 SOT RGBT210 Full Evaluation

Native task `eval_p1a_e020_rgbt210_sot_p26_r1_20261102` completed with exit0.
All 210 sequences and 104706 frames passed the existing complete-artifact,
source, checkpoint, precision, trace and official toolkit gates.
The official CPU scorer ran once; raw evidence is archived in
`reports/p1a_completed_scores_r1_20261103/sot/rgbt210/`.

| Metric | Value |
| --- | --- |
| PR | 87.8488497279 |
| SR | 64.2868049169 |
| Total Parameters | 103374215 |
| End-To-End FPS | 12.1027904756 |
| Mean Tracked Frame GMACs | 55.5155381351 |

Score SHA256:
`0c03ced00198fc80fe2d61e5a0960cdc6188ec186d2b1b85f5435c4e16bcee62`.
Compute SHA256:
`487247315433b8c5f6c0a3c656204a721aebd392ba8ee380c3ad6aba5db05816`.

PR is below the original 88.7 target by 0.8511502721 points; SR exceeds
62.9 by 1.3868049169 points. This fixed pre-P1B epoch20 control does not
establish an improvement from policy refresh or C2 and does not meet both
RGBT210 targets. TEST is exploratory and ineligible for checkpoint or branch
selection. FPS includes decoding and initialization under concurrent GPU
load; MAC accounting excludes initialization, nonlinear operations and
bookkeeping. The route profile is source-matched from P25, not a fresh P26
GPU profile.
