# Fixed Epoch20 SOT RGBT234 Full Evaluation

Native task `eval_p1a_e020_rgbt234_sot_p26_r1_20261102` completed with exit0.
All 234 sequences and 116650 frames passed the existing full-artifact,
source, checkpoint, precision, trace and official toolkit gates.
The official CPU scorer ran once. Raw evidence is archived under
`reports/p1a_completed_scores_r1_20261103/sot/rgbt234/`.

| Metric | Value |
| --- | --- |
| MPR | 89.9216752845 |
| MSR | 66.8347279957 |
| Total Parameters | 103374215 |
| End-To-End FPS | 12.3040910759 |
| Mean Tracked Frame GMACs | 55.1313497378 |

Score SHA256:
`f7838ee14e7332a5f93eb63855dd3909318829a4a576222effbb0161c1770567`.
Compute SHA256:
`46a3b24e6406330c3bb48bf1349754f2762e71e26c20f9cf03c28a7d534c2212`.

MPR falls below the original 90.9 target by 0.9783247155 points; MSR exceeds
65.6 by 1.2347279957 points. This fixed pre-P1B control does not meet both
targets and is not evidence of gains from policy refresh or C2.
TEST is exploratory and ineligible for checkpoint or branch selection.
FPS includes decoding and initialization under concurrent load; MACs exclude
initialization, nonlinear operations and bookkeeping. The source-matched
route profile comes from P25 and is not a fresh P26 GPU profile.
