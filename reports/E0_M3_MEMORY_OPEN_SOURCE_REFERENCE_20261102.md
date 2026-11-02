# M3 memory commitment: primary source comparison

## Actual project problem and boundary

PROTECTED_E0_REDESIGN_20261031.md requires separating provisional
localization from memory commitment and measuring future TRAIN
consequences from the same cloned state. Current M1 can undo a proposed
dynamic-template write after predicting reliability; it does not learn
the future effect of accepting that write. The relative-M2 core may
change position/history but intentionally leaves template commitment
with M1. Its full TRAIN and paired trajectory results are still pending.

Observed full LasHeR M1 guard PR/NPR/SR74.223810/70.504968/59.018088
remains below75.8/71.9/60.2, as recorded in
E0_M1_LASHER_AND_SEED2028_RGBT234_FULL_FINDINGS_20261101.md.
These observations do not prove that template contamination explains
the remaining error. Previously observed blanket-freeze regression
also rules out proposing unconditional template freezing as the remedy.

## Existing open-source mechanisms

| Project and exact source | Verified behavior | Consequence for our proposed contribution |
|---|---|---|
|STARK, researchmm/Stark, revision7e6bae4bdfdc8c3495fa9ee554c4cb78d0ab0f6a, lib/test/tracker/stark_st.py:88-97|A classification confidence and configured update intervals gate replacement of extra templates. The first-frame template remains in the first slot.|Confidence gating, interval-based writes and retaining an initial anchor already exist. These alone cannot establish M3 novelty.|
|MixFormer, MCG-NJU/MixFormer, revision8b87088f45af02c614a277176c63181261e93de3, lib/test/tracker/mixformer_cvt_online.py:142-168|Each frame can replace a provisional best template after passing confidence and decayed-score comparisons. A separate interval commits it to online memory, with bounded replacement for multiple slots.|Separating provisional candidate storage from interval commitment also exists. A simple buffer plus delayed write is not a new mechanism.|

Primary source locations:

- https://github.com/researchmm/Stark/blob/7e6bae4bdfdc8c3495fa9ee554c4cb78d0ab0f6a/lib/test/tracker/stark_st.py
- https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/lib/test/tracker/mixformer_cvt_online.py

E0_M3_MEMORY_PRIMARY_SOURCE_AUDIT_20261102.json records the exact fetched
byte SHA256 and relevant line locations. This inspects implementation;
it does not reproduce either project's accuracy or establish cross-domain
effectiveness. No third-party implementation is copied into this project.

## Evidence needed before implementation

The research question is whether a write that appears reliable now causes
future harm, beyond what the current reliability/identity signals explain.
A learned future-effect decision would require actual TRAIN evidence.
It cannot be inferred from confidence, a memory buffer or these references.

After relative-M2 TRAIN acceptance and independent paired trajectories,
use the existing prescribed H0/H3/H5 TRAIN comparison to isolate memory
effects. At a fixed write opportunity, fork the same complete tracker
state into write/hold arms. Preserve model weights, input order,
localization decision, crop settings, query counters and initial anchor;
only the proposed dynamic memory commit differs. Advance both arms over
the same future images. Report paired future IoU changes, reliable-state
damage, lost-state recovery, write counts and complete runtime cost.
Future images and GT belong only to offline TRAIN label generation and
evaluation, never the deployed decision's inputs.

A useful write predictor must improve this paired TRAIN evidence against
the existing M1 guard and a declared confidence/interval reference before
benchmark deployment. Retain protected E0 if no benefit is demonstrated.
Do not copy STARK/MixFormer numeric thresholds into the current pipeline,
retune from TEST, enlarge the buffer speculatively or launch new M3
collection while the existing M2 chain is unfinished.

This is a conditional experimental direction, not an implemented M3,
validated three-module contribution, novelty claim or benchmark result.
