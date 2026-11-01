# R15 E0+M2 RGBT210 Full TEST Findings

## Complete Result

Native task eval_e0_m2_rgbt210_r15_p26_20261101 completed with exit0
on host26 GPU1. The official scorer succeeded once for all210 sequences
and104706 frames. Its source/input/receipt checks passed, followed by the
separate complete-artifact/trace-weighted compute summarizer.

| Variant | PR | SR | FPS |
| --- | ---: | ---: | ---: |
| Frozen E0 control | 88.828508 | 66.124608 | 11.134969 |
| E0+M1 guard | 89.516426 | 66.612586 | 11.137673 |
| R15 E0+M2 | 81.768710 | 61.201206 | 8.259151 |
| M2 minus E0 | -7.059798 | -4.923403 | - |

All three bind parent checkpoint
ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c.
Earlier controls are archived in reports/e0_m1_full_scores_r1_20261101.
Timing was collected under concurrent workloads, so the FPS difference
does not establish isolated throughput loss. R15 is E0+M2 without M1 or M3.
PR/SR targets88.7/62.9 remain unmet by this variant, by6.931290/1.698794.
These are exploratory TEST observations, test_eligible=false; do not use
them to select checkpoints, thresholds, seeds or an independent final result.

R15 has103490697 parameters. Mean latency121.077825ms, P95 latency
138.042440ms. Formula-checked component MACs weighted by all actual traces
give103.429621GMAC per tracked frame,103.365215GMAC per all frames including
initialization. These cover convolution/dense/attention and exclude
normalization, nonlinearities, crop/ROI, decoding, memory and bookkeeping.
Candidate generation alone contributes4115991176478720MAC across4284 calls.

## Observations And Limits

M2 selected a candidate on4031/4284 invocations (94.0943%);253 chose
no-match. This is replacement frequency, not matching accuracy. A complete
head TRAIN run and79.623188% component-holdout positive top1 did not
establish beneficial closed-loop tracking. Upstream E0 saw that holdout.

Source inspection of dual_rgbt/e0_m2_online.py shows invocation every25
valid frames, independent of M1 failure state. Each invocation computes24
candidate primary forwards from initial templates and an anchor. A positive
argmax replaces base.state and may update templates with the replacement
box using the original E0 route confidence. M1 is absent from this variant.
There is no temporal candidate identity association in this wrapper.

Cadence qualification from subsequent source inspection: full TEST samples
M2 at1,26,51,..., all congruent to1 modulo5. Therefore the replacement-side
`frame % 5 == 0` template-write branch is never executed in these R15 full
runs. It must not be presented as an observed cause of their regression.
The original E0 confidence is still recorded in replacement-side history.
Normal E0 template writes can occur on later divisible-by5 frames after the
tracking state has changed; their effect has not been causally measured.

These mechanisms are candidate explanations for regression, not proven
causes or grounds to fit a new threshold to this TEST result. Keep the two
remaining R15 benchmark runs unchanged, preserve their evidence and score
each once after native completion.

## TRAIN Diagnostics Before A New Policy

Use existing TRAIN collection and sequence-disjoint component partitions.
Report positive-candidate availability and selected-candidate correctness
separately for reliable, uncertain and lost E0 states. Measure false
replacement of reliable targets versus successful recovery of lost targets;
overall top1 hides this distinction. Frozen-cache labels alone cannot prove
closed-loop recovery after state divergence; any changed policy needs its
own reviewed TRAIN execution and fixed acceptance criteria.

The open-source KeepTrack reference propagates candidate identities across
successive frames and maintains a CandidateCollection, unlike the current
independent anchor-to-set argmax. Primary reference inspected2026-11-01:
https://github.com/visionml/pytracking/blob/master/pytracking/tracker/keep_track/keep_track.py
and official repository https://github.com/visionml/pytracking .
This is a design reference, not a reproduction or novelty claim. The
protected route remains E0 -> M1 failure -> M2 identity -> M3 memory;
do not label the current E0+M2 result as that complete architecture.

## Evidence

- Score reports/e0_m2_online_scores_r15_20261101/rgbt210.json SHA256
  8fb87bd7cf3e1e33fd422bdbff0ad530fdb73dfa5942d4e64c80900c7d623c20.
- Compute rgbt210_compute.json SHA256
  71965d9ec4c660f71933aaa04a9c92d993ce5111d0eebf84a33a7f0b2a9e19c4.
- Native status SHA256
  d76eae793065e5c73b76183fe595c61a92ab60935c266cb97361840a3974fc02.
- COMPLETE receipt SHA256
  6420388b22ad3225feda7e69f70cf899bfc775f19f7f6c5690c978d571da6245.
- Plan SHA256
  079d269c03c8f407ebe9220de4953fbf75c4e51f4b64845727655d8be20bfff9.
- Curves, score log, compute and runtime config/plan/native/COMPLETE files
  archived in reports/e0_m2_online_scores_r15_20261101; full traces and
  predictions remain in the original host26 run.
- The first completion probe stopped before scorer launch because the
  score directory did not exist. Creating that directory allowed the sole
  actual scorer invocation to succeed; no inference or scorer was repeated.
