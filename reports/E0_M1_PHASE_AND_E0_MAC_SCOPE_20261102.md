# TRAIN Sampling Phase And E0 MAC Scope

## Fixed TRAIN Phase Description

The eight completed frozen TRAIN diagnostic sequences contain3068tracked
frames, of which3019are annotated valid. This analysis reuses their archived
traces and performs no training, inference or recollection. All eight trace
hashes, the native COMPLETE/exit0, receipt and original verified summary are
checked by tools/summarize_m1_train_phase.py. Raw results:
reports/E0_M1_FIXED_TRAIN_PHASE_DESCRIPTION_20261102.json.

M1 cache collection samples valid frames divisible by5. The diagnostic M2
cadence samples every25valid frames; these are different sampling rules.
The following groups describe all valid frames by raw frame index modulo5,
not just the sparse M2 invocation frames.

| Partition | Phase | Valid | Lost | Lost Recognized | Reliable | Reliable Interrupted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fit | 0 | 457 | 42 | 11 | 316 | 41 |
| Fit | 1 | 459 | 41 | 12 | 312 | 38 |
| Fit | 2 | 459 | 39 | 10 | 319 | 36 |
| Fit | 3 | 458 | 40 | 9 | 312 | 30 |
| Fit | 4 | 457 | 38 | 11 | 318 | 37 |
| Holdout | 0 | 143 | 18 | 4 | 106 | 2 |
| Holdout | 1 | 148 | 19 | 4 | 112 | 2 |
| Holdout | 2 | 147 | 21 | 3 | 112 | 3 |
| Holdout | 3 | 146 | 18 | 5 | 112 | 4 |
| Holdout | 4 | 145 | 19 | 5 | 112 | 2 |

Holdout phase0 lost recognition is4/18=22.2222%, versus4/19=21.0526%
at phase1; across all phases it is21/95=22.1053%. Poor recognition is
also present at the cache sampling phase. These small descriptive counts
do not support attributing the recognition failure primarily to phase
mismatch, or launching another collection solely to fix that hypothesis.
They do not prove equivalence of phases or exclude a phase effect.

These are only four fit and four component-holdout sequences. Upstream E0
saw both partitions, and local E0 proposals share the intervened tracker
history. There is no independent baseline or causal intervention on phase.
No TEST labels, tuning or checkpoint selection are involved.

The previously completed balanced-loss control still worsens reliable
interruption from13.189174% to30.853501% on the full component holdout.
This analysis does not overturn its rejection for online promotion.
Keep the original head and all currently running inference unchanged.

## E0 MAC Coverage

Source inspection compared dual_rgbt/model.py forward_action and predict,
and the calling path in dual_rgbt/tracker.py. Both model entry points call
preview_state, primary, candidates and gain_predictions exactly once,
plus correct once when the chosen action is nonzero. preview_state
contains the preview convolutions and quality MLP; gain_predictions
contains the gain MLP. Therefore the existing executed forward_action
profile already covers these counted convolution/dense/attention calls.
It does not omit a separate policy forward outside that entry point.

The online windowed path additionally calls cal_bbox; inspection of
lib/models/layers/head.py shows reductions, indexing and coordinate
arithmetic, not additional convolution, dense or attention matrix products.
This is a source-path audit, not another measured end-to-end profile.

All three inspected files exactly match source_sha256 in
reports/E0_EP20_E040_P26_ROUTE_MACS_20261102.json:

- dual_rgbt/model.py:
  c7fae6f893797edd10dcb6e625bb304b5a637c80ed1ac404f58d3d223d1b9f7e.
- dual_rgbt/tracker.py:
  406927019ffdcdbc8546755d9f7bff0859aac20766116e4a70f8c89132694fff.
- lib/models/layers/head.py:
  958ee314ae200dd52355050dc1da8b46d5268da032e5fc1d0c6af99bab4c7431.

Retain the stated arithmetic scope: no initialization, nonlinearities,
normalization, crop/ROI, decoding, memory or bookkeeping costs. Full-run
trace route counts are still required for dataset-weighted MACs, and
complete synchronized timings for FPS. No new benchmark score is claimed.
