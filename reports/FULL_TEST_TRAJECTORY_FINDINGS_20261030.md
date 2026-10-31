# Exploratory full-test trajectories: 2026-10-30 local

All runs below cover LasHeR test: 245 sequences / 220703 frames. They remain
`exploratory_test=true`, `test_eligible=false`. These repeated test observations
are development evidence, not an independent final test.

## Raw Official Results

| Family / provider | Epoch | PR | NPR | SR | Concurrent FPS |
|---|---:|---:|---:|---:|---:|
| Parent-cache all882 / 26 | 4 | 66.717686 | 61.824186 | 50.555620 | 8.563222 |
| Parent-cache all882 / 26 | 5 | 67.405901 | 62.580736 | 50.966963 | 8.716696 |
| Parent-cache all882 / 26 | 6 | 67.498396 | 62.600194 | 51.001246 | 8.703498 |
| Parent-cache all882 / 26 | 7 | 67.543368 | 62.650040 | 51.093086 | 8.615936 |
| Parent loss / 26 | 5 | 65.784692 | 60.841643 | 49.740533 | 8.771042 |
| Parent loss / 25 | 6 | 66.889509 | 61.814002 | 50.441545 | 12.060859 |
| Parent loss / 25 | 7 | 65.628468 | 61.024538 | 49.703236 | 12.058860 |
| Parent loss / 25 | 8 | 65.655783 | 61.279735 | 49.764823 | 12.031755 |
| Parent loss / 25 | 9 | 65.049282 | 60.797220 | 49.296764 | 11.902400 |
| Refreshed-cache all882 / 25 | 8 | 66.185390 | 61.309848 | 50.274906 | 11.731192 |
| Refreshed-cache all882 / 25 | 9 | 67.134699 | 62.074075 | 50.630373 | 11.693804 |
| Refreshed-cache all882 / 25 | 10 | 67.356374 | 62.286899 | 50.928631 | 11.793080 |
| Refreshed-cache all882 / 25 | 11 | 66.969336 | 61.998446 | 50.640487 | 11.704709 |

All have 103218887 parameters. FPS includes image decoding and initialization,
excludes model loading, and measures service time under concurrent workloads.
It does not support a paired speedup comparison between providers.

The best observed completed tuple is all882 E7, still below the requested
75.8 / 71.9 / 60.2 by 8.256632 / 9.249960 / 9.106914 percentage points.
This is a descriptive observation, not promotion or final checkpoint selection.
E8 and the other active families remain incomplete.

Within parent-cache all882, E4 to E7 SR changes by +0.537466 points (+1.0631%
relative to E4). Parent-loss E6 to E9 changes by -1.144781 points (-2.2695%).
Refreshed E8 to E10 changes by +0.653726 points (+1.3003%), then E11 falls
0.288144 points from E10. These are single-seed, correlated epoch observations;
they provide neither uncertainty estimates nor evidence of a new module benefit.

Official inputs are in:

- `reports/cached_parent_gain_all882_test_20261030/e004.json` through `e007.json`
- `reports/cached_query_gain_parent_loss_test_p26_20261030/e005.json`
- `reports/cached_query_gain_parent_loss_test_p25_20261030/e006.json` through `e009.json`
- `reports/refreshed_query_gain_all882_test_p25_20261030/e008.json` through `e011.json`

The current local score archive search did not identify a scored full-test
control for parent checkpoint `21e07cd1...` at cost .15 / K8. Locate or run that
exact control before attributing a full-test delta to a head overlay. The historical
reference-E0 result (72.524939 / 69.068894 / 57.732435) uses checkpoint
`ee60704e...`, cost .05 / K0 and a different path, with historical training-source
audit warnings. Its SR is 6.639349 points above E7, but it is context, not a
matched treatment control. Do not erase its warnings or remeasure the user's
provided baseline as though it were a new experiment.

## Verified Trajectory Diagnostics

The read-only CPU analyzer binds each scored report to 982 run artifacts:
config, completion, and all 245 prediction / GT / trace / evidence sets.
It also checks the native COMPLETE/exit0, exact command, plan, checkpoint,
curve and tracker source hashes. Each run has 219925 valid tracked frames;
245 initialization frames and 533 tracked frames with invalid GT are excluded.

| Frame-weighted diagnostic | All882 E7 / 26 | Refreshed E10 / 25 | Parent-loss E6 / 25 |
|---|---:|---:|---:|
| IoU < .5 | 98793 | 97630 | 98651 |
| IoU < .5 and center error > 20 px | 88256 | 86327 | 86972 |
| Percentage of low-overlap frames with center error > 20 px | 89.334% | 88.423% | 88.161% |
| IoU < .5, center <= 20 px, area ratio outside [.5, 2] | 7127 | 7317 | 7520 |
| Lost frames, IoU < .1 | 68019 | 66196 | 65834 |
| Lost frames with confidence >= .65 | 44216 | 44590 | 43085 |
| Template-write frames reconstructed from source rule | 35208 | 35758 | 35278 |
| Reconstructed writes with IoU < .1 | 8666 | 8739 | 8506 |
| Executed queries | 14573 | 12330 | 3324 |
| Helpful / harmful / neutral executed queries | 5515 / 4565 / 4493 | 5532 / 4184 / 2614 | 1493 / 1116 / 715 |
| Mean immediate executed-query IoU gain | .031680 | .034140 | .047862 |
| Predicted gain minus realized immediate IoU gain | .070064 | .079964 | .041570 |
| Queries with negative realized gain-minus-cost | 11715 | 9768 | 2413 |
| Immediate recoveries / immediate failures | 175 / 73 | 178 / 70 | 60 / 12 |

The immutable tracker source has SHA256
`89e18853f60eac309e47206005450164931b68b3934ae36a40034cd3b3ce03ed`.
Its trace write flags are empty. Write frames are reconstructed from the exact
rule `frame % 5 == 0 and confidence >= .65`, not observed template tensor diffs.
A frame may update one or two modality slices; the table counts frames.
IoU/center errors here are continuous geometry, not official toolkit PR/NPR/SR.
Immediate recovery means primary IoU < .1 and final IoU >= .5; immediate failure
means primary IoU >= .5 and final IoU < .1.

## Findings And Consequences

1. Center displacement accompanies 88-89% of low-overlap frames across all three
   trajectories. The old small dev fixed-state sample emphasizing size mismatch
   does not explain these full-test trajectories. Complete the nine already
   running size-tail tests, but do not assume size-tail tuning will solve drift.
   Absolute 20-pixel error is size dependent and does not establish a cause.
2. About 65-67% of lost frames still have confidence >= .65, and roughly 24-25%
   of reconstructed write frames occur while lost. Score maximum alone is a weak
   reliability signal on these visited states. Test a paired memory-update
   intervention before claiming template contamination causes future drift.
3. Executed queries have positive mean immediate gain, yet predicted gains are
   optimistic and 73-80% have negative realized utility at the unchanged cost .15.
   Stronger mean immediate gain with fewer queries does not imply stronger
   sequence tracking. A low-IoU STOP does not prove any unexecuted query could
   recover it. Do not launch another loss/threshold sweep from these counts.
4. No target is reached. Single-seed epoch comparisons and historical baselines
   do not validate the requested three innovations. Train collection and held
   labels must remain on TRAIN; none of this test GT becomes a training label.

## Grounded Next Experiments

Preserve the three existing roles: cheap primary-modality preview, gain-driven
region requests, and shared-backbone local correction. The following are
hypotheses, not implemented or validated new contributions:

1. For preview reliability, collect both-modality agreement and localization
   reliability targets from TRAIN visited states. Check reliability on the
   existing held TRAIN partition, then evaluate the complete test directly.
   STARK's separate `pred_logits` reliability score provides a concrete design
   reference distinct from taking the localization score maximum.
2. For region requests, collect candidate identity/association evidence on
   TRAIN failure trajectories, using cloned pre-query state for comparisons.
   KeepTrack combines candidate matching with explicit uncertain/not-found
   states. Measure recoverable-state coverage before allocating another
   large gain-head sweep.
3. For correction memory, first compare original updates with frozen dynamic
   templates using identical checkpoint, cost, K, inputs and initialization.
   Then test a TRAIN-calibrated reliability gate, if supported by that paired
   result. KeepTrack suppresses classifier updates on uncertain/not-found
   states; STARK combines periodic template updates with its reliability head.
   Neither result proves their thresholds transfer to this model.

Pinned source references, retrieved read-only through the GitHub API/raw source:

- `visionml/pytracking` commit `7eb9e74bd3d40e29dbcec444902237da13de247b`,
  `pytracking/tracker/keep_track/keep_track.py`.
  Archived SHA256 `5983ad16308737f4f3a6a768146be371447f56b54422e65753f10caf360c6f58`.
- `researchmm/Stark` commit `162e1d37a38f92251cb380e90ccdb59608f2df3c`,
  `lib/test/tracker/stark_st.py`.
  Archived SHA256 `7d3c55fbb58056afdf86dabe9f8ad7fcd86c4cebba534436882282cb646d19ec`.

Exact reference sources are under
`reports/source_snapshots/tracking_references_20261030/`; no external code was
installed into the model. New analyzer tests passed 3/3 locally and on actual26.
The three real complete-run analyses all finished successfully.

The prior parent-dev route MAC estimate remains about 40.24 GMAC/tracking frame
under its documented operator exclusions. It is not a measured full-test MAC
result for these overlays; test-route weighting remains outstanding.
