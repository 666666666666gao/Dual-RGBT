# Cached-Gain LasHeR Dev Failure Diagnostic

Status as of 2026-10-30 06:03 UTC. This is an interim diagnostic: epochs 6-9 have complete official LasHeR-dev scores; epoch 10 has passed its 11-frame smoke and its full-dev run is active on server25 GPU1. No test split was used.

## Comparable full-dev scores

All rows use LasHeR dev, 97 sequences and 54,646 frames, evaluation cost 0.15, and 103,218,887 parameters. The protected parent-control row uses the same parent checkpoint and evaluation protocol. FPS includes image decoding and initialization; epochs 6-9 ran concurrently, so these FPS values are not isolated speed comparisons.

| Checkpoint | PR | NPR | SR | FPS | Delta vs parent PR / NPR / SR |
|---|---:|---:|---:|---:|---:|
| Protected parent control | 75.967585 | 70.762121 | 56.047161 | 11.6875 | reference |
| Epoch 6 | 74.550529 | 70.215402 | 55.324635 | 12.9545 | -1.417056 / -0.546719 / -0.722526 |
| Epoch 7 | 74.084633 | 69.380493 | 54.804861 | 12.6975 | -1.882952 / -1.381627 / -1.242299 |
| Epoch 8 | 74.557932 | 70.158057 | 55.088441 | 12.2571 | -1.409653 / -0.604063 / -0.958720 |
| Epoch 9 | 74.437970 | 70.108834 | 54.961368 | 12.6356 | -1.529615 / -0.653286 / -1.085792 |

Epoch 8 has the highest PR of these four; epoch 6 has the highest NPR and SR. Do not combine metrics from different checkpoints. No trained checkpoint beats the protected parent on any of the three metrics. Epoch 10 remains unscored, so final checkpoint selection and cleanup are not authorized.

## Observed query behavior

Actions are STOP, LOCAL_MAIN, LOCAL_ALTERNATE, GLOBAL_LOCAL and GLOBAL_WIDE (IDs 0-4). “Query delta” is final-box IoU minus the logged pre-query primary-box IoU, measured against dev ground truth only on frames where a query actually ran. It is a post-hoc immediate-frame diagnostic, not official SR, a trajectory-level causal effect, or evidence about unexecuted candidates.

| Epoch | STOP / 54,549 frames | STOP with primary IoU < 0.1 | Executed queries | Mean query-frame IoU delta | Improved / worse / unchanged | Immediate recoveries / harms |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 97.606% | 14,789 | 1,306 | +0.045865 | 429 / 271 / 606 | 32 / 15 |
| 7 | 96.722% | 15,247 | 1,788 | +0.046213 | 654 / 378 / 756 | 38 / 20 |
| 8 | 93.344% | 14,039 | 3,631 | +0.014537 | 1,190 / 1,022 / 1,419 | 65 / 42 |
| 9 | 89.263% | 14,553 | 5,857 | +0.000139 | 1,583 / 1,988 / 2,286 | 91 / 66 |

Immediate recovery means primary IoU < 0.1 and final IoU >= 0.5 on that frame. Immediate harm means primary IoU >= 0.5 and final IoU < 0.1. A STOP frame with low primary IoU is only a failed-state count; it does not establish that any candidate would have recovered it.

Epoch 9's action-family results sharpen the diagnosis: LOCAL_MAIN ran 969 times at mean delta +0.02367; LOCAL_ALTERNATE ran 2,854 times at -0.00477; GLOBAL_LOCAL ran 1,884 times at -0.00541; GLOBAL_WIDE ran 150 times at +0.01122. These are selected-query outcomes and are subject to policy selection bias. They do not justify globally lowering the STOP threshold or disabling a query family without a fixed-state counterfactual.

## Persistent sequence regressions

Comparing official per-sequence scores with the same parent-control run, 12 of 97 sequences have lower SR under every one of E6-E9. The largest repeated regressions are `belowrightwhiteboy` (SR delta about -33 to -40 points in all four epochs), `moto2north1` (about -21 to -23), and `midboyblue` (about -13 to -15). `browncar2north` is a useful positive control: its PR and SR improve in all four epochs. These identify repeatable sequences for a paired diagnostic, not the cause of their regressions.

## Evidence gaps and next diagnostic

- The current run artifacts retain the primary box and the executed final box, plus the selected predicted gain/cost. They do not retain executed outputs for every unselected candidate at the same pre-query state. Therefore the 14k-plus low-IoU STOP frames cannot be labeled missed recoveries from these logs.
- The trace `template_updated` field is blank in the scored runs. Actual template tensor writes and their causal effect on later drift are not established by this evaluation.
- Modality selection, query utility, localization, and template writes interact over a trajectory. Immediate IoU deltas cannot explain the full PR/NPR/SR regressions by themselves.
- Do not start another gain-loss or threshold sweep from these observations alone. First complete epoch 10, then run a fixed-state dev diagnostic on the three repeated failure sequences plus the positive control. Record both modality evidence, all candidate responses from cloned pre-query state, selected-vs-counterfactual recovery/harm labels, target size/scale error, and actual template-write events. Diagnostic candidate executions must not update the live track state or template.
- Any logging-only replay must prove unchanged predictions, actions, and template states against the existing run before its diagnostic labels are accepted. Keep dev diagnostic labels out of training and do not inspect test labels.
- Use public implementations as design references, not copied modules or accuracy guarantees: [KeepTrack](https://github.com/visionml/pytracking/tree/master/pytracking/tracker/keep_track) for candidate association, [ATOM](https://github.com/visionml/pytracking/tree/master/pytracking/tracker/atom) and [PrDiMP](https://github.com/visionml/pytracking/tree/master/pytracking/tracker/prdimp) for localization/state uncertainty, [MultiL2D](https://github.com/rajevv/Multi_L2D) for calibrated deferral, and [MoETrack](https://github.com/Zhangyong-Tang/MoETrack) for modality-expert comparisons. Validate each reference against the actual failure before adapting it.

## Provenance

- Parent-control checkpoint SHA256: `21e07cd1f645c6f7237e063c81cf5d60f6a14bca710e0180e8a3ca29dc827073`.
- Epoch 6-9 checkpoint SHA256 values, in order: `c31c54a88f5478298db1c52f44d686582ae43f053af72d40354d8a02ee703eba`, `397824fc92f17f01bd75623c7c0bd773db165ecd012d5c3a372b7e830ea1e142`, `86f435c1b586c4eec1515b103d4ef78c8e161604fd93c719b8e976698201a758`, `f98ae4d77220ec88d511c203e94ae54de3268c6b1d1daf6975949de88dc5e4c6`.
- Official score outputs: `reports/cached_query_gain_dev_20261030/e6.json` through `e9.json`; parent: `reports/size_tail_dev_20261030/parent_control.json`.
- Each score output binds the native-complete run, full prediction/GT/trace/evidence artifact hashes, plan, checkpoint, and official toolkit. The five retained epoch 6-10 checkpoints remain protected pending epoch 10 scoring and best-checkpoint comparison.
- Epoch 10 run: `cached_query_gain_utility_focal_e10_lasher_dev_20261030`, server25 GPU1. Smoke passed with 11 frames, 265 frozen state tensors, zero backward calls and zero optimizer updates. Full-dev official scoring is pending.
