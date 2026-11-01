# M1 RGBT210 Full Paired Result

Both native RGBT210 tasks completed with exit0 on2026-11-01. Each covers
210 sequences and104706 frames. The unchanged official toolkit scored each
run once after full completion. No partial scoring or TEST-driven adjustment
was used for this comparison.

| Configuration | PR | SR | FPS | Parameters | Mean Tracked-Frame GMAC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Protected E0 control | 88.828508 | 66.124608 | 11.134969 | 103374215 | 64.023388 |
| Protected E0 + M1 guard | 89.516426 | 66.612586 | 11.137673 | 103491722 | 64.034768 |
| Guard minus control | +0.687918 | +0.487978 | | +117507 | +0.011380 |

The guard result exceeds the requested RGBT210 PR88.7/SR62.9 thresholds
descriptively. It is an exploratory full TEST result, with
`exploratory_test=true`, `test_eligible=false`; it is not independent final
validation or evidence that the other benchmarks or full three-module goal
are complete. Both runs use fixed original E0 checkpoint
`ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c`
and fixed epoch5 M1 head
`658c84275ad1e1624d4b453fee89dd312273633664024bf0b42ae78dcc189ebf`.
This is not the still-running CADTrack-ep20 E0 finetuning experiment.

## Guard Behavior And Limits

The guard blocked164 of16874 proposed dynamic-template writes, or0.9719%.
The control proposed16866 writes and blocked none. Proposal totals can
differ after template changes alter future trajectories. This is the observed
write-event count, not the13.1892% TRAIN component-holdout reliable-state
interruption fraction; those quantities have different denominators.

These paired full trajectories support a positive observed RGBT210 effect
of the fixed M1 guard. They do not establish a general benefit, a causal
explanation of the offline fit/held gap, seed robustness, M2 identity quality,
or M3 recovery. LasHeR and RGBT234 remain running. Keep the fixed head and
existing evaluation plans; no threshold, checkpoint, loss or architecture
was chosen from this TEST outcome.

FPS includes decode, initialization, observer/M1 and synchronized rollback,
excludes model loading, and was measured under concurrent server load.
Do not interpret the tiny FPS difference as a demonstrated speed gain.
MACs use actual complete trace route counts and the matching-source
CPU-executed arithmetic profile. Mean GMAC excludes initialization,
normalization, nonlinearities, ROI/decode and bookkeeping; it covers104496
tracked frames. The guard adds117376 head MACs per tracked frame; its
different future route counts also affect total E0 MACs.

## Evidence

Directory: `reports/e0_m1_full_scores_r1_20261101/`.
The archive contains both official score JSONs, curves, compute summaries,
and exact config/native-status/COMPLETE receipts under `runtime/`.
The scorer verified full predictions, traces, immutable source and input
bindings, complete receipt artifacts, and unchanged model state.

| Artifact | SHA256 |
| --- | --- |
| rgbt210_control.json | cb280cf315bb7b740920ebe7d4ad34b84144e93386df8158b052dd27082cd88a |
| rgbt210_guard.json | f3820e47f2561dfce3e613494eee72478bdcd962c0583415c74f41275e46c838 |
| rgbt210_control.npz | e89d9bb3b71cbd9ad9a3f0ec7986776ca91bdc085cd547bf3662de332b4cf8d0 |
| rgbt210_guard.npz | 6b77fb19a662dbe4ee6c4df5a3fc3a40277b19094b551ab890e228fcb663a68d |
| rgbt210_control_compute.json | b3e8bbe7c40075f286715da2f93295ec651e35d9cb5d1f22ad3c22bc6e602516 |
| rgbt210_guard_compute.json | 8372afa827336681f1cfff761049ea42b60dda3b1c7b314c923c3a686d0d316a |

Source review remains same-family/UNATTESTED/provisional; no mandated Astra
execution or independent scientific acceptance is claimed. Preserve these
weights and audit dependencies. Continue the already admitted other
benchmarks, E0 finetuning, and M2 full-collection acceptance to training.
