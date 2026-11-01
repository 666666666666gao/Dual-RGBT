# E0 ep20 and M1 evidence

## Seven Completed E0 Epochs

Read-only TRAIN snapshot at2026-11-01 06:29:46 UTC:
`reports/E0_EP20_TRAIN_PROGRESS_P25_20261101T0631.json`.
Trainer897282 was directly confirmed live. Seven epochs completed and the
CSV contained17701 updates, including201 updates of epoch8. Each complete
epoch has2500 updates/20000 pairs and250 updates per modality/action route.

| Epoch | Mean TRAIN Loss | Epoch Minutes | Mean Recorded Step Seconds |
| --- | ---: | ---: | ---: |
| 1 | 2.307220 | 22.29 | 0.2742 |
| 2 | 2.258315 | 19.47 | 0.2712 |
| 3 | 2.200882 | 19.53 | 0.2701 |
| 4 | 2.217789 | 19.54 | 0.2704 |
| 5 | 2.173570 | 19.38 | 0.2713 |
| 6 | 2.189899 | 23.04 | 0.2697 |
| 7 | 2.178577 | 23.29 | 0.2711 |

Loss fell about5.58% from epoch1 to7, but epochs5-7 fluctuate. Different
seeded TRAIN pairs enter each epoch; this is not a fixed validation set.
These observations establish neither convergence nor tracking improvement.
Do not stop, change the live recipe or choose a checkpoint from this table.
The next useful TRAIN diagnostic boundary is completion of epoch10 and the
first epoch after its LR decay. Epoch11 should use executor LR1e-6 and
backbone LR1e-7; the existing second decay remains after epoch30.

Mean completed epoch time is20.93minutes. At the observed epoch8 progress,
the remaining40-epoch schedule extrapolates to about11.49hours. This is a
runtime estimate, affected by data loading and concurrent work, not convergence.
The original final-five saving rule still produces epochs36-40 only.

Epochs6-7 take about19% longer than epochs2-5 while mean recorded training
step time remains about0.27seconds. The extra wall time is outside the
recorded step interval, which excludes loader waiting and epoch setup.
Read-only host25 inspection found all four loader workers alive; two were
momentarily in `wait_on_page_bit_common`. Available RAM was202133MiB,
with no swap-in/out in the short vmstat sample; sampled CPU idle was52-58%.
This supports investigating the data/epoch overhead path, but does not prove
a single bottleneck or that M2 collection caused the change. No trainer was
paused, restarted or modified, and no new concurrency was added for occupancy.

## Actual E0 finetuning

The requested experiment is `train_e0_executor_ep20_r1_20261101` on host25
GPU1, not the preserved CADTrack-only task on GPU0. The entry calls actual
`DualRGBT.forward_action` for both modalities and five actions, trains the
backbone/decode/head, and retains the parent's preview/quality/gain state.
Exact initialization, source review and real-update admission are recorded in
`E0_EXECUTOR_EP20_FORMAL_LAUNCH_R1_20261101.json`.

At 2026-11-01 04:22:24 UTC the trainer PID897282 was alive and the flushed
log reached epoch1/global-step2350 of2500 updates per epoch. Training is still
in progress. Step logging every50 updates happens to print modality0/STOP;
it does not describe the actual balanced ten-route training distribution.
The full `steps.csv` records every route. At1801 flushed rows, counts per
action were361/360/360/360/360. Mean losses by action were approximately
1.460/2.599/2.650/2.560/2.329. Non-STOP loss includes correction supervision,
so these values must not be interpreted as comparable tracking accuracies.

No formal finetuning checkpoint is expected before epoch36 under the existing
40-epoch final-five recipe. Do not interpret its current absence as a crash.
Preserve the live trainer. No new benchmark score or convergence claim exists.
The first completed epoch took1337.394 seconds (22m17s). At04:31:44 UTC
epoch2/global-step3600 was logged and trainer897282 was alive. Forty epochs
would take about14h52m at the first-epoch rate; this is a runtime estimate.

The frozen policy was fitted to the earlier executor. Updating its backbone
and localization heads does not establish that its gain estimates remain
calibrated. The next evidence must come from complete causal tracking runs
of the actual retained E0 checkpoints, including routes and service timing.
Do not substitute direct CADTrack scores or the ten-update TRAIN smoke model.

## Evaluation implementation findings

The original E0 benchmark runner binds historical training and dev selection,
so its contracts cannot simply be reused for these new checkpoints. The two
separate entries `eval_e0_finetuned.py` and `score_e0_finetuned.py` use the
existing E0 model/tracker and unchanged official metric functions.

Read-only host25 checks covered every image path: LasHeR245 sequences,
220703 frames; RGBT210210 sequences,104706 frames; RGBT234234 sequences,
116650 frames. All884118 paired image paths exist. All manifest annotations
equal the original toolkit ground truth;211/469 annotation hashes passed
for RGBT210/RGBT234. These checks are not image-decode or benchmark evidence.

Two source issues were identified and corrected before deployment:

- Annotation hash keys contain Windows backslashes. Normalize path separators
  before joining on Linux, following the existing scorer.
- Initial receipt lists omitted `patch_embed.py` and scoring helper versions.
  Host25's `joint_candidate_results.py` and `joint_evaluation_contract.py`
  have different complete-file hashes from the local checkout. The called
  `official_metrics` AST is equal, according to the reviewer; this does not
  authorize replacing either server file. Both new entries now check the
  complete target source map from a hash-bound specific review before use,
  and scoring records the actual metric helper hashes.

Source acceptance and real11-frame TRAIN inference are separate steps.
The preflight alone does not admit a GPU evaluation or establish runtime
success. Full test results remain exploratory and ineligible for independent
final validation. Test labels must not train the model or select thresholds.

### Source binding and TRAIN inference completed

The final supplemental engineering report is SOURCE_PASS/0, SHA
`bbcd785dc68a12529e7690dfa4b901abd03862a5b3ad15d49307f6b6a80734fd`.
It binds49 sources and remains UNATTESTED/same-family/provisional, with
scientific acceptance false. The reviewer verified47 existing host25 files
and rejected108 simulated source/report mismatches.

Native task `smoke_e0_finetuned_eval_r1_20261101` on25 GPU2 completed with
exit0, one TRAIN sequence/11 frames/10 trace rows. Predictions, saved GT and
timing coverage were verified. All49 source hashes matched, model state
was unchanged, and no backward/optimizer update ran. Strict E0 loading and
103374215 total parameters are runtime evidence. The ten-update training
smoke checkpoint was used only as a loading witness; this is not a benchmark.

The reviewer finalized report metadata after the inference task had launched.
The admitted report SHA is
`eff8c79d50d8610bf44233c20e043dea76b1cc0abd3a15d84547ff9b1b0c4f8c`,
archived as `E0_FINETUNED_EVALUATION_ADMITTED_REVIEW_R1_20261101.json`.
Its49-source map exactly equals the final report. The remote original report
path remains at those admitted bytes so the smoke's original command/config
binding stays valid. The final report is separately deployed at
`reports/E0_FINETUNED_EVALUATION_FINAL_SOURCE_REVIEW_20261101.json`;
future full evaluation commands should bind that distinct path and final SHA.
Do not overwrite the admitted report or rewrite completed run/config evidence.

Runtime binding and artifact hashes are in
`E0_FINETUNED_EVALUATION_SMOKE_LAUNCH_R1_20261101.json` (SMOKE_COMPLETE).
Full official scoring and actual-model MACs remain pending. The code prepares
complete245/210/234-sequence evaluations; it does not itself schedule them.

## M1 training evidence

Native host26 task `train_protected_e0_offline_m1_r5_20261101` completed with
exit0. The actual `runs/protected_e0_offline_m1_20261031/COMPLETE.json` and
all9 recorded artifact hashes were rechecked. This was12 epochs/216 updates
of the standalone117507-parameter M1 head, not online E0 integration.

There are71843 fit states (reliable48570, uncertain5749, lost17524) and19248
component-holdout states (13966,1264,4018). The best component-holdout macro
recall is0.531125 at epoch5; lost recall is0.423843 and reliable interrupted
fraction0.131892. At epoch12, fit macro recall rises to0.789976, but holdout
macro recall falls to0.471850, lost recall to0.343206 and holdout cross-entropy
rises to0.797714 from0.590431 at epoch1.

This is observed overfit and weak held-out failure recognition. The uncertain
class is also a minority of the fit states. These measurements motivate a
TRAIN-only diagnosis before turning the head into a memory/update gate;
they do not establish which intervention will improve tracking.
The component holdout is not independent: upstream E0 saw those sequences.
No LasHeR/RGBT210/RGBT234 improvement, safe online gating threshold or
M2 identity/recovery behavior is proved by these offline numbers.
