# Size-tail experiment plan

The current native-complete parent has LasHeR-dev PR/NPR/SR
75.9675846229/70.7621205262/56.0471607387. Targets are75.8/71.9/60.2.
Protected historical E1 is76.5577172394/71.3659982337/56.4331437120.
Compare new results against both. No new adapted full-dev score is available.

Only the existing final size convolution is trained:66 parameters out of
103,218,887 total. Backbone, decoder, score/offset, preview, quality, gain and BN
statistics remain frozen. Existing localization loss and train882 GT-jitter pairs
are used; this is not causal rollout training. Dev97 selects/stops; test245 is not
used for tuning. The historical17M-parameter readout experiment lost14.7686 SR
points despite crop-loss improvements. Preserve that negative result.
This experiment is not evidence of three validated novel modules.

## Verified and running

- B8/B64/B384/B576 four-update probes passed finite-gradient, two-tensor-change,
  same-input score/offset and strict-reload checks with native exit0.
- B576/LR7.2e-5 measured allocated15.01GiB/reserved18.77GiB and409.65seconds
  for2304pairs/four updates. It fits alongside the existing approximately2.3GiB
  recovery eval. This is resource admission, not convergence or accuracy evidence.
- Parent equality and B384/B57611-frame tracking probes passed. Full97 parent,
  B384 and B576 dev runs remain active on25 GPU1/2/0 with CPU scoring watchers.
- Resumed B8 epochs restore optimizer states exactly and advance steps4 to8.
  All265 final network tensors/losses match uninterrupted training. Final moment
  differences up to2.235e-8 also have a same-order counterpart in repeated
  uninterrupted training. The audit does not claim exact final optimizer equality.
- Six Linux synthetic workflow tests pass, including failure preservation,
  patience/selection/cleanup and immediate rejection after severe dev regression.

## Formal pilot remains unadmitted

Reviewable draft:reports/SIZE_TAIL_EPOCH_PILOT_DRAFT_20261030.json.
`formal_launch_allowed=false`; completed-dev report hashes remain unset.
Review parent equivalence and both adapted diagnostics before enabling it.
No formal pilot controller has been launched.

Proposed configuration: B576,LR7.2e-5,AdamW weight decay1e-4,seed2027,
9216sampled pairs per epoch/16updates, maximum10epochs. A sampled epoch is not
an exhaustive pass through all video frames. Each epoch is a separate native
task that restores the previous network/AdamW state, then completes full97dev
and official scoring before another epoch starts.

Normal convergence monitoring observes at least5epochs and stops after3epochs
without SR improvement greater than0.1pp over the last progress anchor. Train
loss is not the selection or stopping metric. The declared exception is a dev
SR drop greater than5pp below the parent: reject immediately and preserve the
forensic checkpoint, responding to the observed historical readout collapse.
The5pp threshold is an experiment-budget decision, not a learned threshold.

After each completed dev evaluation, retain the latest5checkpoints. An in-flight
epoch may temporarily add one. At terminal selection rank the latest5 by SR,
then PR, then NPR. Keep the winner and remove only the other verified owned
checkpoints. Failed native tasks cannot trigger selection or final cleanup.
All scores and rejected trials remain documented. DEV_SELECTED is test-ineligible.

Five dev evaluations cost roughly5.5-9.5GPU-hours at historical8-14FPS; concurrent
load adds uncertainty. Formal training time and convergence are unmeasured.
Keep the eight recovery GPU slots active and avoid duplicate tasks.

## References and interpretation

ATOM (https://arxiv.org/abs/1811.07628) and open-source pytracking
(https://github.com/visionml/pytracking) motivate overlap/localization refinement;
no ATOM module was transplanted. KeepTrack association is a possible later
response to identity/drift failures, not an implemented claim.

The recovery gainpair cost sweep favors0.05; increasing to0.15 loses1.5002SR
points, with46sequences improving and48worsening. A penalty is not universally
transferable across models. Recovery training status remains FAILED and its
scores remain diagnostic. See docs/SIZE_TAIL_EPOCH_TRAINING_20261030.md.

## Static-template causal diagnostic (2026-10-30 01:17 UTC)

The verified parent has1215of7608 reconstructed template writes atIoU<0.1,
including1214STOP writes. Freeze both initialized templates while retaining
the same parent weights,cost0.15,K8,initialization and full97dev protocol.
This removes both helpful and harmful updates; a positive result motivates
selective memory reliability, while a negative result rejects global freezing.
It does not identify the counterfactual benefit of each individual write.

The31-frame native smoke passed with6actual mutations reverted and265frozen
state tensors. Full run static_template_dev_20261030 is active on25GPU3,
wrapper1155267,CPUscorer1155571. No new training or test is admitted by this
ablation. Compare officialPR/NPR/SR with grid parent and protectedE1 after
97sequences/54646frames and native COMPLETE/exit0. Do not rank smokeFPS.
See docs/TEMPLATE_FAILURE_HYPOTHESIS_20261030.md for evidence and open-source
OSTrack/STARK references. Existing formal-pilot admission gates remain intact.

## Query recovery supervision evidence (2026-10-30 01:20 UTC)

The audited182182historical chosen-modality training samples include1393
lost-to-good recoveries,1069onSTOP decisions. But global local/wide sampled
queries have negative average gain(-0.4273/-0.3651)and large harmful counts.
Do not respond with unconditional extra queries. After the current full-dev
controls, collect frozen-parent causal train-only features and labels to compare
recovery/harm supervision with expected gain under fixed candidate budgets.
Historical CSVs contain outcomes, not the955input features needed for refitting.
Measure candidate availability and false recoveries; require full closed-loop
dev improvements. See docs/QUERY_RECOVERY_DIAGNOSTIC_20261030.md. No formal
launch is admitted by these historical training-state counts.
