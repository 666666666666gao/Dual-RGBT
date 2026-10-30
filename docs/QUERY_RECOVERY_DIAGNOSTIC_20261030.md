# Query recovery evidence and next decision

The best server25 grid parent remains PR75.9676/NPR70.7621/SR56.0472.
No new dev accuracy is reported here. This audit reads historical training
labels; it does not train, select a threshold, or modify an active evaluator.

## Provenance and scope

`tools/audit_training_query_opportunities.py` checks native COMPLETE/exit0,
the evaluation-pinned native/config hashes, training-plan hash, parent identity,
all882sequence membership,182182contiguous update IDs, label/step joins,
unique purpose/role labels, IoU arithmetic, per-family and completion counts,
and unchanged input hashes before/after scanning. The182182updates generated
364364sampled labels and15207extra selected-query labels,379571total.
New hashes bind the labels for this audit; they are not a previously recorded
label-file attestation. Training code attests teacher queries preserved the
online tracker state and reproduced the executed selection.

Each labelled state has one sampled query per modality. K8 global candidates
are randomly sampled within the selected query kind; all candidates were not
executed. The model changes during training. This is neither final-checkpoint
dev recall nor a bound on all possible recovery actions. A counterfactual
sampled query has an immediate executed-box IoU label, not a future trajectory.
Only the chosen-modality subset describes alternatives to that step's STOP.

## Measured chosen-modality evidence

| Measure | Count |
|---|---:|
| Sampled labels on the chosen primary modality |182182|
| Gain>0.01 |38382|
| Gain<-0.01 |100849|
| PrimaryIoU<0.1 |43951|
| PrimaryIoU<0.1 and sampled queryIoU>=0.5 |1393|
| Those recoveries on steps whose policy choseSTOP |1069|
| PrimaryIoU>=0.5 and sampled queryIoU<0.1 |41469|

Recovery occurred in3.1694%of sampled lost states. The policy choseSTOP on
76.7408%of these1393sampled recoveries. This demonstrates missed *sampled
training-state* opportunities; it does not establish that76.74%of dev
recoveries are missed. Across all chosen-modality sampled labels,21.068%are
helpful and55.356%harmful by the0.01gain threshold.

| Query kind | Labels | Gain>0.01 | Gain<-0.01 | Lost-to-good recovery | Recovery whileSTOP | MeanIoU gain |
|---|---:|---:|---:|---:|---:|---:|
| LOCAL_MAIN |45546|16288|16372|93|72|0.002563|
| LOCAL_ALTERNATE |45546|15841|17341|390|325|-0.002847|
| GLOBAL_LOCAL |45545|2995|33435|336|250|-0.427326|
| GLOBAL_WIDE |45545|3258|33701|574|422|-0.365056|

Kinds2/3have substantial negative average gain alongside genuine recoveries.
An unconditional global-query increase is therefore unsupported. Round1has
724sampled recoveries/550STOP misses; round2has669/519. These changing-model
trajectory samples do not prove training convergence or final-model calibration.

## Confidence source and external references

The pinned `dual_rgbt/model.py` primary path multiplies the sigmoid center
heatmap by a Hann window. `CandidateEvidenceTracker` uses its maximum for
confidence. `EvidencePolicy.query` invokes `LocalEvidenceReader` without a
window; its final confidence is an unwindowed center-heatmap maximum.
Both feed the same0.65template-update condition. The existing query gain head
receives primary heatmap statistics, but no separate post-query reliability
head controls the update. A high peak is not a trained IoU or presence estimate.
This source difference motivates a reliability test; it does not prove that
the window or threshold causes all drift. Most reconstructed bad writes are
STOP, so query-only calibration cannot cover the principal observed writes.

[STARK-ST](https://github.com/researchmm/Stark/blob/main/lib/test/tracker/stark_st.py)
uses separate classification logits for template-update confidence and
retains the initial template. [KeepTrack](https://github.com/visionml/pytracking/blob/master/pytracking/tracker/keep_track/keep_track.py)
uses candidate matching and explicit not-found/uncertain states for tracking
and updates. Their implementations support separating reliability and identity
from a single localization peak; their thresholds or accuracy do not transfer
automatically. No external implementation has been copied into this model.

## Next experiment decisions

1. Finish the parent-equivalence, B384/B576 and static-template full97dev runs.
   Reject or admit size-tail training using the existing full-score gates.
2. If fixed templates improve tracking, test selective dynamic writes using
   target reliability, with the initial template protected. If freezing worsens,
   reject unconditional freezing; retain the observed bad-write diagnostic.
3. For query changes, first collect a frozen-parent, train-only causal replay
   with pre-query features and sampled executed IoU labels. Split by sequence,
   retain label sampling probabilities, and measure recovery precision together
   with destruction of previously good tracks. Historical labels alone lack
   the original955-dimensional gain inputs needed to refit that head.
4. Compare the existing expected-gain objective with recovery/harm supervision
   under the same candidate set and compute budget. Do not increase global
   query frequency or change cost merely because there are missed recoveries.
   A future replay must measure candidate availability before attributing failure
   solely to the gain head. Require full-dev closed-loop improvement afterwards.

These are evidence-led hypotheses within the preview, query-policy and
executor/memory responsibilities, not three validated core innovations.
No new formal experiment is admitted by this document alone.
