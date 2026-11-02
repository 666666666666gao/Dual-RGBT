# Relative Current-E0 Head TRAIN Recipe Draft

Status: predeclared recipe, not source, update or full-training admission.
This document supplements the existing implementation contract without
relaxing its TRAIN protection/recovery acceptance.

Use only the original706fit sequences for updates and176component-held
TRAIN sequences for engineering selection. Full supplement acceptance
must establish all882sequence/frame joins before full training. E0 saw
these sequences upstream; component holdout is not independent final
validation. No TEST annotations, scores or checkpoint ranking enter
this recipe.

Initialize `E0RelativeIdentity` from the original TRAIN-selected epoch9
identity head, checkpoint SHA256
`7e7df5d25d3e8667fe38a4ca8f6ec3662308ffcb52a09bd16aecaab59a9cb4be`.
Strictly preserve its original state dictionary. Keep the unused no_match
scalar frozen. Train only the existing projection/matching network;
no E0/M1/M3 parameters, topology or protected tracker state are updated.
This retains the existing matching architecture while adding explicit
current E0 as the KEEP alternative.

Fixed recipe: seed2027,12epochs,512sets per batch, AdamW learning rate
0.0003, weight decay0.0001, no scheduler, TF32 disabled. This matches the
original head recipe to limit confounding. Observe every epoch's loss,
fit/held protection and recovery metrics. A small standalone head does
not justify dummy allocations or duplicated samples to occupy GPU memory.
Record actual peak allocated/reserved memory and elapsed time.

On each fit state, compute common float64 E0/candidate IoUs against the
TRAIN annotation. Candidate supervision is available only for original
known labels and fully observed evidence. Unknown original labels remain
excluded from both relative target choice and the loss denominator.
Choose the highest-IoU known candidate with IoU>=0.5 and strictly greater
IoU than current E0; exact candidate-IoU ties use original candidate order.
Otherwise target KEEP. If current E0 has no fully covered modality,
exclude the state from loss and retain the scorer's KEEP-only execution.
Use cross entropy over KEEP and known candidates, with unavailable
candidate evidence masked. Exact inference score ties choose KEEP.

Before any full update, execute a separate reviewed real five-update smoke
on the six accepted93-frame TRAIN-prefix sets. Require finite loss and
gradients, changed matching parameters, unchanged frozen no_match, strict
save/reload output equality, native COMPLETE/exit0 and bound receipts.
Do not claim this six-set check establishes fit quality or full acceptance.

Record the frozen explicit-E0 head's fit/held metrics before epoch1 as
the no-update control. At every epoch, report reliable-to-below0.5 harm,
sequence-macro reliable IoU change, lost recovery, sequence-macro overall
IoU change, KEEP/replacement counts, unknown-candidate selections and
per-state population counts. Evaluate inference choices using actual
coverage, rather than hiding ambiguous candidates at inference.

An epoch qualifies only if the exact component-held baseline populations
remain2,899reliable and800lost, reliable-to-below0.5 harm is zero,
sequence-macro reliable IoU change is nonnegative, and lost recoveries
exceed179. Rank qualifying epochs by sequence-macro overall IoU change,
then lost recoveries, then earliest epoch. If no epoch qualifies,
retain protected E0 and report that the learned decision failed acceptance.
Never lower the gates after observing training.

Preserve latest and lastfive epoch files, plus the best qualifying TRAIN
checkpoint when one exists. Do not write a misleading best.pth when none
qualifies. Do not delete initialization, smoke or audit dependencies.
Closed-loop M1-gated integration requires subsequent independent source
review and real paired parity/update evidence; offline acceptance alone
does not authorize online deployment or establish benchmark gains.
