# Relative M2 Full TRAIN Result

The native task `train_e0_relative_full_r1_p26_20261102` completed with exit0:
12 epochs, 348 optimizer updates, 116482 head parameters. The receipt reports
50.284274 seconds inside the training engine; native runtime also includes
collection verification and admission. This is cached-head training, not a
new end-to-end E0 fine-tune.

Canonical terminal verification completed once and was archived in
`E0_M2_RELATIVE_FULL_TRAIN_ACCEPTANCE_R1_20261102.json`. All retained epochs
8-12, latest and frozen initialization were replayed on CPU. Decision metrics
matched; early deleted epochs have structural history verification only.

No epoch qualified. The receipt and verifier both require retaining protected
E0, with `best_qualifying_epoch=null` and `online_allowed=false`. Source review
and successful execution do not establish useful tracking behavior.

| TRAIN component holdout | Frozen initialization | Epoch 12 |
|---|---:|---:|
| Reliable states crossing below IoU .5, out of 2899 |306|419|
| Lost states recovered to IoU >= .5, out of 800 |197|169|
| Reliable sequence-macro IoU delta vs KEEP |-.0827463911|-.0852414463|
| Loss |2.2056226244|2.2035945634|

Fit loss fell from 2.1598097108 at initialization to 1.4597881877 at epoch12;
the holdout behavior above did not follow that improvement. Every epoch had
332-419 harmful reliable-state crossings. This supports a training/decision
mismatch or generalization problem; it does not identify a causal fix.

The fixed qualification gates remain zero harmful reliable crossings,
nonnegative reliable sequence-macro delta, and more than 179/800 lost
recoveries. No threshold, seed or checkpoint is selected using TEST.

Keep latest and epochs8-12 with their native/config/history/receipt evidence.
Do not deploy the paired runtime using these rejected heads. The authorized
existing review chat was asked to diagnose the actual failure and propose a
minimal next TRAIN experiment. No new training recipe has been approved by
that request alone. User authorization includes all four P25 GPUs; three
remain occupied by fixed E0 LasHeR evaluations, with GPU2 available.
