# Selective Prediction Reference For M2 Failure

Actual source inspected:
[geifmany/selectivenet](https://github.com/geifmany/selectivenet), commit
`a6d0a8fd33dae61da910b61a2aae93102d2d4869` (ICML2019 SelectiveNet).
File hashes are recorded in
`E0_M2_SELECTIVE_RISK_PRIMARY_SOURCE_20261102.json`.

`models/cifar10_vgg_selectivenet.py:143-149` constructs classification,
selection and auxiliary outputs. Its `selective_loss` at lines217-222
combines selected-example classification loss with a squared coverage
shortfall penalty. `selectivnet_utils.py:23-38` evaluates risk and coverage
on accepted examples and offers optional percentile calibration.

The relevant lesson is that accepting fewer actions alone is insufficient:
one must also measure accepted-action risk and useful coverage. Our original
M2 gate already requires both zero reliable-target harm and substantial lost
recovery. All-KEEP cannot satisfy that recovery gate. The rejected relative
head shows why lower fit CE alone is insufficient.

This reference does not provide a zero-harm guarantee or solve tracking
state shift. Its optional calibration, numerical thresholds, loss weights,
architecture and TEST-score handling are not copied. Our proposed context
advantage head instead regresses executed-candidate utility relative to KEEP
on TRAIN and explicitly penalizes reliable-state degradation.

Selective rejection itself is established prior work, not our core novelty.
Whether causal context plus utility supervision yields a useful M2 remains
unproven until actual TRAIN qualification and paired closed-loop acceptance.
