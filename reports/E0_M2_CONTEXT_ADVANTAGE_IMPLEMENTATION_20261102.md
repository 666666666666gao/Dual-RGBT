# Context Advantage: Local TRAIN Preparation

Actual rejected-head attribution is saved in
`E0_M2_RELATIVE_FULL_TRAIN_ATTRIBUTION_20261102.json`. The diagnostic performed
zero updates and initialized no CUDA context. It reproduced epoch12 decision
counts against the canonical saved terminal acceptance.

| Accepted TRAIN partition | Fit | Component holdout |
|---|---:|---:|
| Harmful reliable replacements |500|419|
| Harmful replacements choosing original unknown |326|108|
| Harmful replacements choosing known negative |174|311|
| Lost recovery |1545|169|
| Available-candidate lost recovery oracle upper bound |1810|374|
| Harm with GT-dependent unknown masking |181|322|

The counterfactual mask uses TRAIN GT, so it cannot be an online rule.
It also leaves substantial harm. The oracle counts require an observed
current proposal and at least one observed candidate with IoU >= .5.
They establish candidate availability on these fixed states, not achievable
online recovery or benchmark performance.

The existing authorized review chat's diagnosis is
`E0_M2_RELATIVE_FULL_FAILURE_TRAIN_DIAGNOSIS_20261102.md`. Its proposed bounded
experiment is now being implemented without modifying any sealed producer,
old head, old loss, or active evaluation source.

New local files:

- `dual_rgbt/e0_context_advantage.py`: 28 causal inputs, 28->32->1 GELU head,
  961 trainable parameters. Zero-initialized output; KEEP logit exactly zero;
  unavailable observations cannot be selected.
- `tools/context_e0_train_data.py`: uses canonical accepted cache verification
  and exact original sequence order, freezes the original epoch9 appearance
  head, and joins existing candidate/current/previous boxes, detector scores,
  routes and the 8-D observation input history at offsets128:136.
- `tests/test_e0_context_advantage.py`: six CPU behavioral checks covering
  KEEP/ties/missing evidence, geometry and history, numeric dtype, fixed
  sequence weights, fit normalization, and utility/protection gradients.

Inputs are appearance scores, detector score, coverage, current/candidate
motion relative to previous box, previous input history, current route, and
predicted-box overlap. GT IoUs, identity labels and sequence IDs never enter
the network forward. Unobserved -inf appearance scores are encoded as zero
only in feature rows excluded by the identical observation mask.

Normalization is fitted once on observed fit rows. Coverage, modality,
action and the two history modality indicators are not standardized.
Observed candidates with original identity label -1 retain that label;
their executed boxes supply a separate TRAIN localization-utility target.
This does not relabel them as negative identities.

The proposed loss is SmoothL1(deltaIoU, beta=.1) plus softplus penalty on
predicted advantages for reliable states' degrading candidates, coefficient1.
Fixed full-fit sequence weights avoid favoring long sequences and avoid
renormalizing each minibatch. Prediction has a fixed zero KEEP boundary;
no held/TEST threshold calibration is proposed.

No training entry, launch plan, source approval, GPU update, or online runtime
is supplied by these files yet. Next: review these concrete bytes, finish
the bound CPU tensor export and training entry, then perform real update
smoke and one fixed TRAIN run after exact source/input admission. Prefer
P25 GPU2 using an isolated snapshot so active P25 evaluation sources remain
unchanged. Original split706/176 and qualification gates remain unchanged.
No qualifying head means retain E0; no rejected-head paired/benchmark run.
