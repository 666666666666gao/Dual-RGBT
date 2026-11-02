# One Recovery Objective Experiment

The actual Context R2 run completed all12 epochs but failed every qualifying
gate. Its exact acceptance and TRAIN recovery-gap audit are preserved.
The independent supplemental diagnosis is
`E0_M2_CONTEXT_FULL_TRAIN_DIAGNOSIS_R2_20261102.md`.

This candidate adds exactly one term to the unchanged original loss:

```text
R_ij = observed_ij and current_iou_i < .1 and candidate_iou_ij >= .5
w_i = sequence_weights(fit_sequence_index, any_j(R_ij))
L_rec = mean_i(w_i * sum_j(R_ij * SmoothL1(score_ij, delta_ij, beta=.1))
               / max(1, sum_j(R_ij)))
L_new = L_adv_existing + L_protection_existing + L_rec
```

The coefficient is fixed at1. Recovery weights use only the full fit partition,
remain fixed under minibatching, and do not enter the forward function.
No held/TEST group is used to compute them. Original advantage and protection
weights, appearance scorer, context features, normalization, zero-KEEP policy,
706/176 split, seed2027, batch512, AdamW/LR.0003/WD.0001 and12epochs348updates
remain fixed. The fit-recovery mean prediction and state-level recovery gap,
then all original held gates, are the prespecified diagnostics.
This is a hypothesis test, not a proven remedy.

Six additive source/test files implement the objective, training, exact source/
data/native admission, retained CPU replay and five CPU behavior checks.
Original163 source files and prior sealed snapshots remain unchanged.
The two disabled drafts bind169 sources:

| Draft | SHA256 |
|---|---|
| smoke | eb8f7c5e83db38e342338f6e024f3651661178248d26fbc19441ba187a97835d |
| full | 9fe6ec70145c526bb643f2ccf924940d0274952f0a499220e0a14e8fd5fda3e9 |

Target is P25GPU2,
`/home/gaob/Dual-RGBT/snapshots/context_recovery_r1_20261102`.
The exact previous dataset/export receipt/base full-plan are reused, with
their original hashes and163-source inheritance checked separately. No new
collection or export is required. The new snapshot receives old sources from
the actual prior P25 snapshot, preserving the three unrelated local lib edits.

Five new CPU tests passed, including actual5 optimizer updates/reload.
They do not establish real GPU or source admission. A supplemental source
review has been requested from the already authorized review chat for
stage `M2_context_recovery_execution`, both draft hashes, full169-source map
and exact target root. The requested route is unavailable; do not attribute
Astra or independent-family scientific acceptance.

Only after actual SOURCE_PASS0 and exact bindings: minimally create the new
snapshot; verify sources/data/remote CPU checks and capacity; launch a new
5-update smoke; CPU replay its real native/receipt/checkpoint; bind one full
run starting from fresh zero-KEEP; then run the new terminal verifier once.
Keep latest/lastfive/qualifying best and all audit dependencies.

Original gates remain zero reliable harmful crossings, nonnegative reliable
sequence-macro IoU delta and lost recovery greater than179/800. No qualified
epoch means retain protected E0, no online/paired/benchmark deployment.
The current TEST inventories remain exploratory and ineligible for selection.
No further run of the rejected R2 candidate, seed grid or gate relaxation.
