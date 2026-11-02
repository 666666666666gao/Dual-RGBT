# Actual Recovery Objective TRAIN Result

The single objective change was executed on P25GPU2, using the accepted
actual TRAIN tensor export from the prior experiment. It made no collection,
export, TEST inference or E0 fine-tune updates. Original163 sources stayed
unchanged; six additive sources used an isolated new snapshot.

| Evidence | Result |
|---|---|
| Source/draft review | SOURCE_PASS0,169 sources/two exact disabled drafts |
| Real snapshot/data/remote CPU admission | PASS,14762fit/3949held sets,5new tests |
| Real GPU smoke | COMPLETE/exit0,5 updates |
| Real GPU full TRAIN | COMPLETE/exit0,12epochs348updates |
| Terminal CPU verifier | Completed ONCE; epochs8-12/latest replayed |
| Qualifying epoch | None; retain protected E0 |

Target `/home/gaob/Dual-RGBT/snapshots/context_recovery_r1_20261102`;
task `train_e0_context_recovery_full_r1_p25_20261102`, wrapper659102.
The native elapsed9.164901sec and training-engine3.348793sec are for the
961-parameter cached head, not full E0 training or benchmark throughput.
Peak allocated/reserved GPU memory7676416/25165824bytes. Initial allKEEP,
finite gradients, changed parameters, fit-only normalization/recovery weights,
strict reload parity and fresh initialization were verified. Latest and
epochs8-12 retained; no qualifying best was created.

| Epoch | Held reliable harm /2899 | Held recovery /800 | Qualifies |
|---|---:|---:|---|
|1|828|250|No|
|2|1050|265|No|
|3|1129|271|No|
|4|1197|276|No|
|5|1175|276|No|
|6|1161|275|No|
|7|1163|274|No|
|8|1201|273|No|
|9|1240|273|No|
|10|1271|271|No|
|11|1293|273|No|
|12|1305|275|No|

Original gates require zero reliable harmful crossings, nonnegative reliable
sequence-macro IoU delta, and lost recovery greater than179/800. Every epoch
failed. Epoch12 reliable macro delta is-.3273872720; overall macro delta
is-.2408831632. The recovery gate alone is insufficient.

The prespecified actual CPU audit reproduced the accepted latest metrics
and used zero updates/no CUDA. Comparing fixed epoch12 in the two
prespecified runs, without selecting an epoch:

| Diagnostic | Original R2 | Added recovery objective |
|---|---:|---:|
| Fit lost recovery /3477 |245|1427|
| Held lost recovery /800 |52|275|
| Fit recovering-action predicted mean |-.24833313|+.54559869|
| Held recovering-action predicted mean |-.25885832|+.50879067|
| Held reliable harmful changes /2899 |12|1305|
| Held no-positive-correct lost states /374 |293|23|
| Held positive-correct but outranked /374 |29|76|

The added objective substantially changes the intended fit recovery signal
and held recovery, but causes a large protection regression already at epoch1.
It does not produce a usable module. Retaining the original soft penalty
did not enforce the original zero-harm requirement under this intervention.
This supports examining state-conditioned protection/action admission or
explicit risk constraints, not blindly extending epochs or sweeping
recovery coefficients/seeds. It does not prove that any particular gate,
network size, constraint or proposed solution will succeed.

The observed-action groups, fit/held macro decision metrics and
KEEP/wrong/recovered decomposition are archived in
`E0_M2_CONTEXT_RECOVERY_RESULT_AUDIT_R1_20261102.json`.
GT defines TRAIN supervision/diagnostic groups only; it is never a forward
input or online gating rule. Holdout has been used during component
development and is not an independent final validation set.
The first exploratory audit output was inspected without archival; a second
zero-update invocation saved the report. The canonical terminal verifier
was executed only once; it was not rerun for diagnosis.

| Binding | SHA256 |
|---|---|
| Source review |cb4a00e29c7fb196a97f2a0f374cb5480b016639abd6ca290240db336ad1f46e|
| Full plan |7958b09f753e87112bb89b35fc2af3a3d81d048b7216201b2c909930de77e83a|
| Accepted tensor |b79b4a3db2bd7d1c2d639b737a5cd0b7d6b8c10857a59a5a1fbd7af3f2b031f3|
| Terminal acceptance |dacd4f4c3be16434768b7b1ab22214c5040feac6620c50a460fee6c9791eec04|
| Latest |0832aaf34445956d5a7318ba4fe025a73c768a6fb66c902b882f47646ce65295|
| Native |9b346077b1e612a9d65c5626b1c7e778317b6a9fff34170e5aa5f8766f5d223e|
| History |a2378d52f161a7e9e5ca981822e3e4a41c9940b3d919498133b3a3fac810c359|

The authorized supplemental review chat has been asked to diagnose this
actual failure and trace whether learned M1 failure recognition currently
participates in M2 action admission. No next recipe is admitted by that
request. Do not deploy the rejected head online or to paired/benchmark runs;
do not repeat this fixed experiment or change original qualification gates.
