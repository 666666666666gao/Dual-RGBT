# C2 First Full TRAIN Epoch

These are native full-training measurements, not ten-update smoke estimates.
No benchmark improvement or convergence is established by an epoch completing.

| Branch | Variant | First epoch seconds | Minutes | Mean TRAIN loss |
| --- | --- | ---: | ---: | ---: |
| DropMoE / P25 | zero | 1054.500280061009 | 17.5750 | 2.723912375152111 |
| DropMoE / P25 | FiLM | 1054.4136543110071 | 17.5736 | 2.7248514322042463 |
| SOT / P26 | zero | 1283.0420270000177 | 21.3840 | 2.7212409415602683 |
| SOT / P26 | FiLM | 1283.2002831709979 | 21.3867 | 2.719324039709568 |

Both P25 runs completed 2500 sequential optimizer updates in epoch1.
Each of the ten modality/action routes received exactly250 updates.
All observed loss and gradient-norm values were finite. At07:09:25UTC both
actual training children were still live on their assigned GPUs and their
CSV files had reached2750 updates in epoch2. Exact native command, config,
review hash and child GPU ownership passed the existing runtime witness.

Raw evidence is in
`c2_training_full_r1_20261103/dropmoe/{zero,film}/epoch_001_witness/`.
All fourteen archived raw-file SHA256 bindings were read back and verified.
The observer `work/archive_c2_completed_epoch.py` reuses the existing
read-only start-witness checks; it does not deploy or mutate training.

At unchanged first-epoch throughput, twenty P25 epochs imply about5.86h of
epoch execution. This is an estimate, not a measured full-run duration;
later data loading and saving/strict-reloading epochs16-20 add uncertainty.
The first epoch includes its dataset/loader construction and epoch-end
finite-state/frozen-policy checks. Native wrapper startup is outside the
printed epoch duration. There is no checkpoint saved before epoch16.

At07:11:59UTC both P26 runs also had an actual `EPOCH_COMPLETE 1`
log line and2500 sequential epoch1 updates. Ten routes each had250 updates;
all recorded losses and gradient norms were finite. Native tasks, actual
children55899/55900, config/review/precision and GPU1/GPU2 ownership passed.
Their evidence is in
`c2_training_full_r1_20261103/sot/{zero,film}/epoch_001_witness/`.
The unchanged first-epoch P26 pace implies about7.13h for20 epoch executions,
with the same startup/checkpoint caveats. Neither branch is converged merely
because its first epoch finished. These timings supersede the earlier
startup-inclusive19min/23min rough estimates.

Keep zero and FiLM recipes fixed. A mean TRAIN loss difference between the
two runs is not a reason to select a tracker. Later epochs must establish
TRAIN trajectory; newly bound policy refresh is required before claiming a
fully adapted tracker, and complete exploratory benchmark evaluation is
still required. Original prefix-sampled P1A is not the matched control for
this full-timeline sampler.

## Next Loader Boundary

Source inspection confirms `dual_rgbt/e0_policy_refresh.py:111` builds the
original `DualRGBT`, expects P1A config `hashes` and native command arguments
`--run`, `--cadtrack`, `--manifest`, `--parent`. The new C2 trainer is invoked
with `--plan`, saves `c2_config`, and its FiLM model has learned missing-stream
parameters. Thus the current P1B entry is not a C2 policy-refresh entry.
Do not pass C2 checkpoints into the old P1B loader or strip its assertions.
A separate versioned C2-aware loader/collection/refresh plan must bind actual
C2 completion, checkpoint, architecture config and newly collected TRAIN
records before later GPU admission. Preserve both original host snapshots
and currently running policy FITs.

## Second Epoch Observation

P25 second epochs completed in924.2070409240114s (zero) and
924.2379472930043s (FiLM), about15m24s. Both are shorter than their first
epochs; update the throughput estimate instead of assuming fixed startup
overhead repeats. Both have2500 sequential epoch2 updates, ten routes250each
and finite values. Raw evidence:
`c2_training_full_r1_20261103/dropmoe/{zero,film}/epoch_002_witness/`.
At07:31:17-19UTC both actual children were in epoch3 at6350 total updates.
Mean TRAIN losses were2.7093974417448043 and2.7099563408851624, versus
2.723912375152111 and2.7248514322042463 in epoch1. The small early reduction
does not establish convergence, superior tracking or permission to stop.
Keep both admitted recipes and evaluate a longer TRAIN trajectory.

P26 second epochs completed in1184.9928113479982s (zero) and
1184.9947915250086s (FiLM), about19m45s. Their mean TRAIN losses were
2.697331616806984 and2.6973360337257386. Both actual children reached
epoch3/5450 total updates at07:35:10-11UTC, with2500 sequential epoch2
updates and250 per route. Their raw second-epoch witnesses are archived
under the corresponding `sot/{zero,film}/epoch_002_witness/` directories.
These observations still do not establish convergence or a tracking gain.

## Latest Completed Epochs

P25 epoch5 completed in856.5600957069983s (zero) and857.2656706370035s
(FiLM), about14m17s. Mean TRAIN losses were2.644270270729065 and
2.6371117196798326. Both had14525 total flushed updates in the live process
at08:17:55-57UTC and had already entered epoch6.

P26 epoch4 completed in1125.41070167601s (zero) and1125.4603821559867s
(FiLM), about18m45s. Mean TRAIN losses were2.6722501623153687 and
2.674163984155655. Both had11150 total flushed updates at08:17:56-57UTC
and had entered epoch5.

Each archived epoch passed2500 sequential updates, ten routes250each,
finite losses/gradients, native child/GPU ownership and exact plan/config
checks. Raw witnesses are in the matching `epoch_005_witness` P25 and
`epoch_004_witness` P26 directories. These are still TRAIN trajectory
measurements, not convergence, checkpoint-selection or benchmark evidence.

P25 epoch6 completed in859.3457382560009s (zero) and855.5157703139994s
(FiLM), about14m19s/14m16s. Mean losses were2.6263293848514557 and
2.6352189131736754. Both live processes had15875 total updates and entered
epoch7 at08:25:38-39UTC. Raw witnesses:
`c2_training_full_r1_20261103/dropmoe/{zero,film}/epoch_006_witness/`.

P26 epoch5 completed in1082.3991268620011s (zero) and1082.228866949008s
(FiLM), about18m02s each. Mean losses were2.6333509739160537 and
2.634044414615631. Both live processes reached13250 total updates and
entered epoch6 at08:33:04-06UTC. Raw witnesses:
`c2_training_full_r1_20261103/sot/{zero,film}/epoch_005_witness/`.
