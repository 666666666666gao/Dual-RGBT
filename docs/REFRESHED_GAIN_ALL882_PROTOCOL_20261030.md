# Frozen-policy TRAIN refresh over all 882 sequences

Status: preregistered protocol; no training admission or accuracy claim.
Recorded on 2026-10-30 before the complete 882-sequence cache was available.

## Question and Scope

Can the existing gain head learn useful decisions from the TRAIN states
visited by frozen focal E6 when all 882 TRAIN sequences are represented?
The matched100 experiment remains a separate negative admission result:
its holdout has 9 sampled recoverable states, below the declared minimum 15.
Neither its partition nor its admission threshold will be changed.

This experiment expands both sequence coverage and the visited-state data.
It is not a single-factor or equal-update comparison against the original
100-sequence parent cache. Report actual states, batches, updates and epochs.
The existing three-module architecture remains unchanged.

## Fixed Inputs

- Use all 16 registered complete caches: matched100 shards 0-7 and
  remaining782 shards 0-7 from the existing per-server refresh registries.
- Preserve the preregistered 706 fit / 176 holdout sequence membership.
  Verify disjoint coverage of exactly the manifest's 882 TRAIN sequences.
  Do not select sequences, frames or candidates using observed outcomes.
- Initialize learning from the original parent checkpoint
  `21e07cd1f645c6f7237e063c81cf5d60f6a14bca710e0180e8a3ca29dc827073`.
  Frozen focal E6 supplies visited states and recorded-output verification,
  not the learning initialization.
- Use the existing original regression/margin objective (`parent_loss`),
  seed 2027, batch_states 2866 and LR 0.00112.
- Keep formal min_epochs 8, max_epochs 30, patience 4 and min_delta 0.0001.
  Stopping uses the existing component-holdout sampled utility regret.
  This component holdout was seen during parent training; it is not an
  unseen-system generalization benchmark.

## Admission and Execution

Every cache must have exact planned sequence/frame coverage, native
COMPLETE/exit 0, artifact hashes and matching frozen E6 identity.
Preserve originating server commands and receipts during any transfer.
Use the existing audit and loader contracts to verify transferred bytes.

Require at least 100 fit sampled recoverable states across 10 sequences
and 15 holdout sampled recoverable states across 3 sequences.
Do not lower these gates or replace missing states after seeing outcomes.

A separate all882 preparation/launch registration needs independent source
review before execution. The existing matched100 plans and reviewed sources
remain immutable. An all882 two-epoch smoke must verify finite real updates,
the same four gain-head tensors changing, recorded E6 output agreement and
strict checkpoint reload. Its exact cache/recipe/source bindings must match
the formal plan before that single formal run can be admitted.

This cached-head workload has historically reserved about 0.6 GiB. Do not
call it full-network training or allocate dummy tensors to reach 20 GiB.
Use a measured fitting card and retain whole cards for admitted size training.
Poll long tasks at 180-300 seconds or near measured completion boundaries.

## Selection and Reporting

Retain the latest five weights and each weight's complete 97-sequence,
54,646-frame LasHeR dev score. Share an evaluation GPU where measured capacity
allows it; keep full-dev inference and the official scorer unchanged.
Use dev only for settings and weight selection. Keep the protected E0,
parent and previous loss results in the comparison.

Report PR/NPR/SR from complete dev separately from sampled recovery counts,
regret, crop loss and concurrent execution timing. Preserve the selected
configuration before any final LasHeR/RGBT210/RGBT234 test.
