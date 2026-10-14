# Global evidence: full tracking evaluation protocol

This protocol adds an evaluation entry point for the existing global evidence
policy. It does not alter the 30 training source files or the active training
plan. Full dev/test execution is NOT_RUN as of 2026-10-14.

`scripts/run_global_evidence_tracking.py` loads the global policy strictly,
including its global query head. The old evaluation entry point cannot load
this policy. The new entry point reuses the existing causal global tracker,
split loader and unchanged `scripts/score_tracking.py` with the user's RGBT
toolkit.

## Admission

The evaluation plan records the 30 existing training source hashes plus the
new driver hash. Before running the driver, an independent source review must
record `SOURCE_PASS`, zero blocking issues, the same source map and the exact
evaluation plan hash. Review remains same-family/provisional; routing cannot
be attested by the executor.

The first GPU execution is an 11-frame prefix of training sequence
`11runthree`: one initialization and ten online predictions. It uses a completed
formal A or B checkpoint, not a live checkpoint or a sanity checkpoint. This
read-only driver has zero optimizer updates and zero backward calls. The
existing A/B/C admission runs supply the training path's separate gradient
and optimizer witnesses. This driver still requires its own actual frozen
execution and strict checkpoint reload checks.

The smoke run must finish with a native exit code of zero, all model state
tensors unchanged, no parameter gradients, finite predictions with positive
width/height, and a strict reload that reproduces the last box, routing fields,
history and templates exactly. It writes `SANITY.json` and `SMOKE.json`; it
does not write benchmark `COMPLETE.json`. The unchanged scorer rejects smoke
runs. GPU 0 and 1 currently have their own training owners; do not start this
smoke run until an authorized card is free.

## Full dev and test

Full evaluation requires the completed formal global C checkpoint with
182,182 updates. The driver checks the native training task's actual completion,
checkpoint identity, protected baseline identity, manifest hash, source hashes
and training plan hash. A/B checkpoints are accepted only for execution smoke.

Dev contains 97 sequences and 54,646 frames. Run full dev at each planned cost
weight, 0.0, 0.05 and 0.15. Select by highest full-dev SR, then PR, NPR and lower
mean service latency, using the same rule as the previous pipeline. Record all
three full-dev results. Freeze the selected checkpoint hash, cost weight, dev
coverage, evaluation plan hash and training plan hash in
`reports/global_evidence_frozen_test_selection.json`. This is a separate file
from the older pipeline's selection.

Test contains all 245 sequences and 220,703 frames. Both test shards must use
the same frozen selection and cover distinct complete sequences. The scorer
requires all shards, all sequences, correct frame counts, finite predictions,
and the unchanged toolkit's ground truth. Only completed full test scoring
produces the final LasHeR PR/NPR/SR. No baseline accuracy rerun is planned.

## Causal state and annotation use

Only each sequence's initial annotated box enters the tracker. Later search
centres, scale, history and template updates follow its own predictions. Future
annotations are written for offline scoring after prediction; they are never
passed to the online tracker, selector or evidence query. Test annotations are
not used for training, threshold selection or checkpoint selection.

## Timing

Frame service timing includes image decoding, initialization, both cheap
previews, main encoding, policy selection, crop/gather work, the selected
auxiliary encoding, correction and applicable template updates. The existing
tracker synchronizes GPU work before returning. Model loading, checkpoint
verification, trace/prediction writes and metric scoring are excluded from
service timing and are reported separately where applicable.

The aggregate scorer reports the mean, P95 and frames divided by summed
service time across shards. Concurrent shard wall throughput must not be
reported as single-stream FPS. This entry point supplies timing for the global
system only; a lower cost claim still needs a paired measurement of the
reference system on the same device and evaluation frames.

The source review, smoke admission, full C, full dev selection, full test and
paired efficiency measurement are separate evidence requirements. Preparing
this entry point is not evidence that any accuracy or latency target is met.
