# C2 Remote Scoring And Archive

`work/score_c2_policy_completed_remote.py` is local source preparation only.
It wraps the existing reviewed C2 CPU scorer without changing its metrics
or inference/model code. No concrete C2 evaluation has completed or been
admitted by adding this helper.

## Exact Future Plan Contract

In addition to the C2 scorer/entry requirements, the separately reviewed
enabled full-evaluation plan must bind:

- `plan_path`: the exact absolute remote evaluation plan path, also the
  final argument in its native evaluation command.
- `scoring_interpreter`: the executing host's accepted cpu-scoring Python.
- `score_output`: JSON outside the inference run, with associated curves
  NPZ and `.scoring.log` paths.
- Both `work/score_c2_policy_completed_remote.py` and
  `tools/score_c2_policy_completed.py` in the exact `sources` map.

These fields are included in the canonical admitted binding. Actual completed
FIT5ep/890updates/finalepoch005, eight terminal pins, original host sources,
CPU model reconstruction, own TRAIN smoke and full evaluation runtime admission
remain prerequisites. No old P1B/C2 loader substitution or repeated replay.

## Retrieval

The local invocation takes authenticated local copies of the final enabled
evaluation plan and review, plus a new archive destination:

```text
<local-python> -B work/score_c2_policy_completed_remote.py \
  --plan <local-exact-final-evaluation-plan> --review <local-exact-review> \
  --archive <new-local-archive-directory>
```

The helper authenticates the exact plan/review binding and local helper/scorer
source pins, then uses SSH ClearAllForwardings=yes/BatchMode=yes and the host's
CPU scoring interpreter. Remote bytes, sources, terminal native COMPLETE0,
command/task/GPU and exited wrapper are checked before the scorer.
The unchanged scorer performs its full existing native/FIT/input/source/
precision/GT/trace/timing/coverage/provenance checks.

A first score invokes toolkit metrics once and preserves its scoring log,
including failures. A complete existing score uses the scorer's validation-only
path; it does not execute metrics or rewrite score/curves/original log. A
partial curves/log output without score is preserved and rejected. Do not
retry failures by deleting evidence or reuse an existing local archive.

The archive contains score, curves, original scoring log, exact evaluation
plan/review/native/config/complete/precision/protocol and all eight FIT
documents. All 18 raw SHA256 values are checked before a new directory is
created. The final audit records metrics, source helper pin, plan binding and
whether the remote score already existed. Weights, dataset images and raw
per-frame inference artifacts stay on the executing host; the score binds
their artifact hashes.

TEST remains exploratory and test-ineligible. No training, checkpoint/
threshold/branch selection or MAC profiling occurs. Timing includes decode
and initialization under concurrent load, excluding model loading; no isolated
speed or scientific-success claim follows.

## Verification Scope

Tests call the real existing C2 scorer on its synthetic three-benchmark
fixtures, with mocked toolkit curves and a mocked subprocess boundary. They
cover first-score retrieval, validation-only reuse, native/plan/source failure,
partial preservation, failed scorer log, tampered transfer and existing local
archive rejection. They are not real remote CPU/GPU/model/benchmark witnesses.
Actual deployment and invocation require their own exact plan review.
