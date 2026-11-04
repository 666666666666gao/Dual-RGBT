# C2 Completed Policy Scorer

`tools/score_c2_policy_completed.py` is a LOCAL, CPU-only source candidate.
It does not deploy, launch inference, train, choose checkpoints, or admit
an evaluation plan. The active five P1B evaluations and three C2 jobs
remain unchanged.

The old E0/P1B scorer cannot consume the C2 entry's nested-plan config.
In particular its LasHeR helper rereads the old flat config from disk.
This separate scorer calls the supplied toolkit's PR_LasHeR/NPR/SR_LasHeR,
PR/SR, or MPR/MSR directly. Metric implementations and aggregation match
the existing scorers: precision column20 and mean success curve, reported
as percentages. No existing toolkit, host core, or active snapshot changes.

## Required Evidence

A future exact enabled `P2_C2_policy_evaluation` plan must include this
scorer and benchmark_inputs in its reviewed source map, all existing C2
inference dependencies, and `toolkit` plus SHA-bound `protocol` fields.
The exact canonical evaluation binding includes those scoring inputs.
Source-only review is not an execution binding.

Only full TEST is accepted: LasHeR245/220703, RGBT210210/104706, or
RGBT234234/116650. Fixed final policy epoch005/cost0.05, no dev or TEST
selection; exploratory_test=true/test_eligible=false. All eight FIT
evidence documents must match their admitted hashes, with genuine
COMPLETE0 FIT, five epochs/890 updates/91091 records and final weight hash.
The evaluation entry's reviewed C2 loader performs architecture/tensor
validation before inference. This CPU scorer authenticates that evidence
and final weight bytes without importing Torch or replaying training.
It does not independently reconstruct or certify the trained model.

The scorer authenticates evaluation native COMPLETE0/command/GPU,
plan/config/review/source/input/checkpoint/precision/completion bindings;
exact progress and sequence/frame counts; finite positive predictions;
saved GT versus manifest and toolkit; every N-1 trace frame and its exact
decode-inclusive timing; complete finite positive frame timings and total.
Toolkit source pins must cover all seven existing protocol metric/dataset
files. RGBT additionally checks the manifest's annotation file hashes.

## Future Invocation

Run from the separately reviewed inference root using the accepted host
CPU scoring interpreter, after genuine full inference completion:

```text
<cpu-python> -B tools/score_c2_policy_completed.py \
  --plan <exact-enabled-evaluation-plan.json> --run <completed-run> \
  --toolkit <unchanged-toolkit> --protocol <pinned-protocol.json> \
  --protocol-sha256 <admitted-protocol-sha> --output <outside-run/SCORES.json>
```

This is documentation only, not an authorized concrete command or plan.
No evaluation plan, deployment, model CPU/GPU witness, smoke, full
inference, or actual scoring was performed for this source preparation.
Future execution needs exact terminal FIT evidence, full host-specific
dependencies, actual CPU reconstruction, own TRAIN smoke and runtime
capacity/ownership admission. Old P1B factories remain invalid for C2.

## Output And Preservation

The first invocation writes curves NPZ and score JSON only after validation,
using exclusive creation. Curves already present without score are partial
failure evidence, never overwritten. Output must be outside the run.
Every prediction/GT/trace/native/config/completion/precision/progress/timing
artifact hash, eight FIT document pins, input/source/review/protocol/toolkit,
checkpoint and scoring script hashes are bound in the report.

An existing score is validation-only: reauthenticate current evidence,
exact curves hash and aggregate/per-sequence values, never rerun metrics
or overwrite files. Preserve any failure and all raw inference artifacts.
Future remote archive orchestration still needs its own exact source/input
review; no new remote archive helper is admitted here.

FPS includes decode/initialization under actual concurrent load, excludes
model loading, and establishes no isolated speed advantage. There is no
MAC calculation or new benchmark/goal-success claim.

## Local Validation

Actual focused scorer tests:102 passed. Combined scorer, C2 evaluation
entry, policy loader and executor loader tests:259 passed. Fixtures shrink
counts only via test monkeypatches; production counts remain exact. These
are synthetic CPU fixtures with mocked toolkit/model boundaries. Direct
toolkit class dispatch is exercised with test doubles; genuine toolkit
benchmark scoring and actual C2 inference remain pending.
