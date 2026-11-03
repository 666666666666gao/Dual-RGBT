# Executor Pair Coverage and C2 CPU Findings

## Observed TRAIN Search Domain

The original `dual_rgbt/data.py:15` sets the static template to the first
valid frame and restricts search candidates to the next 200 frames.
Existing `tests/test_executor_v2_data.py` explicitly tests that restriction.
This is the implemented P1A V2 recipe, not a newly introduced C2 behavior.

Read-only metadata audit of the exact P25 TRAIN manifest found:

| Domain | Search Frames |
| --- | ---: |
| Current first-valid-frame plus 200 window | 152,786 |
| Full timeline with a valid earlier template within 200 frames | 457,239 |
| Later eligible frames excluded by the prefix window | 304,453 |

550/882 sequences have excluded later frames. The current candidate domain
covers 33.4149% of the full-timeline causal domain. This is domain coverage,
not a reconstruction of realized minibatches or proof of an accuracy cause.
Exact manifest/source hashes and all per-sequence counts are in
`P1A_V2_TRAIN_SEARCH_DOMAIN_AUDIT_20261103.json`.
The audit uses no images, GPU, dev or TEST data.

The repository's CADTrack `lib/train/data/sampler.py:306` samples search
frames across the sequence and applies its max-gap window to the dynamic
template relative to search. Its `trident` sampler can also use a future
dynamic template; the proposed causal variant keeps only earlier templates.
No claim of exact official-recipe equivalence is made.

A separate LOCAL `dual_rgbt/c2_data.py` permits all valid search frames with
an earlier valid frame within 200, keeps the first valid static template,
and selects a past-only recent template. Original data.py and active
snapshots were unchanged. Six new sampler tests and both original V2 tests
passed locally (8 tests, 1.53 seconds). This code is not reviewed/deployed
for training. Any future C2 comparison using it requires a fresh zero/no-LoRA
matched control with the same sampling recipe; reusing the prefix-sampled
P1A checkpoint would confound sampling and C2 changes. Existing P1B work
continues as its separately declared experiment.

## C2 Real CPU Check

Source-only independent review
`P2_C2_EXECUTOR_CPU_SOURCE_REVIEW_R1_20261103.json` returned SOURCE_PASS/0
for the exact four-file payload, conditional isolated CPU deployment.
Actual model attribution remains UNATTESTED/same-family/provisional.
The original98-source closure, mapped dependencies and existing factory,
executor and TRAIN-manifest hashes were verified before execution.

The real checker ran using P25's warm NN Python in the new isolated root
`/data2/gaob/Dual-RGBT/c2_cpu_check_r1_20261103`,
with CUDA hidden, one CPU thread and no active-snapshot source changes.
It exited1 at the per-adapter nonzero auxiliary-bank gradient assertion
`assert adapter.b.grad[2].abs().sum() > 0` before optimizer.step.
It wrote no result.json or checkpoint and completed zero optimizer updates.
The exact invocation, source review, failure log and four source files are
archived in `reports/c2_executor_cpu_r1_20261103/`.

Control flow reached that assertion only after four variants' ten-route
nested equality checks, four predict comparisons, expected parameter counts
and optimizer-coverage checks. These preceding checks do not make the
overall checker PASS. C2 GPU/native training remains unadmitted.

The original YAML configures DROP_PATH_RATE0.1. A real CPU-only diagnostic
reproduced the initialization order and captured actual DropPath outputs.
Block10 auxiliary MLP output was entirely zero, exactly matching zero
auxiliary-bank gradients for its fc1/fc2 adapters. Block11 primary MLP
output and its primary-bank gradients were also zero. The other active
branch B gradients were nonzero; every inactive local-bank B gradient
was zero. Missing embedding gradient L1 was10.49579, active FiLM weight/
bias L1 values6.25511/2.38248; unused FiLM gradients were None. Diagnostic
TRAIN loss0.753978 was finite; no optimizer step or GPU initialization
occurred. Actual evidence is `gradient_diagnostic.json` in the R1 archive.

The local R2 checker disables DropPath only during its deterministic
single-pair gradient-connectivity diagnostic, retaining all nonzero-bank
assertions. Native training's stochastic-depth configuration is unchanged.
This corrected checker still requires a fresh source supplement and a new
isolated run; no R2 pass is claimed yet. Missing FiLM gradients for an
unused modality are recorded as None, not as a numerical error.
Merged inference and P2 policy-collection integration are still absent.
