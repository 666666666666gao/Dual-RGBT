# C2 Policy Loss R1

## Actual Interface Problem

The old `dual_rgbt/e0_policy_refresh.py` policy loss invokes `primary` and
`correct` without `preview`. C2's methods in `dual_rgbt/c2_executor.py` require
the keyword-only `preview` argument to condition missing-stream embeddings.
The old loss cannot be called directly on a C2 executor. Merely adapting the
checkpoint loader does not resolve this runtime interface mismatch.

New local-only `dual_rgbt/c2_policy_loss.py` keeps the original supervision:
quality is negative primary localization loss, gain is primary minus each
query localization loss, plus0.1 spatial cross-entropy. It computes preview
once per update and explicitly passes that same preview to two primaries and
all eight queries. Executor outcomes and labels are recomputed without
gradient; preview, quality and gain predictions retain their gradient paths.
The whole model enters eval mode, with only those three policy groups
trainable. Backbone, decode/head, missing embeddings and adapters stay frozen.
The original loss and both running host snapshots are unchanged.

## Actual Checks

Four new tiny CPU tests cover same-preview flow, all ten outcomes, detached
labels, policy gradients and actual optimizer changes, preservation of
nonpolicy parameters/BN buffers and inputs, updated-preview relabeling,
off-crop spatial omission and an explicit window boundary.

Actual local commands:

```powershell
E:/anaconda/python.exe -B -m pytest -q tests/test_c2_policy_loss.py tests/test_c2_policy_refresh.py tests/test_c2_training_admission.py tests/test_c2_primitives.py
E:/anaconda/python.exe -B -m pytest -q tests/test_e0_policy_refresh.py tests/test_c2_policy_loss.py tests/test_c2_policy_refresh.py tests/test_c2_training_admission.py tests/test_c2_primitives.py
```

Results:114passed in6.38s; expanded old-P1B regression125passed in5.89s.
These are toy CPU/interface checks, not real C2 forward/backward, actual
checkpoint reconstruction, full TRAIN acceptance or benchmark evidence.

## Supplemental Source Review

The authorized reviewer produced
`P2_C2_POLICY_LOSS_SOURCE_REVIEW_R1_20261103.json`:
stage `P2_C2_policy_loss`, SOURCE_PASS/0, exactly the two new files,
empty admitted plan bindings and no deployment/GPU/collection/FIT admission.
Report SHA256:
`c07cc1a18f4d860cf38f21377eed56096ce61110fc86316ea89bb84b951bda7d`.

Exact source SHA256:

- Loss:abae997522841b66cc338da2a0c7fad277efae2ae275df2be33f8394c4852404
- Tests:a899f191607fe61de068b507ff25006fd0280dc05e32084bcadf32bfa00d4705

The reviewer actually reran114 tests in5.56s and compared normalized ASTs
against the original functions. Apart from docstrings, equivalent syntax/
local names and the two new preview keyword arguments, the loss formulas
and parameter groups matched. Attribution remains
UNATTESTED/same-family/provisional, not a claimed Astra or independent-family
scientific acceptance.

The later collection/FIT entry must separately admit a source closure and
recipe bound to actual full C2 completion, last-five weights, the exact
executor and newly collected TRAIN states. Actual C2 reconstruction,
CPU/GPU forward/backward, frozen-state checks, serialized reload and capacity
remain mandatory before new GPU work. This primitive is not deployed.
