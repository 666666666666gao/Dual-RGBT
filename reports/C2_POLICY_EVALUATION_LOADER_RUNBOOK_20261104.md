# C2 policy FIT evaluation loader

This is local source preparation for a later evaluation boundary. The active
C2 policy FIT is not complete. This module does not admit evaluation, deploy
sources, run a tracker or produce benchmark evidence.

`dual_rgbt/c2_policy_evaluation.py` authenticates fixed final epoch005 from
the five-epoch, 890-update policy FIT. It checks eight explicitly pinned
evidence documents: enabled plan, actual config, native terminal status,
COMPLETE receipt, precision, FIT admission review, collection acceptance and
replay native status. It authenticates admitted FIT source bytes and the
original C2 executor through `completed_executor_state`.

All five retained FIT checkpoint hashes must match the completion receipt.
The final checkpoint must bind the exact plan and executor; all tensors must
be finite, every non-policy tensor must equal the original C2 state, and
preview, quality and gain must each change. The original C2 checkpoint
provides the architecture config because FIT checkpoints do not save a
top-level `c2_config`. Reconstruction uses `C2DualRGBT` and strict loading.
The original P1B evaluation entry constructs `DualRGBT` and is unsuitable.

CPU verification:

```powershell
& E:/Anaconda/python.exe -B -m pytest -q tests/test_c2_policy_evaluation.py tests/test_c2_policy_refresh.py
```

Actual result: 128 passed. The new suite uses synthetic checkpoint and
receipt fixtures for zero and FiLM. Its constructor test substitutes a
small class to check architecture arguments, strict loading, eval mode and
frozen parameters without the local training dependency stack. It is not
an actual full-model load or GPU witness. The original C2 completion chain
is tested separately in the existing refresh suite.

Exact source-only review is SOURCE_PASS/0:
`reports/P2_C2_POLICY_EVALUATION_LOADER_SOURCE_REVIEW_R1_20261104.json`,
SHA256 `f1f80fa32d419d3ecd0aed3dff82514e1519676600f5ec5d3bfdac76b333de3d`.
Attribution is UNATTESTED/same-family/provisional, execution bindings empty,
and scientific acceptance false. This admits no runtime plan.

The loader checks the pinned replay acceptance and terminal command; it
does not repeat sequence collection or numerical replay. Admission and FIT
already validate the full collection chain. A future evaluation entry and
its exact sources, terminal FIT inputs, plan, runtime admission and TRAIN
smoke require separate review before full TEST. TEST remains exploratory
and test-ineligible, with fixed epoch005/cost0.05 and no dev or selection.

Preserve active roots and host-specific original snapshots. Do not deploy
this local preparation into active FIT or replay roots. Do not invoke it
with a RUNNING FIT or substitute the old P1B loader.
