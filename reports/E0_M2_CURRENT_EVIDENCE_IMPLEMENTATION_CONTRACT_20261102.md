# Current E0 Evidence Implementation Contract

The original design and its `DESIGN_REVISIONS_REQUIRED` review remain
unchanged. This document adopts the concrete implementation requirements
from `E0_M2_CURRENT_PROPOSAL_TRAIN_DESIGN_REVIEW_20261102.json`.
It grants no collector, trainer or GPU admission.

1. Use the exact original `build_model` then `build_tracker` import/crop
   setup. The original collection binds `six_channel_crop` before tracker
   import. A later nominally equivalent crop patch is insufficient.
2. Reconstruct the current image search crop from accepted
   `predictions[frame-1]`, using original factor4/size256 pixels and
   geometry. Map the returned clipped `predictions[frame]` into it;
   do not recenter the crop or use the observer's pre-clipping box.
3. Use fixed initial templates for both full-grid encodes. Reuse accepted
   anchors and24candidate arrays. Dynamic-template primary evidence does
   not generally equal this evidence. Call encode, not primary/box_head.
4. Own one observer for the shared model. Call begin(False) for the
   control tracker and begin(sampled) for the observed tracker, require
   exactly one E0 predict for each, copy observations immediately, and
   restore the saved predict once. Never stack M1 guard observers.
5. Require current and previous boxes bit-exact to accepted predictions,
   plus paired non-timing route/template/history/frame/query parity.
   Snapshot the observed tracker before/after supplemental encodes and
   check no mutation. Stop on failure rather than join new states to old
   candidate rows.
6. Collect914-D M1 observations separately at identical sampled keys.
   GT is read only after all observations/encodes; no GT-driven online
   gates. Record total paired replay, supplemental encodes and timing.
7. KEEP must remain finite and executable even when its ROI lacks a fully
   covered modality. Proposed future scorer takes KEEP-only in that
   measured case; the descriptor helper only reports coverage. It does
   not implement KEEP, masks, logits or a policy.
8. Relative targets require a reviewed common-float64 geometry convention
   and explicit loss participation for unknown/covered candidates.
   Original identity no-match and labels must remain immutable. Exact
   action-score ties prefer KEEP. No TEST-derived margin.
9. Before the one fixed training recipe, predeclare conservative TRAIN
   acceptance: zero reliable-to-below0.5 crossings, nonnegative
   sequence-macro reliable IoU delta versus KEEP, and more than179/800
   lost recoveries on the same held baseline states. Rank qualifying
   epochs by sequence-macro overall IoU delta, then recovery count, then
   earliest epoch. If none qualifies, keep baseline; do not relax after
   seeing results. This is proposed engineering selection, not a
   guarantee that training will pass or independent validation.

## Implemented Local Helper

`dual_rgbt/e0_current_evidence.py` accepts caller-provided original search
tensor/geometry, clipped current box and fixed templates. It performs
two encoder calls and existing ROI pooling, with no box head, GT,
tracker state, observer installation, optimizer or online connection.
It returns(2,768)descriptors,(2,)coverage and normalized current box.
The caller remains responsible for actual crop, source and trajectory
provenance; this function cannot attest those by shape checks.

Local `tests/test_e0_current_evidence.py` passed2checks for actual
crop-derived coordinates, modality order, zero coverage, no gradients,
exactly two encodes and unchanged inputs. Existing identity tests passed5.
The first package-style unittest invocation failed because tests is not
an importable package; direct existing script invocation passed.
Tests use synthetic encoders, not real pretrained model or GPU execution.
No remote deployment or supplement collection has occurred.

Source helper SHA256:
26a6824f1282f05e52229d89f88fd8f5d84b845263a6e734465538c9f11eb1c0.
Test SHA256:
c0dc4343f762078e6fd3cbe267a3b4017ac743c1cd0efc46c18f6eb8ce4d5024.
Fresh helper-only source review requested. Its PASS, if obtained, is not
admission of the still-required image replay entrypoint/verifier/plan.
