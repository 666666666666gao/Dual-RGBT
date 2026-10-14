# Same-coordinate auxiliary content diagnostic

Status on 2026-10-14: SOURCE_PENDING; GPU admission and formal execution NOT_RUN.
This diagnostic addresses the recorded confound in `GLOBAL_OPPORTUNITY_RESULTS`:
GT-centred crops supply a correct-location prior, so their positive IoU gains
alone do not show that the auxiliary image content improves localization.

Use the completed full global B checkpoint, with 6,250 updates and 50,000 pairs.
Do not load B while it is training. The existing executor and policy remain
frozen. Only physical GPU 1 may run this diagnostic, after it is free and after
the independent source review and actual GPU smoke pass. No training, backward
or optimizer call occurs in this driver. The existing global A/B/C admission
runs provide the separate training-path gradient witnesses.

## State and paired controls

Reuse the same 20-sequence prefix of the frozen 882-sequence training execution
plan as the earlier opportunity diagnostic. A primary-only causal tracker
selects a modality with the trained cheap scheduler, executes STOP, and updates
only the observed primary template. Its own predictions control future search
and history. The reference visits can differ from the earlier A checkpoint's
visits, so A/B aggregate differences are not a paired causal comparison.

At each of the 1,752 valid stride-five states, retain the same prior search
state, templates, history, primary prediction and main features. Recompute the
selected primary once and require its executed box to equal the factual STOP
box exactly. Construct four deployable query geometries plus two diagnostic
GT-centred geometries. Future GT is used only for current IoU/coverage and these
two controls; it never enters the reference tracker or its template updates.

Each geometry receives three content variants:

- TRUE: the actual cropped auxiliary search pixels.
- ZERO_AUX: replace only the auxiliary search channels by zero in normalized
  input space. Keep auxiliary templates, cached main features, geometry and
  token budget unchanged. This is a constant-content intervention, not a
  natural image or a guarantee that the auxiliary path has no learned prior.
- SHIFT_AUX: cyclically shift only auxiliary search pixels by half the crop's
  width and height. Preserve geometry and token budget. This tests spatial
  correspondence within the crop, not a naturally occurring sensor fault or
  an independent distractor-identity experiment.

The six query geometries are LOCAL_MAIN, LOCAL_ALTERNATE, GLOBAL_LOCAL,
GLOBAL_WIDE, GT_CENTER_LOCAL and GT_CENTER_WIDE. Local variants execute 193
heavy tokens; wide variants execute 385. Three variants of each region have
identical coordinates, auxiliary templates, main features and heavy input size.
The main search channels are checked bit-equal before the reader. The existing
reader selects only the auxiliary search channels for auxiliary encoding.

The complete diagnostic expects 8,854 track calls and 31,536 comparison rows.
Report paired TRUE minus each control IoU differences, including the subset
whose target centre lies outside the main search source. Store counts and
sums for subsets, including a possible zero count; do not fabricate a mean for
an empty subset. Full population means for every region/content are also
recorded. One seed and one reference rollout do not establish statistical
stability or a complete tracking performance bound.

## Admission, outputs and interpretation

The new driver and plan have their own source seal. They do not change the
active 30-file training seal or the previous opportunity diagnostic's 31-file
seal. A fresh review must record SOURCE_PASS/zero blocking issues with the
exact plan hash and source map before execution. Reviewer attribution remains
same-family/provisional and routing UNATTESTED.

The first run uses the first sequence's ten track calls and two labelled
states, expecting 36 content comparisons, zero backward/optimizer calls,
unchanged model state and strict checkpoint reload of the existing seven-field
probe. The driver also verifies unchanged input checkpoints and source hashes
after execution. Formal execution uses the same completed B, never smoke weights.

Raw `content_comparisons.csv` contains GT-derived coordinates and stays private
on the server and in local `.codex_tmp`; it must not enter Git. Publish only
safe aggregates, native completion and source/admission evidence after checks.

A TRUE advantage over ZERO_AUX, at identical coordinates, is evidence of a
content contribution in these visited states. It does not remove the GT
coordinate prior in GT-centred controls or prove that the deployed query
proposal finds that region. SHIFT_AUX additionally tests sensitivity to the
spatial arrangement of auxiliary content. These are diagnostic IoUs, not
LasHeR PR/NPR/SR, system latency or proof of three successful contributions.

Evaluate learned candidate coverage and the existing same-budget controls
with the unchanged opportunity diagnostic after B. Full 882-sequence C,
full-dev calibration, full-test scoring and paired system timing remain
required before the intended accuracy/efficiency claim can be made.
