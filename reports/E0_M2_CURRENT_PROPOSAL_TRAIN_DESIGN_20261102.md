# Current E0 Proposal As An Explicit M2 Alternative

Status: proposed next TRAIN experiment, not source or GPU admission.

## Measured Problem

The verified full frozen TRAIN replay has2899reliable component-holdout
states. M2 replaces2790 and moves326belowIoU0.5. Lost-state coverage is
374/800; correct instantaneous recovery is179/800. Existing identity-set
classification therefore does not establish replacement utility.
The original no-match label means absence of a trustworthy candidate,
not that E0 is better than a positive candidate. Evidence:
`E0_M2_RESTORATION_AND_STATE_REPLAY_20261102.md`.

## Minimal Evidence Supplement

Reuse the original882sequence706/176partition, cost0 parent, initial
templates, sampled valid-frame cadence and accepted candidate NPZ hashes.
Do not repeat24candidate queries or overwrite the accepted files.

Use the original direct E0 tracker in observe-only mode. At each sampled
decision frame, require the returned box, previous box and frame alignment
to match the accepted `predictions` exactly. Maintain an unobserved paired
tracker and check boxes, non-timing routes, history, templates and counters.
The supplement must never commit M1 or M2 interventions.

Use the current E0 box in the correct current search-crop coordinates.
Using the original fixed initial templates, execute both modality encoders
on that current crop and pool the actual E0 ROI with the existing
`primary_candidate_evidence`/`normalized_cxcywh` convention. Save
`e0_features` with shape(N,2,768), `e0_coverage`(N,2), the current image-space
box, crop geometry, original sampled frame indices, parent cache hash and
the original current E0 confidence/action/modality.

Also save the already available914-dimensional `E0Observation` feature
at the same decision frames. It is an observation, not a trained M1 state
or extra E0 forward. This supplies a future diagnostic of the executable
M1 failure gate at the actual M2 decision cadence. It must not be
silently concatenated with candidate features or joined to the old M1
cache by row number.

All encoders and observations finish before reading current TRAIN GT.
No TEST input or threshold selection is permitted. Record two encoder
calls per sampled frame, any observation overhead and complete synchronized
timing. This is a real image pass, not an inexpensive rename of old data.
Its parity and frozen-weight checks must be tested before full execution.
Do not run two observers on a shared model predict hook; own and restore
the one observer explicitly.

Supplement verifier must independently reconstruct coordinates and check
all original/supplement sequence names, frame indices, current/previous
prediction equality, exact parent hashes, source maps, native receipts,
nonintervention evidence and finite array shapes. Preserve original
collection admission rather than treating restored bytes as fresh admission.

## Minimal Learned Decision

Reuse the candidate identity projection/matching network; the benchmark
architecture remains the protected E0 plus M1/M2/M3 responsibilities.
Score the explicit current E0 proposal against the same immutable identity
anchor used for the24candidates. KEEP gets that observation-dependent
score, rather than the original fixed no-match scalar.

Current E0 is always a valid rejection path. A cropped/uncovered identity
ROI must not remove the ability to keep E0 or create all-negative-infinity
action logits. Source review must settle that actual observed coverage
contract before the scorer is specified; do not add hypothetical branches
before the supplement measures coverage.

TRAIN action supervision must distinguish identity overlap from relative
replacement quality. The proposed initial objective is:

- Candidate action is positive only if it has original fully observed
  evidence, IoU>=0.5 and strictly greater IoU than current E0.
- KEEP is positive when there is no such improving observed candidate.
- E0/candidate equality preserves E0. Unknown candidate evidence is not
  silently relabeled a supervised negative.

These rules require review before implementation: multiple improving
candidates can remain positive even when they differ in IoU, and strict
floating-point improvement can include tiny localization differences.
The actual TRAIN distribution should quantify these cases; do not fit
an improvement margin on TEST or silently assert that this objective
guarantees reliable-state protection.

Freeze E0 and start with the existing matching architecture and seed2027.
Predeclare checkpoint acceptance before training. Report reliable harm,
reliable IoU degradation, lost recovery, candidate availability, true
candidate absence, relative selection correctness and per-sequence paired
changes. Original macro candidate accuracy cannot remain the sole
selection criterion. Preserve best/latest/lastfive; no TEST checkpoint
selection.

## Required Comparisons

1. Frozen original head on original cache: already completed, do not repeat.
2. Explicit-E0 scorer with frozen weights in shadow mode, without changing
   tracker state, against the original replay choices on identical states.
3. Trained relative-decision head, same cache/supplement/partition and fixed
   recipe, with the predeclared TRAIN protection/recovery acceptance.
4. Only after actual paired smoke and TRAIN acceptance, separately reviewed
   closed loops using the predicted M1 gate and explicit rejection path.
   Report both missed failures and harmful gate openings.

M1 class balancing is already rejected; no new M1 head is implied by this
proposal. M3 temporal association/reliable memory is a separate subsequent
experiment. Frozen-state replay cannot establish online recovery persistence
or the three benchmark targets.

## Open-Source Mechanism Reference

The archived official `visionml/pytracking` KeepTrack source at commit
`7eb9e74bd3d40e29dbcec444902237da13de247b` associates previous/current
candidate descriptors and maintains candidate identities through
`CandidateCollection`. The relevant archived code is
`reports/source_snapshots/tracking_references_20261030/keep_track.py`,
localize_target_by_candidate_matching, and exact upstream file identities
are in `reports/keeptrack_source_provenance.json`.

This supports a later temporal-association design. It does not supply
current-E0 preservation labels or prove this proposed action objective.
No third-party implementation is copied into a new module here.

## Execution Boundary

Only host26 GPU0/1 may be used. Existing full evaluations retain their
source and GPU assignments; GPU2/3 currently belong to other workloads.
Host25 GPU0 has an unresolved PCIe failure and no new GPU work is
authorized there. New supplement source, plan, verifier and smoke need
their own fresh review with exact actual-host hashes before image/GPU use.
User authorization for training/collection already exists; completing the
technical admission does not require another generic permission question.
