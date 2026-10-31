# M2 Candidate Identity Prototype

## Implementation And Limits

Added standalone `dual_rgbt/e0_identity.py`, not imported by the E0 model,
tracker, observer, collector or M1 trainer. No real data training or deployment
has occurred. It is not a completed global recovery module or validated novelty.

`primary_candidate_evidence` reads each candidate's deep3x3ROI from the existing
16x16x768 primary grid using the existing observer sampler. Boxes are normalized
cxcywh in that grid's crop, not full-image xywh or query coordinates. It returns
one768-vector and sampled coverage fraction per candidate. Full-image proposals
outside this crop need their own actual encoder evidence; zero padding must not
be called an observation. No global proposal generator or query executor added.

`initial_identity_anchor` describes a first-frame RGB/TIR anchor using the frozen
E0 encoder with its existing templates and a supplied256search crop. Caller must
convert the allowed initial target to normalized cxcywh correctly. The helper
requires all nine sampled ROI points covered and returns detached1x2x768.
It makes two additional encoder calls at initialization. This is a real planned
cost, not free reuse of the32-channel preview descriptors. The function was only
tested with a synthetic encoder, not run with the real E0 checkpoint/GPU.
No trustworthy online anchor update or template memory policy is implemented.

`E0CandidateIdentity` has116482parameters:
- Shared LayerNorm768 and768->128GELU projection for anchors and candidates.
- Per-modality comparison of absolute difference and elementwise product,
 256->64->1GELU match readout.
- Scores average only modalities with all nine ROI sample points covered.
  A candidate with neither modality covered receives negative infinity.
- A separately learned no-match logit competes with all supplied candidates.
  Index0 means no trusted identity match; it is NOT STOP, retaining E0, a
  recovery action or a tracker state transition.
- Variable candidate count and permutation-equivariant scores. No GT, future
  images, cost threshold or test label enters forward.

The new parameter count is measured by summing instantiated head parameters.
Source-derived linear MAC count for a single-frame N-candidate call is
196608+229504*N, excluding LayerNorm/GELU/normalization/pointwise arithmetic and
ROI sampling. This is analytic head-only accounting, not GPU profiling, whole
model MACs or FPS. Initialization's two E0 encoder calls are additional.

`identity_set_loss` takes externally supplied TRAIN target assignment:
0=no-match,1..N=known target candidate. It rejects a target index whose candidate
has no observed modality. Unexecuted/unlabeled candidates cannot automatically
be assigned negative labels. Current M1 IoU-only cache has neither these
candidate features nor identity-set supervision and cannot train this head.
No synthetic loss reduction establishes real identity discrimination.

## Evidence

Five synthetic CPU checks PASS locally and on canonical26 Python3.7 with CUDA
disabled: ROI locality/unobserved coverage; two-modal initial encoder reads and
unchanged inputs; no-match competition; candidate permutation/ignored unobserved
modality; set loss updates and strict reload. All15E0 tests PASS locally.
These do not test the real initial-coordinate mapping, real encoder anchors,
full candidate generation, calibrated identity, recovery or benchmark accuracy.

SourceSHA25689b15dea17514e5516dfe43e76295662ae47da8f98ea590e3fc5565925278669.
TestSHA2568e0b3cc7fbcaeb178f182ce19bb75959c945225cebd227558ed421106893aa3a.

Archived KeepTrack code was consulted directly at the already pinned source:
`reports/source_snapshots/tracking_references_20261030/keep_track.py`,
upstream visionml/pytracking commit7eb9e74bd3d40e29dbcec444902237da13de247b.
Its extract_descriptors_and_keypoints/extract_matches and candidate collection
implement temporal candidate association. This prototype uses fixed initial
dual-modal anchors and is not a reproduction of that association mechanism.
No novelty or effectiveness claim follows from consulting that source.

## Admission And Next Required Work

Fresh source review is required before real use. The mandated reviewer model
route remains unavailable; no review-chat message has been authorized or sent,
no substitute reviewer was used. Existing observer approval cannot cover this
new file. Full paired E0 TRAIN collector remains unlaunched/review-gated.

First complete paired E0 TRAIN collection/M1 and inspect true failure coverage.
For M2, declare and review a separate collection that obtains initial anchors,
actual candidate-local evidence from each executed crop, geometry/coverage,
hard distractors and target-set assignments on TRAIN only. Then train and
evaluate complete candidate sets with no-match, including absent-target cases.
Do not wire this untrained head into tracking or label it recovery-ready.

At04:03:31/04:04:07UTC old utilityE6/E7/E9 are COMPLETE0;25 has onlyutilityE10
live235seq, GPU0/1/3 idle2MiB, GPU2oneworker.26elevenworkerslive,
freshtraces<=11.9s.12total GPUchildren, no unexpected exit. Existing scorers
remain attached. No new old-route jobs or new-route GPU launch.
