# Protected E0 Redesign: Evidence And Execution

## User Steering

The pasted redesign request supersedes further gain-only loss/K/cost grids.
Existing live jobs continue without interruption. New capacity prioritizes
protected-E0 controls, failure sensing, target-identity evidence and separated
state/memory commits. GPU occupancy and a lower training loss are not research
success criteria.

The active goal first requires LasHeR PR/NPR/SR75.8/71.9/60.2, RGBT210
PR/SR88.7/62.9 and RGBT234 MPR/MSR90.9/65.6. The pasted report additionally
names a higher LasHeR objective78.74/75.16/63.84. Preserve both tiers explicitly;
never change the reference to claim success.

No new dev is admitted by this implementation. The pasted R1-R5 matrix mentions
complete97dev, but the standing direct instruction says no dev. Use TRAIN
component holdout for learning/selection, retain complete test as explicitly
exploratory reporting, and record that a genuinely untouched final evaluation
remains unavailable. Repeated test observations are not independent final proof.

## Protected References

- E0 checkpoint: `runs/stage_c_all882_20261013/latest.pth`,
  SHA256 `ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c`.
- Historical cost0 LasHeR PR/NPR/SR:
  74.74970272036138/71.06767196147948/59.08900667625745, as archived in
  `reports/E1_LASHER_TEST_PRELIMINARY_20261017.json`.
- FP32 reference cost.05 results are a different configuration. Preserve
  `reports/E0_REFERENCE_THREE_BENCHMARK_FINAL_RESULT_20261018.json` and its
  audit warnings; do not merge benchmark columns from different configurations.
- `reports/E0_NUMERIC_REPLAY_RESULT_REVIEW_20261027.json` shows three archived
 11-box prefixes reproduce different historical/reference boxes when toggling
 cuDNN TF32. It does not establish the original whole-run precision settings
 or causally explain the entire full-run gap. New engineering parity uses
 TF32-disabled FP32 and does not claim historical high-score reproduction.
- The currently running unchanged21e07-parent full-test control must finish
 before assigning a full-test gain-head delta. It is not an E0 control.

## Confirmed Implementation Differences

`dual_rgbt/model.py::gain_predictions` uses candidate-specific768-dimensional
primary token features within1690-dimensional gain inputs. In
`dual_rgbt_global/candidate_policy.py::candidate_gains`, the primary deep
representation is globally averaged; candidate-specific evidence includes
32-dimensional auxiliary-preview region averages within955 inputs.

E0 `correct` scatters auxiliary evidence into the original16x16 grid. The
candidate path uses `LocalEvidenceReader` in query coordinates. These are
simultaneous changes in representation and correction geometry, not an isolated
gain-loss intervention. Neither difference has a measured share of the total
regression yet.

Full-test drift statistics and the verified TRAIN reliability diagnostic
support separating localization confidence from failure sensing. They do not
prove template contamination causes all drift. Existing blanket-freeze negative
evidence remains binding; do not repeat it as the proposed remedy.

## Implemented First Step

`dual_rgbt/e0_observation.py` adds a read-only E0 observer:

- Calls the existing predictor once and returns the identical output object.
- Does not replace local correction, route selection, box clipping, template
  updates, history, counters, checkpoint loading or original network sources.
- On sampled states only, reads a3x3 primary-feature ROI at the final predicted
  normalized box. Explicit coverage marks points outside the primary crop.
- Stores914 features: preview pools/descriptors128, history8, primary ROI768,
  coverage1, score statistics3, predicted box4, modality identity2.
- Does not infer target identity from IoU or call the descriptor an identity
  classifier. It is input evidence for future TRAIN-supervised reliability.

`scripts/check_e0_observation_train.py` checks the first31frames of three TRAIN
sequences against direct E0, with observer disabled and enabled. It compares
unrounded tracker-returned boxes (after original clipping), all non-timing route
fields, dynamic templates, history and both counters
at every step; model weights remain frozen. TRAIN labels are read only after the
online prediction/update. This is90distinct tracked positions, not complete
trajectory parity, full882 data collection or a validated new module.

CPU tests cover identical call arguments/return identity, one predictor call,
read-only feature snapshots, ROI locality and explicit out-of-crop coverage.
Three tests passed locally and on the accepted26Python3.7 environment.

The original E0 model/tracker sources remain byte-identical to the protected
reference. New files are additive under `/data2/gb/Dual-RGBT`; the17 existing
workers continue in their original25root or sealed26snapshot.

## Actual Engineering Result And Full TRAIN Draft

At2026-10-31 01:20:58UTC, native wrapper3851521 completed on26 GPU3 with
exit0 in59.666960seconds. Three TRAIN prefixes passed observer-off/on equality
at90distinct tracked positions,270tracking calls in total. The returned boxes,
non-timing routes, templates, history and counters are identical. All265model
state tensors remain unchanged; zero backward calls and optimizer updates.
The collection contains90states of914features, not a trained M1 model.

`reports/E0_OBSERVER_TRAIN_PARITY_VERIFIED_20261031.json` verifies122source
files,189inputs and all3result artifacts against their bound hashes.
PlanSHA256 `2d41aa7524028efabc3be8b1c4777f7512b0cbb3250d98177d021acaff1b43be`;
resultSHA256 `9d2406fdeb1516e928cf7f9499a797b4ba9618b2d56d2d86a98b346be2dd5004`.
Pre-clipping predictor tensors are not compared. Prefix parity does not prove
complete-trajectory equality, historical TF32 replay, or any accuracy gain.
The fresh supplemental engineering review has0blockers but remains same-family,
provisional and actual-model-UNATTESTED; scientific acceptance is false.

`scripts/collect_e0_reliability_train.py` is implemented and compiled on actual
26Python3.7. It pairs a direct-E0 step with a separately initialized observed-E0
step at every frame, compares returned boxes and complete online tracker state,
and samples TRAIN-only reliability features/labels every fifth valid frame.
The existing706fit/176component-holdout partition is reused. All882sequences
have at least5eligible sampled states; the manifest contains459435frames.
The component holdout was seen by upstream E0 training and is not an untouched
generalization benchmark. Five focused CPU tests pass locally and on26
(three observer tests plus two shard/scope tests).

The actual-root draft `reports/E0_PAIRED_TRAIN_COLLECTION_DRAFT_20261031.json`
binds123sources and190inputs, including completed prefix/native evidence.
Eight disjoint shards contain111/111/110/110/110/110/110/110sequences and
53892/49472/63885/62033/44942/54510/64818/65883frames.
Full image paths and labels are manifest-bound; only186prefix images have
individual pixel-file hashes. Do not claim every full-TRAIN image was hashed.
The draft is explicitly `full_collection_allowed=false`. The new collector was
added after the completed observer review and is NOT covered by that review.
Its independent source review and capacity admission are still pending; no
full-TRAIN collection or M1/M2/M3 training has been launched.

At01:32UTC, all17existing full-test workers remain live with fresh trace writes.
The unchanged21e07parent has40/245sequences and30871frames, still no full score.
Numbered weight inventories are25:46files,26sealed:22files,26canonical:0files;
no directory group exceeds five numbered epochs. This does not count best/latest
weights or prove any existing weight unreferenced. No speculative deletion.

## Next Admission Gates

1. R1 requires real engineering-prefix parity first, then paired complete TRAIN
   trajectories with the same original E0 checkpoint, initialization, precision
   and numerical settings. Observer-prefix equality alone cannot satisfy R1.
2. M1 training must predict localization/failure reliability from TRAIN data,
   with labels covering stable, lost/high-score, recoverable and modality-conflict
   states. Recoverability labels need actual cloned-state executions; a lost
   frame or an unselected request is not automatically recoverable/negative.
3. M2 must restore candidate-local deep features with coverage masks and a
   declared initial/trusted identity anchor. Compare candidate sets including
   STOP; four random family samples cannot establish full-pool decision quality.
4. M3 must separate provisional output from memory commit. Test H0/H3/H5 future
   TRAIN consequences from the same cloned state, with no future image/GT access
   during deployment. Do not enforce overlap with a known-lost primary box.
5. Initially freeze protected E0. Later joint readout/last-block adaptation needs
   matched TRAIN budgets and trusted-frame E0 consistency, with changes, total
   parameters and genuine update counts recorded.
6. Complete paired evaluations and full-minus-one ablations must improve over
   protected E0, not merely a failed SR51 parent. Preserve checkpoint/config
   identity for every PR/NPR/SR tuple and all dataset-specific variants.

## Review Boundary

The installed scientific audit route requests `gpt-6-astra`, which is not exposed
by the available native agent schema. Record that route as unavailable and do
not claim its acceptance. Supplemental native engineering source review is
same-family/provisional/actual-model-UNATTESTED and cannot substitute for formal
scientific result acceptance. No new mechanism is a validated innovation yet.
