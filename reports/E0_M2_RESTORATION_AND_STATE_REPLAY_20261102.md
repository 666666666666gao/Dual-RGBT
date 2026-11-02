# M2 TRAIN Cache Restoration And State Replay

## Original Data Restored

The eight original accepted M2 TRAIN shards were found on host25 at
`/data2/gb/Dual-RGBT/runs/protected_e0_m2_identity_full_r1_p25_20261101_s0`
through `s7`. The relevant root is not the separate `/home/gaob` live
training checkout or `/data2/gaob` E0 fine-tuning output.

Before transfer, every source NPZ was rehashed against the original
acceptance. All 882 files passed. Files were copied to identical relative
paths under host26 `/data2/gb/Dual-RGBT`, preserving the originals and
the historical native receipts. No candidate collection was repeated.
The destination auditor independently rehashed every restored file:

| Item | Verified Value |
| --- | --- |
| NPZ files | 882 |
| Compressed bytes | 2807716143 |
| Shards | 8 |
| TRAIN frames in original acceptance | 459435 |
| Candidate sets in original acceptance | 18711 |
| Original fit/component-holdout sequences | 706 / 176 |
| Acceptance SHA256 | 74eb5104ee223c8fa4eec4992c47f1e466efc6f27a9679a95d372fdd521f2522 |

Byte-audit result:
`reports/E0_M2_RESTORED_CACHE_BYTES_P26_20261102.json`.
Auditor source:
`tools/audit_m2_restored_cache.py`, SHA256
`dc1d17d640b8691d3cd3a03ce1cb8370f19742b6690f1bff7f8dad0364fb8c28`.

This restores accepted NPZ bytes. It is not fresh acceptance of the
entire collector, all other historical inputs, or a new training plan.

## Initial Replay Stopped On A Precision Check

The first frozen epoch9 M2 replay exited1 at 2026-11-02 02:45:32 UTC,
before producing a result JSON. The failed source, original source
review and runtime log are preserved under
`reports/source_snapshots/m2_train_state_replay_r1_20261102/`.
The original remote log remains
`reports/E0_M2_FULL_TRAIN_STATE_REPLAY_P26_20261102.log`.

The check incorrectly expected stored collector IoUs to agree within a
small tolerance with a new all-float64 calculation. The collector performs
box-coordinate addition and box-area multiplication in float32 before
combining with float64 GT. Changing that order changes small overlaps.

A concrete affected example: `girlatwindow`, frame201, candidate21.
Box float32:
`[568.8729248046875,234.26434326171875,50.060272216796875,64.24268341064453]`.
GT: `[617,233,48,58]`.
Stored collector IoU: `0.018620869423258454`.
All-float64 IoU: `0.018620568865816003`.
Every NPZ hash still matches acceptance; this is not evidence of corruption.

The revised replay reproduces the collector formula separately for all24
candidates and requires exact `np.array_equal` with saved IoUs. It then
uses the same float64 formula for both current E0 and selected candidate
when comparing proposal quality. Saved labels and frozen classifier are
unchanged. There is no relaxed tolerance or forced PASS.

Local checks extracted the original collector function through AST and
verified exact equality on24000 random candidates, reproduced the actual
failure case and checked identical-box float64 IoU=1. A source review is
required for the revised replay before its actual model execution.

## Verified Full R2 Replay

Separate R2 engineering review passed with0blocking issues:
`E0_M2_TRAIN_STATE_REPLAY_SOURCE_REVIEW_R2_20261102.json`, SHA256
`bde44e0ba32b0d8a7e3fafb2fd2ef06f882e2770286bb5afc38e6eea58869fd2`.
The exact reviewed replay source is
`f4ed4a70cbfb202e1143fd1348073b7fa4e5144d013df8cb391fda0a52ec4bb3`.
Attribution remains UNATTESTED/same-family/provisional; scientific
acceptance is false.

Actual host26 CPU replay started02:57:47 UTC and exited0 at02:58:56 UTC.
All882sequences and18711sets passed exact stored-IoU checks on all24
candidates. Frozen weights were unchanged, CUDA remained uninitialized,
and optimizer updates were0. Both saved fit and holdout classification
counts reproduced exactly, including holdout2747/3450supervised correct,
2685/3318candidate-positive correct and62/132no-match correct.

Raw report:
`E0_M2_FULL_TRAIN_STATE_REPLAY_R2_P26_20261102.json`, SHA256
`5e9aa45d766c5bcf56cf95c4613c2090bf2ad13e957c379fca92ad1c0094e78e`.
The separate R2 log is archived beside it.

Rows describe the original current E0 proposal, before any M2 replacement.
Correct available means at least one observed candidate labeled positive
by the original collector. Harm means reliable E0 becomes IoU<0.5;
recovery means lost E0 becomes IoU>=0.5.

| Partition | E0 State | Sets | Correct Available | Replacements | Harm / Recovery | Mean IoU Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fit | Reliable | 10073 | 9853 | 9801 | 960 harm | -0.075967 |
| Fit | Uncertain | 1212 | 618 | 1013 | not applicable | +0.042371 |
| Fit | Lost | 3477 | 1810 | 2396 | 1108 recovery | +0.260670 |
| Component holdout | Reliable | 2899 | 2845 | 2790 | 326 harm | -0.085247 |
| Component holdout | Uncertain | 250 | 99 | 208 | not applicable | +0.013504 |
| Component holdout | Lost | 800 | 374 | 566 | 179 recovery | +0.191874 |

The reliable holdout replacement rate is2790/2899=96.2401%;326/2899=
11.2453% cross below IoU0.5. Of all2899reliable states,1729 become worse,
1057 improve and113 are equal. A positive candidate often exists even
when E0 is already reliable; presence does not establish replacement gain.

Lost holdout candidate availability is374/800=46.75%. The remaining426
states have no observed positive candidate.179/800=22.375% recover in
this instantaneous replay, or179/374=47.8610% of states with a positive
candidate. This separates deficient candidate coverage from selection.

Summed proposal-IoU delta is+192.487045/14762=+0.013039 on fit and
-90.256553/3949=-0.022856 on component holdout. This is a sampled-state,
set-weighted statistic, not full-video SR or closed-loop tracking gain.
The diagnostic state thresholds are not tuned here;
the state definitions remain reliable>=0.5 and lost<0.1.

## Next Mechanism To Test

1. Retain the original M1 checkpoint; the rejected balanced-loss head
   worsens reliable protection. Existing joint TRAIN diagnostics already
   show missed failures. Measure any new gate with predicted states,
   never GT state supplied to online inference.
2. Give M2 an explicit scored current E0 alternative and TRAIN
   preservation/relative-quality supervision. Existing no-match describes
   absence of an observed positive candidate; it is not a learned KEEP
   preference when E0 is better than other positive candidates.
3. The restored cache contains pooled candidate descriptors, not the
   current E0 dual-modal descriptor or914-dimensional M1 observation.
   Neither can be invented or recovered by renaming candidate0. Any
   required new evidence should be a separately reviewed minimal
   supplement bound to the original E0 parent and paired predictions.
   Do not repeat the full24-candidate collection just to restore bytes.
4. Keep M3 temporal association/reliable memory as a later separate
   intervention after protection and acceptance are measured. The pinned
   official KeepTrack implementation provides a mechanism reference for
   previous/current candidate association; it does not solve current E0
   preservation or prove novelty or gain. Reference provenance is in
   `reports/keeptrack_source_provenance.json`, official
   `visionml/pytracking` commit7eb9e74bd3d40e29dbcec444902237da13de247b.

These are next experiment requirements, not an admitted new collector or
completed module. The current CPU replay supplies no new GPU admission
or automatic TEST promotion. Existing E0 and R15 full evaluations remain
unchanged while the next TRAIN mechanism is prepared.

## Interpretation Boundaries

Replay uses the existing frozen epoch9 head, selected by the original
TRAIN component-holdout plan. It performs no optimizer updates and no
TEST inference or selection. The current E0 proposal is reconstructed
from `predictions[frame]`; candidate generation used the previous state.
Candidate absence, incorrect selection and harmful replacement must be
reported separately.

These are one-step decisions on original protected E0 cost0 trajectories.
Upstream E0 saw the component holdout. M1 features are not present in the
M2 cache; this replay cannot report executable M1-gated outcomes. Future
closed-loop state, recovery persistence, identity and benchmark improvement
are not established by this cache analysis.

Existing eight-sequence joint TRAIN execution already found holdout lost
recognition21/95, and the balanced M1 loss control worsened reliable
interruption. Do not repeat those runs, promote the balanced head or
attribute the entire failure to M2 without separating the measured effects.
See `E0_M1_M2_FIXED_TRAIN_DIAGNOSTIC_20261101.md` and
`E0_M1_BALANCED_TRAINING_20261101.md`.
