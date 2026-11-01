# M2 Training Preparation And Candidate Availability

## Subsequent Full Training Completion

The preparation and smoke sections below describe earlier stages. The full
TRAIN collection and head training subsequently completed, as recorded in
HANDOFF.md at 2026-11-01 10:25 UTC. On 2026-11-01 the R15 host26 plan and
copied training receipt were directly rechecked; do not repeat collection
or training merely because the earlier preparation text says it is pending.

- Collection acceptance covers 882 TRAIN sequences, 459435 frames and
  18711 candidate sets; acceptance SHA256
  74eb5104ee223c8fa4eec4992c47f1e466efc6f27a9679a95d372fdd521f2522.
- Full training completed 12 epochs and 312 optimizer updates. Its receipt
  runs/protected_e0_offline_m2_identity_full_r2_p25_20261101/COMPLETE.json
  has SHA256 88b2b566cebe9cff9754736d385bc2162b10702850ffaf1e99a96f65400e13f4.
  Strict reload is bit-equal, with 116482 parameters and E0 not loaded.
- Best TRAIN component-holdout positive top1 is 0.796231884057971, at
  epoch9. The R15 integration binds that best checkpoint SHA256
  7e7df5d25d3e8667fe38a4ca8f6ec3662308ffcb52a09bd16aecaab59a9cb4be.
  The copied receipt records 5.229525 seconds of head-training wall time;
  this excludes the expensive upstream candidate collection.
- R15 E0+M2 full TEST inference is running on host26 GPU0/1. This
  standalone training metric is not tracking PR/SR or semantic identity
  proof. Upstream E0 saw the component holdout; TEST remains exploratory
  and ineligible for checkpoint/threshold/seed selection.

## Observed Problem

The existing identity prototype's `identity_set_loss` accepts one target index.
The actual collector can produce several positive candidates per set and uses
`-1` for ambiguous or unobserved candidates. Assigning one positive and treating
the other candidates as negatives would contradict these saved labels.

The new standalone trainer instead minimizes
`logsumexp(known logits) - logsumexp(positive logits)`. Unknown labels are
excluded from both terms; sets without any positive label do not enter updates.
Inference metrics still choose among all observable candidates, including
ambiguous ones, and separately count predictions of ignored or negative items.
No-match is a trusted-match decision within this executed set, not a claim that
the target is absent from the image.

The authors' open-source MIL-NCE implementation provides a related example of
summing positive probability mass before taking its logarithm. Source:
https://github.com/antoine77340/MIL-NCE_HowTo100M , commit
`b282b1534324b3ac9960eebf89312c9a8a912eec`, `loss.py`.
An exact source snapshot is `reports/source_snapshots/mil_nce_b282b153/loss.py`,
SHA256 `865657380fb8436011164d406776eafe11ff0a6696b9bfdaab5f2ecea0c5c6fb`.
Our supervised candidate-set task and ignore mask differ from its video-text
batch construction. This reference establishes neither novelty nor tracking gain.
The earlier pinned KeepTrack source remains the candidate-association reference;
we have not reproduced its temporal association mechanism.

## Partial TRAIN Evidence

At 2026-11-01 06:14:44 UTC, the saved descriptive snapshot contains 110 completed
sequences. It is not full collection acceptance and does not contain TEST results.
Full evidence:
`reports/E0_M2_IDENTITY_PARTIAL_CANDIDATES_P25_20261101T0615.json`.

| Component Partition / E0 State | Sets | Positive Candidate Available | No-Match | Ambiguous |
| --- | ---: | ---: | ---: | ---: |
| Fit / reliable | 1197 | 1172 | 12 | 13 |
| Fit / uncertain | 97 | 42 | 3 | 52 |
| Fit / lost | 340 | 142 | 75 | 123 |
| Holdout / reliable | 306 | 297 | 1 | 8 |
| Holdout / uncertain | 19 | 11 | 0 | 8 |
| Holdout / lost | 69 | 59 | 2 | 8 |

E0 state uses its saved prediction overlap: reliable >=0.5, lost <0.1,
uncertain otherwise. In this partial sample, positive-candidate availability
on fit lost states is 142/340 (41.76%), versus 1172/1197 (97.91%) on reliable
states. Overall accuracy could therefore hide the very cases recovery needs.
Partial sequence order and component distributions differ; the holdout was
also seen upstream by E0. These counts do not establish generalization,
learned identity accuracy or recovery success.

After full collection, repeat this stratified audit and report both candidate
availability and learned matching accuracy. If availability remains limited on
lost states, increasing head training alone cannot supply missing candidates.
Do not change the currently collecting candidate policy mid-run or select it
using TEST results.

## Verification And Review

New files:
`tools/verify_e0_identity_collection.py`,
`scripts/train_e0_identity.py`,
`tests/test_train_e0_identity.py`.
Six focused CPU tests pass locally and on host25 Python3.7. The three real
collector TRAIN smoke files pass reconstructed labels and geometry checks:
3 sequences, 93 frames, 6 candidate sets; five sets have positive candidates
and one remains ambiguous. This is not full training.

R1 source review reproduced a concrete gap: modified query geometry, global
crop, refinement crop, or jointly modified anchor geometry/normalized box
could pass the first verifier. No actual cache corruption was found.
Original R1 source bytes and blocked review are preserved. R2 independently
reconstructs initial geometry from the initial target, the local crop from the
previous prediction, all nine global crops from actual RGB/TIR image headers,
and both refinement crops from their scored parents. It then checks all query
geometries exactly. New corruption checks and the three real smoke caches pass.
Image headers are checked; this is not a hash audit of every image pixel.

R2 proposals are in `reports/e0_m2_training_p25_r2_20261101`, with
`launch_allowed=false`. Admission requires a fresh engineering source review
for `M2_identity_training`, five real GPU updates and strict reload, then all
eight complete native collection shards with every saved artifact and label
verified: 882 sequences, 459435 frames, 18711 candidate sets.

The proposed full head training is 12 epochs, batch512, AdamW LR3e-4,
weight decay1e-4, seed2027, with best/latest and final five numbered weights.
Best uses TRAIN component-holdout top1 positive fraction, never TEST selection.
The protected E0 and M1 weights are not loaded or changed by this training.
M2 has 116482 parameters. Its linear-layer MAC count per24-candidate set is
5704704, excluding normalization/nonlinearities and the24 real E0 encoder
queries; this is not a whole-system MAC measurement or FPS result.
Actual M2 training duration, convergence, online recovery metrics and integrated
FPS/MACs remain unverified.

## Actual Update Smoke

R2 engineering review is SOURCE_PASS/0, stage `M2_identity_training`,
target25 `/data2/gb/Dual-RGBT`, 130-source map and two exact draft hashes.
Report SHA256:
`7e6f57e68cbd621ae3b6b0d28fe7bcd47c825a2aa2fcc55cacabe53b6778e68b`.
At06:19 UTC the native task
`train_protected_e0_m2_identity_smoke_r2_p25_20261101`, wrapper1045567,
was COMPLETE/exit0 after22.20 seconds. It performed five genuine GPU updates
on five supervised sets among six real TRAIN prefix sets, changed parameters,
and passed bit-equal strict save/reload. E0 was not loaded or updated.
Its supervised top1 result was0/5 after these five mechanical updates.
This validates the execution chain, not useful matching accuracy.

Plan SHA256:
`19f55454a4b93863a8692a3389b723afab7fdc8a480dd7969b5f70c86d3d7ff7`;
receipt:
`8de8d59b3273f3cd90e32b51c7424f367a96b0bf741997dbd2a823a4b7ff2c75`;
native status:
`61412793e496a677b114f05fe28cf26966541544ad66853fd1413d9338993842`.
The full trainer's actual `verify_update_smoke` accepted these raw receipts
on25. Config/status/log/receipt/plan are archived in the R2 folder/runtime;
the smoke weight remains remote. Complete882-sequence acceptance and the
12-epoch M2 training have not started because collection is still running.
