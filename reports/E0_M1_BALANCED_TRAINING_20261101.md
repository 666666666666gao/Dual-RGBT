# M1 Fit-Class-Balanced Training Control

This is one actual same-architecture M1 training experiment motivated by
the completed fixed TRAIN diagnosis. It is not a new module, E0 fine-tune,
TEST threshold search or proof of closed-loop tracking improvement.

Original882TRAIN cache:706fit sequences/71843states,176component-holdout
sequences/19248states. Fit class counts reliable48570/uncertain5749/
lost17524. Labels retain IoU>=0.5reliable and IoU<0.1lost. Upstream E0
saw both partitions; holdout is not independent benchmark validation.

Only the loss changes: class weights N/(3*n_class), from fit only, feed
the training cross-entropy. Keep914->128GELU->3head/117507parameters,
seed2027, AdamW LR0.001/weight_decay0.0001, batch4096,12epochs/216updates.
Fit-only normalization is unchanged. Component-holdout macro recall
selects best; keep last5numbered epochs,latest,best. Do not deploy a new
head into online evaluation merely because a classification metric rises.

## Admission And Launch

Review reports/E0_M1_BALANCED_TRAINING_SOURCE_REVIEW_R1_20261101.json
SOURCE_PASS/0, SHA256
f8cca95feda705e610959f47d424c10ee87b87bd8370bc41155006db9954e1e3.
Review is same-family/provisional, actual_modelUNATTESTED, not scientific
acceptance. Exact draft SHA256
22d1fc5f2e1297a2bb36c474a58756233176d349376e8769a05ef62e610db356.
Final plan SHA256
30d94eadee0fc604626717add0d6e82a9cc67a3ef498502f27cfeb455b64196b,
only launch_allowed differs from draft.

The sole historical collection-source mismatch was the task wrapper.
Its exact Git8103296CRLF bytes,cc2b87b3..., are preserved at
reports/source_snapshots/m1_balanced_train_r1_20261101/scripts/run_task_four_gpu.py.
The loader explicitly verifies that snapshot for historical provenance.
The live wrapper is unchanged. Previous loader ea9efd94... is preserved
remotely as train_e0_reliability_pre_balanced.py in that snapshot directory.
Neither collection plans nor receipts nor caches were changed.

After minimal deployment,9remoteCPUtests passed. Canonical149source/
20input admission and complete8shard loader verification passed, with
2812audited input hashes. No recollection or model environment rebuild.

2026-11-02 00:20UTC launched on26GPU1:
train_e0_m1_balanced_r1_p26_20261101, wrapper613551/child613561,
run runs/protected_e0_offline_m1_balanced_r1_p26_20261101.
Completed at00:20:46UTC, native COMPLETE/exit0, wall33.214seconds including
cache verification. Actual216updates changed the head; strict reload passed.
First cached-head epoch1.548seconds, later epochs0.058-0.068seconds.
This is not E0 image-training speed. Only26GPU0/1 are authorized;
GPU0LasHeR R15 continues,25temporarilyunused.

## Completion Checks

Require native COMPLETE/exit0,216actual optimizer updates, changed
head tensors, finite gradients/loss, strict reload equality, exact
source/input/plan/review/config/native bindings, full receipt/artifact
map, and last5numbered epochs plus best/latest. CPU comparison tool
replays both original and new best heads on identical full caches.
Report all three class recalls and reliable interruption, with epoch
and raw confusion matrices, even if macro recall improves.

## Verified Results

Full CPU replay on the same accepted caches passed all provenance checks
and exactly reproduced both models' stored classification counts.
Raw evidence: e0_m1_balanced_training_r1_20261101/comparison.json,
runtime/config.json, runtime/history.json, runtime/COMPLETE.json,
status.json and run.log. All2812cached inputs were verified.
Best epochs are original5 and balanced1, selected on component-holdout
macro recall only. Lastfive epochs008-012 plus best/latest remain on26.

| Component-Holdout Metric | Original | Balanced |
| --- | ---: | ---: |
| Macro recall | 53.112501% | 60.097712% |
| Lost recall | 42.384271% | 43.504231% |
| Reliable interruption | 13.189174% | 30.853501% |

True rows/predicted columns are reliable, uncertain, lost.
Original confusion: [[12124,274,1568],[605,381,278],[1914,401,1703]].
Balanced confusion: [[9657,2370,1939],[227,855,182],[1110,1160,1748]].
Uncertain recall improves from30.142405% to67.642405%, while reliable
recall falls from86.810826% to69.146499%. Most macro improvement comes
from uncertain recall. Lost recall rises only1.119960percentage points,
and reliable interruption worsens17.664328points.

Do not promote this head into the online tracker. The class-balanced loss
control does not satisfy reliable-state protection and demonstrates no
closed-loop benchmark improvement. Further TRAIN work must address this
measured tradeoff rather than assume class balancing solved M1.

Native SHA256:90e960f40828bb84932f6737a28145ad5e322d8c502bdf61a37b157ceb192fd5.
Receipt SHA256:b4eb0b85f0dfdc8d7708b04204e140ce40da1f142cbb075192eb782e47f03ed9.
Best SHA256:f26c53c80b883e34fa35a161c234b9b82a9a2e426be10b8020418f5738257c14.
