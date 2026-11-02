# Context Advantage: Actual Full TRAIN Result

The R2 proposal completed on P25 GPU2 in its isolated reviewed snapshot.
It used real exported accepted TRAIN tensors, not a synthetic fixture.

| Evidence | Result |
|---|---|
| Exact source/draft review |SOURCE_PASS, zero blockers, 163 sources|
| Remote CPU checks |9 passed on each of P26 and P25|
| Accepted real TRAIN tensor export |14762 fit / 3949 held sets, 706/176 sequences|
| Actual GPU smoke |COMPLETE/exit0, 5 updates, strict reload and finite gradients|
| Actual full TRAIN |COMPLETE/exit0, 12 epochs, 348 updates|
| Canonical terminal CPU replay |Completed once, retained8-12/latest verified|
| Qualifying head |None; retain protected E0|

Full plan SHA:
`7d899f37f5b3e6fb762238a78203f32660691f52669e4456b14f42c746df9cca`.
Tensor SHA:
`b79b4a3db2bd7d1c2d639b737a5cd0b7d6b8c10857a59a5a1fbd7af3f2b031f3`.
Terminal acceptance SHA:
`cf596d4a5bbe5bd8dcb09f50403f1177ec5326186c90f904628b89a8a33836cc`.
Latest checkpoint SHA:
`b49bdd553464a64381a2e36e4060df37e4f3dbb648b23397a7faa7ea285340a5`.

The small cached head has 961 trainable parameters. The exported appearance
scores came from 116482 frozen parameters. Full training-engine time was
3.051984 seconds (about .2543 seconds/epoch); native elapsed time was
9.031122 seconds including admission and loading. Peak allocated/reserved
GPU memory was 7675392/25165824 bytes. This is not E0 end-to-end runtime,
tracking FPS, full network parameter count or benchmark MAC profiling.
No padding or duplicate jobs were used to occupy GPU memory.

| Held TRAIN epoch | Harmful reliable changes / 2899 | Lost recoveries / 800 | Reliable macro IoU delta | Qualifies |
|---|---:|---:|---:|---|
|1|1|1|-.0002665400|No|
|2|1|1|-.0002665400|No|
|3|1|1|-.0002665400|No|
|4|1|2|-.0002759159|No|
|5|2|4|-.0006424992|No|
|6|5|21|-.0019186235|No|
|7|7|46|-.0023325621|No|
|8|12|63|-.0019213356|No|
|9|14|66|-.0013027034|No|
|10|13|69|-.0017958637|No|
|11|11|63|-.0014575816|No|
|12|12|52|-.0018108419|No|

The original gates remain zero harmful reliable changes, nonnegative reliable
macro IoU delta and more than179 lost recoveries. All12 epochs failed.
The inventory is descriptive: no rejected epoch is promoted or selected for
online/paired/benchmark evaluation.

Against the prior rejected relative head's epoch12, harmful held changes fell
from419 to12, but lost recoveries fell from169 to52. The current candidate's
overall held sequence-macro delta became positive (+.0071965706), while its
reliable subset still regressed. This is a useful measured change, not success.

Additional actual-tensor attribution is archived in
`E0_M2_CONTEXT_RECOVERY_GAP_AUDIT_R2_20261102.json`, with its exact source.
Of374 eligible held lost states with an observed correct candidate:
187 chose KEEP,135 chose an incorrect candidate,52 recovered. In293,
no correct candidate received positive predicted advantage. On fit,
1476/1810 recoverable lost states had the same nonpositive-prediction issue.
Thus the recovery failure is present on fit as well as component holdout.

Fit observed reliable-degrading actions numbered217471, versus8452
lost-recovering actions. The latter had mean target utility +.7510926905
but mean predicted advantage -.2483331263. Held corresponding means were
+.7308076030 and -.2588583231 (1602 actions). These observations motivate
examining action/state objective balance; they do not prove which loss
change will fix it. Sequence weighting alone does not make these groups
equally influential.

The attribution uses TRAIN GT only for diagnostic groups, never online
inputs/rules. It made zero optimizer updates and initialized no CUDA context.
No threshold, seed or hyperparameter was selected by it. The authorized
existing review chat has been asked for a targeted diagnosis of these real
results. Stop this fixed proposal rather than extending epochs or running
a seed/threshold grid.

Keep latest and epochs8-12, status/log/config/history/receipts and reviews.
No qualifying head means no context online deployment or paired launch.
Any different objective must have its own concrete source/input review and
actual update/run admission, retaining the original split and gates.
