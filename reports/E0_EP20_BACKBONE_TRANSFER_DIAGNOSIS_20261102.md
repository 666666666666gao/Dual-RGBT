# E0 ep20 backbone transfer diagnosis

Read-only source and checkpoint inspection on P25,2026-11-02.
No training, threshold changes or new benchmark inference was performed.
The remaining already scheduled epoch39 LasHeR evaluation completed;
its official scorer and compute summary were run once after full receipt.

## Results and comparison boundaries

All15 scheduled epoch036-040 evaluations across three benchmarks are complete.
See E0_EP20_FIXED_LAST_FIVE_SCOREBOARD_20261102.json for exact bindings.
Relative to the user's reported DropMoE-derived stripped-CADTrack baseline:

| Metric | User baseline | Descriptive five-epoch maximum | Delta,percentage points |
|---|---:|---:|---:|
| LasHeR PR |75.6365|75.5873846215(epoch40)|-0.0491153785|
| LasHeR NPR |71.7728|71.6144248298(epoch40)|-0.1583751702|
| LasHeR SR |60.1257|60.0960017977(epoch40)|-0.0296982023|
| RGBT234 MPR |90.6771|91.0211715721(epoch39)|+0.3440715721|
| RGBT234 MSR |66.8452|67.7770063172(epoch39)|+0.9318063172|

These are descriptive TEST maxima,not a selected deployable checkpoint.
Epoch39 and40 must not be combined into a fictitious single-model result.
The supplied baseline has not been independently reproduced with this evaluator.
The small LasHeR differences do not establish statistical equivalence or
significance; no repeated-seed uncertainty estimate exists.

The archived original E0 control checkpoint is exactly the current training
parent ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c.
Its LasHeR73.7909401597/70.2934346247/58.6172613374 and
RGBT23490.4594404063/68.4442525207 are available in
reports/e0_m1_full_scores_r1_20261101/{lasher,rgbt234}_control.json.
BUT those control configs use cost_weight0.0 versus0.05 for this evaluation.
They are not a controlled backbone ablation; do not attribute their deltas
solely to initialization. The user's stripped-CADTrack baseline and our
selective E0 tracker are also distinct inference pipelines.

## Actual initialization and training

The deployed scripts/finetune_e0_ep20.py matches local SHA
3e28d0e084a38dca38c75ffa4de4020ad10cf60bdbb5568fdee5671840ad848c.
Its initialize() strict-loads the parent E0,then overwrites ALL supplied
backbone,box_head,decode_fuse_search tensors into corresponding E0 modules.
It preserves preview,quality,gain and token_cost.
The source checkpoint d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8
has net_type CADTrack and250 state entries:153backbone/90box_head/7decode.
No expert/router/moe-named entries exist. The deployed ViT uses ordinary
attention and Mlp blocks. This does NOT disprove a different pretraining
origin compatible with the same ViT architecture. This audit cannot certify
the upstream DropMoE/DropMAE provenance from filename or key layout alone.

Direct CPU torch.equal inspection of epoch040 confirms:
- preview,quality,gain tensors remain exactly equal to the parent.
- Relative to supplied ep20,151backbone,54box_head and4decode state tensors
  changed. Replacement and executor learning occurred.

Training uses ten equally scheduled modality/action routes; preview,
quality and gain are frozen. Online predict() uses those heads to choose
the primary modality and argmax(gain-cost_weight*token_cost).
An old policy operating on changed executor features is a concrete
calibration hypothesis. There is no new-policy comparison establishing
that it causes the observed metric differences.

The executor recipe is40epochs/100000updates/20000pairs per epoch/batch8,
initial LR1e-5 and backbone LR1e-6,milestones10/30,gamma0.1.
From epoch31,backbone LR is1e-8 and executor LR1e-7.
The small final rates are a recipe fact,not evidence that increasing LR
would improve tracking. Finite TRAIN losses and a4.254% mean-loss reduction
do not prove tracking convergence,overfitting or optimal stopping.

## Inference path and next discriminating evidence

Remote and local source hashes match:
- dual_rgbt/model.py c7fae6f893797edd10dcb6e625bb304b5a637c80ed1ac404f58d3d223d1b9f7e
- dual_rgbt/tracker.py 406927019ffdcdbc8546755d9f7bff0859aac20766116e4a70f8c89132694fff
- lib/models/cadtrack/vit_care.py ae74beb13742793524bf8c69c2ddc2bcdc175c983ce30d37d32d71be6872e13c

Remote stripped-CADTrack forward encodes full RGB and thermal search streams
and concatenates their search features before fusion and box prediction.
E0 predict instead selects a primary modality; it can stop without the
other modality or encode a64-token auxiliary region,or all256auxiliary tokens.
Its retained pretrained modules therefore operate with a different input
and action distribution from ordinary dense paired inference.

The manual E0 block loop applies the same attention/MLP residual operations
as this deployed plain ViT Block; no missing expert execution has been
demonstrated. Do not diagnose a bypassed MoE solely from that loop.

P26 eval first constructs a factory from ep14,then strict-loads the entire
finetuned E0 net. The factory filename alone is not evidence that ep14
overwrote the finetuned parameters.

A discriminating next experiment would first reproduce the supplied ep20
in its exact stripped-CADTrack evaluator,then compare initializations under
the same E0 code,policy,cost,sampling and training budget. Policy adaptation
must be assessed on TRAIN with a prespecified recipe,not by TEST tuning.
No such new experiment is launched or admitted by this diagnosis.
