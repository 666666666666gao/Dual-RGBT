# Fixed P1A DropMoE RGBT234 Full Result

The fixed P1A epoch20 DropMoE control completed all 234 RGBT234 TEST
sequences and116,650 frames, native COMPLETE/exit0. The reviewed official
CPU scorer ran once; the existing matching-profile compute summarizer
then validated full traces/artifacts and calculated route-weighted MACs.

| Measurement | Result |
| --- | ---: |
| MPR | 89.2732589407 |
| MSR | 66.1734310133 |
| Full-run FPS | 17.7517889366 |
| Total parameters, including frozen | 103,374,215 |
| Mean tracked-frame GMACs | 57.4360641327 |

Against the original90.9/65.6 goal, MPR is1.6267410593 points below and
MSR0.5734310133 points above. This fixed pre-P1B control does not establish
that the RGBT234 target pair or the overall goal has been achieved.
It is not a policy-refresh, C2 or M1 result. The result remains
exploratory_test=true/test_eligible=false and does not select a checkpoint,
threshold or branch or alter the predeclared P1B recipe.

FPS includes image decode and initialization under concurrent load, excludes
model loading and is not isolated throughput. MACs weight actual ten-route
counts over116,416 tracked frames, excluding initialization arithmetic and
nonlinear/normalization/ROI/memory/bookkeeping operations. P25 executed the
matching-source fixed-route CPU profile.

Raw score/curves/compute/native/config/receipt/progress/review and operation
logs are archived in
`reports/p1a_completed_scores_r1_20261103/dropmoe/rgbt234/`.
Score SHA256:
`2ece1e10d2a207468fe743d957df8ab796c11bb5c864614d05f87308b64edbc7`;
compute SHA256:
`c06d4184d8e4ebd9ec66224879641dd65dc1296309ac0e84bcbb614a93007d00`.
Do not rerun the scorer.
