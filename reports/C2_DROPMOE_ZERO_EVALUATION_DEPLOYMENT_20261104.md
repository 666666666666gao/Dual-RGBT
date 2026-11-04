# C2 DropMoE-Zero Evaluation Deployment

## Actual CPU Reconstruction

Exact operation review R1 is SOURCE_PASS/0; report SHA256 is
`40ffab8f8b0083e74035738da5d97f790ab74645040e5eb308f99ecc67ed8ef4`.
It admits only the two previously pinned static plan bindings and exact
operator SHA256 `acc5b1ab67c99e119ba2daebc2cf9d4bc6b4dc3cee5006a87ce880899732d904`.
Attribution is UNATTESTED/same-family/provisional; this is engineering
admission, not independent scientific validation.

The operator created the distinct absent root
`/data2/gaob/Dual-RGBT/c2_policy_evaluation_dropmoe_zero_r1_20261104`.
It preserved the112original FIT host-source bytes and supplied only the
five reviewed new evaluation sources. All117source pins, original FIT
documents, manifest/protocol/toolkit and fixed checkpoint hashes passed.
No original root, active source or retained checkpoint was changed.

Actual `load_completed_policy(...,device='cpu')` returned successfully.
The loader authenticated original C2 executor architecture and terminal
FIT/replay bindings, rehashed all five retained weights, checked finite
tensors, unchanged executor tensors and changed preview/quality/gain
modules, and strictly reconstructed C2DualRGBT. The result has103374215
parameters, evaluation mode, gradients disabled and CUDA uninitialized.
This actual model check is separate from the earlier metadata-only archive.

Fixed epoch005 SHA256 remains
`5d1f33e09e9dd28b076bc444d8156340fc2318f2f4d10b3b4dbcfd6d60d40d04`.
Raw deployment output is
`c2_policy_evaluation_r1_20261104/dropmoe/zero/deploy_result_r1.json`;
remote witness is the new root's `reports/cpu_reconstruction.json`.

## Own TRAIN Smoke

Fresh GPU0 runtime admission passed at18:12:03UTC with24247MiB free,
12MiB used,40C and250W power limit. Native smoke wrapper/child
2503559/2503560 started with the exact reviewed command; live CUDA
ownership on GPU0 was observed. Raw admission/launch output is
`c2_policy_evaluation_r1_20261104/dropmoe/zero/smoke_launch_r1.json`.
Smoke genuinely completed at18:12:43.934637UTC, native COMPLETE/exit0
with wrapper exited. Receipt binds1sequence11frames, fixed checkpoint,
exact plan/config hashes, unchanged state, zero optimizer updates and
zero backward calls. All precision flags remained false. Its3.486072494s
timing is smoke-only and is not a full-benchmark speed claim.

## Full LasHeR Launch

The full operator independently authenticated own smoke terminal evidence
and repeated source/input/CPU-witness checks. Fresh GPU0 admission passed
at18:16:10UTC:24247MiB free/12MiB used/39C/250W. Full runtime plan
SHA256 is `67bedd73ac9ef53231d9095dfd7053798cf0cd55887f0ea485c11064e3ea3f82`.
Native task `eval_c2_policy_dropmoe_zero_e005_p25_r1_20261104_lasher`
started at18:16:10.913576UTC, wrapper2511436/actual child2511437.
Initial native status was RUNNING with exact reviewed argv. The registered
scope is LasHeR FIRST,245sequences/220703frames, fixed epoch005/cost0.05.
No benchmark completion or score is implied. Do not repeat launch or score.
At18:18:20UTC the actual child owned GPU0,2258MiB compute memory; GPU0
was59C/155.44W under250W limit. Native state remained RUNNING and the
completed-sequence progress contained3sequences/751frames. Raw native,
child argv, CUDA ownership, GPU telemetry and log-tail evidence is
`c2_policy_evaluation_r1_20261104/dropmoe/zero/native_cuda_start_r1.json`.
No finite-loss/convergence claim follows from inference progress.

Raw deployment, admissions and inspected native/receipt results are under
`c2_policy_evaluation_r1_20261104/dropmoe/zero/`; raw CPU/review/smoke/full
start documents are copied once under its `start_witness/` directory.
The local disabled source plans remain unchanged.

## Other Work

At18:11UTC P25 FiLM replay R3 child2228314 and P26 SOT-zero R6 /
FiLM R1 children2322990/2535698 remained live with their original argv.
No acceptance outputs existed. P25 limits remained250W with observed
temperatures40/42/68/41C; P26 temperatures41/34/74/76C. No verifier
restart, repeat collection, FIT or scoring was performed.

Targets remain unmet. Full TEST remains exploratory and test-ineligible;
CPU reconstruction and TRAIN smoke are not tracking metrics or model selection.
