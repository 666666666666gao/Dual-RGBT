# C2 SOT Zero Replay Completion and Policy FIT

P26 native task `verify_c2_policy_sot_zero_full_r6_p26_20261104` completed
with exit0 at2026-11-04T20:39:03.884046Z. Wrapper2322989 and child2322990
have exited; GPU3 was released. The matching TRAIN replay receipt is PASS
for882 sequences,458553 tracked frames and91091 records. This is an
engineering gate, not tracking accuracy or benchmark success.

The unchanged terminal bytes were archived once at
`reports/c2_policy_replay_completed_r1_20261104/sot/zero/`.
All three local files match their remote bytes and SHA256SUMS:

- acceptance.json:368dceb54dc6355295ac4a64796d71ee0bb3388a5918df6c7d421ae430d80a64
- native.json:cdec48b9f7b7490e96945e6cf71f3c6a9c9e9b5a11b1aa69d90c65a7bc1bb5e8
- run.log:33ea4cbf64602df684e73d023de1c55e97aeb10211a67a189fc5cd96976ede6d

All111 original remote source hashes and the four canonical collection
plan hashes were rechecked before archiving. The existing preparer's
validation passed full totals, executor, native argv and four-shard
bindings. Do not rerun collection, replay or this terminal archive.

The disabled local FIT draft is
`reports/c2_policy_fit_r1_20261104/sot/zero/plan.json`, SHA256
`2fc6022b96780e9d4c9d1750ae1e5b0fd4daa0fc159e87ca566e4f0308735a81`.
Its canonical binding is
`d0277effd225b1fa53fff32e92bc825c3ba94520f164fddeb2a2d4ce9e1e8a34`.

Planned task:`fit_c2_policy_sot_zero_full_b512_r1_p26_20261104`, GPU3.
Independent source root:
`/data2/gb/Dual-RGBT/c2_policy_fit_sot_zero_r1_20261104`.
Run parent:`/data2/gb/Dual-RGBT/c2_policy_fit_runs`.
The112-source map retains111 original host sources and adds only the
existing C2-specific policy FIT entry. Do not substitute dirty local cores.

Recipe remains five epochs, batch512, workers4, inherited seed/cost/stride,
preview LR4e-5, quality/gain LR4e-4, AdamW weight decay1e-4 and clip1.0.
Executor stays frozen; preview/quality/gain train. Keep all five weights
and strict reload. No dev/TEST training or selection is enabled.

Read-only launcher preflight passed at20:42:19UTC, including exact
source/input/native/four-shard metadata and actual CPU executor checkpoint
validation. GPU3 had23956MiB free,169MiB used,54C,350W limit and no
compute owner; destination disk had6811089559552bytes free. Witness:
`reports/c2_policy_fit_r1_20261104/sot/zero/preflight.json`.
Separate exact FIT engineering review R1 is SOURCE_PASS/0 at
`reports/P2_C2_SOT_ZERO_FIT_ADMISSION_R1_20261104.json`, SHA256
`15495855aac7506375cbbda5f83fb9f3c7a8fa515b438cfd5c528cd9786de625`.
It binds all112sources, the unique FIT plan binding and launcher SHA256
`5eecefabfbce353f0accdf2dad11bdde504eecb66414a23d86f0720995908ec1`.
The reviewer independently passed29 focused CPU contracts. This admits
only the exact engineering source/plan, with no execution bindings.
Require exact source/input/plan/review binding and fresh canonical runtime
capacity before launching. Review attribution remains UNATTESTED,
same-family/provisional; scientific acceptance is false.

At20:46UTC the P25 fixed C2-zero LasHeR evaluation remained RUNNING
at140/245 completed sequences and126602/220703 frames, with a fresh
trace. Both FiLM replay verifiers remained RUNNING without final receipts.
Preserve those tasks. No benchmark scoring or RGBT gate was released.

## Actual Native Launch

The unchanged launcher repeated live admission at20:47:48UTC. GPU3 still
had23956MiB free,169MiB used,53C,350W and no compute owner; exact
source/input/native/four-shard/CPU executor checks passed again. It copied
only the pinned original P26 host sources plus the reviewed FIT entry to
the new isolated root and validated the deployed trainer's enabled plan.
Launch witness:`reports/c2_policy_fit_r1_20261104/sot/zero/launch.json`.
Enabled plan SHA256:
`8173da78bf515120c800d2dd44251da72cf98a1f4c30afe15d49279408fd07f7`.
Do not invoke this launcher or task again.

Native RUNNING started20:47:50.139473UTC, wrapper3458315/child3458316.
At20:48:37UTC both were live and the actual child argv matched the native
command. It was initializing before config/precision creation and before
CUDA ownership. Dataset validation precedes those stages; no early empty
log or absent CSV proves failure. Actual CUDA/config/precision and
optimizer progress still require later observation. Preserve all partial
evidence and use180-300second observation boundaries. No FIT completion
or benchmark success is claimed.

At20:51:51UTC actual child3458316 owned GPU3 UUID
`GPU-a594c014-6f57-08e1-3b65-06e99b6e312a` with1252MiB CUDA context,
53C and350W power limit. Both processes were live and exact argv matched.
Actual config bytes equal the admitted plan bytes; precision is allfalse.
All112 deployed source hashes and review SHA matched. Seven raw files
plus observation.json were archived ONCE under the draft's
`start_witness/`; SHA256SUMS authenticates all eight files. Do not repeat
the startup archive. Empty initial steps.csv/run.log precede the first
25-update flush; this is execution evidence, not a counted optimizer
update, completed epoch or convergence result.
