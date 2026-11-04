# C2 DropMoE Zero First Policy FIT

The matching full TRAIN replay completed on P25 at
2026-11-04 09:19:32.687557 UTC, native COMPLETE/exit0. Both original wrapper
1806777 and child1806778 have exited. The verifier reports PASS for
882 sequences, 458,553 tracked frames and 91,091 records.

Exact terminal bytes are archived once in
`reports/c2_policy_replay_completed_r1_20261104/dropmoe/zero/`.
Its SHA256SUMS matches all three remote files. Do not rerun replay.
Acceptance SHA256:
`32eed847b952cc429fdc6c15341cfe3340aba8c2d755310198810907fd7786f5`.
Native SHA256:
`7fdf4a2659422093951a64ffb1c4b6c9dccf32f9716544b0b2fc9698a79eb7ac`.

The unchanged reviewed preparer generated a disabled plan under
`reports/c2_policy_fit_r1_20261104/dropmoe/zero/`.
Draft SHA256:
`d3507eb4afddb1eea746cc620906879d914faf05ca5c789fb59526b920d81f75`.
Canonical binding:
`dcf21689127d139c9921bb5d6cdb939d289130a5b387a759b139010e3b45a8f5`.

The read-only live preflight passed exact source/input/executor/four-shard
metadata checks and CPU checkpoint validation. Its witness is preflight.json.
P25 GPU1 had24,247MiB free,12MiB used, no compute process,250W and42C.
Destination disk had931,784,974,336 bytes free.

Plan task: `fit_c2_policy_dropmoe_zero_full_b512_r1_p25_20261104`.
Independent source root:
`/data2/gaob/Dual-RGBT/c2_policy_fit_dropmoe_zero_r1_20261104`.
Run:
`/data2/gaob/Dual-RGBT/c2_policy_fit_runs/fit_c2_policy_dropmoe_zero_full_b512_r1_p25_20261104`.

Before deployment, require a separate exact FIT admission review at
`reports/P2_C2_DROPMOE_ZERO_FIT_ADMISSION_R1_20261104.json`, stage
`P2_C2_policy_refresh_fit`, with the complete inherited source map plus
the FIT entry, exact canonical plan binding and launcher SHA256.
Then the unchanged launcher repeats live admission before launch.
Its local launch witness must use a new path, launch.json.

Recipe: five epochs, B512, workers4, seed2027, preview LR4e-5,
quality/gain LR4e-4, AdamW weight decay1e-4, clip1.0, fixed cost0.05.
Freeze executor; update preview/quality/gain; retain all five checkpoints
and require strict reload. TRAIN only, no dev or TEST.

LAUNCHED is a process-spawn receipt. Verify native RUNNING, actual child,
exact argv, enabled plan/config, precision and GPU ownership before claiming
training started. Initial dataset validation hashes and loads all882 saved
artifacts before CUDA/model/run creation; no early CSV alone is failure.
Observe at estimated180-300second boundaries. Do not repeat admission,
deployment or a failed task ID, and preserve every partial artifact.

After actual five-epoch COMPLETE/exit0, verify890 updates and matching
epoch005 lineage before any separately reviewed tracker evaluation. This
FIT is C2-specific and cannot use the old P1B loader. Engineering review
is UNATTESTED/same-family/provisional. Replay/FIT completion alone does not
establish a benchmark gain or meet the research goal.

## Actual Launch

The separate exact FIT review is SOURCE_PASS/0, SHA256
`ac17a1fb3fa9c291d2d3b2410928dbd45fe8d6a915adbd4c428a14b02e07192b`.
The reviewer passed29 focused CPU tests. Live capacity and source/input
admission repeated successfully. At09:38:44.214266UTC the task became native
RUNNING, wrapper2096742/child2096743. Enabled plan SHA256:
`8d33d391b0bbb0203a98ebac53f249c8440f2ff6bd5e92d8662cbb7f9d450c74`.
The launch witness is launch.json. Do not call the launcher again.
Initial CPU dataset validation precedes CUDA/config/run creation.

At09:41:23UTC actual child2096743 owns GPU1 UUID
GPU-0a03017b-9a59-13c9-be5f-f36fc7ae2508, with1250MiB context,40C and250W.
Actual config bytes equal admitted plan bytes and precision is allfalse.
Raw launch ownership/config/precision/review evidence is archived once in
start_witness/. Do not repeat. Initial CSV/log are empty before first25
updates flush; this witness establishes execution, not an optimizer update.

At09:58:25UTC the first25 actual sequential epoch1 updates are flushed;
all four loss fields are finite. Exact112sources, native argv, plan/config,
review and precision remain matched. Actual child2096743 owns GPU1 with
19422MiB context,100%GPU,70C/250W. Raw first25_updates/ is archived once
beside start_witness/. CSV SHA256
`16530d3b9214a13010664fd6498e442ef9b76e1b1cab8155b58dff4deacebc63`.
No epoch completion or convergence is claimed. Do not repeat this archive.
