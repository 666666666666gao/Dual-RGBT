# P26 SOT FiLM TRAIN Collection Completion

Observed on November 4, 2026, at 07:13 UTC. All four native tasks are
COMPLETE with exit code 0. The final shard, s0, completed at epoch
1793775104.9626975. These are TRAIN integrity records, not benchmark results.

| Shard | Sequences | Tracked Frames | Records |
| --- | ---: | ---: | ---: |
| s0 | 221 | 98,613 | 19,574 |
| s1 | 221 | 103,761 | 20,578 |
| s2 | 220 | 128,483 | 25,541 |
| s3 | 220 | 127,696 | 25,398 |
| Total | 882 | 458,553 | 91,091 |

The executor is the fixed SOT-FiLM C2 epoch-20 checkpoint, SHA256
`dc17b0b4b36e670c85151fb9db664874e33a26f65a2dc53305dc9a516756b2ab`.
All four receipts state `executor_preserved=true` and
`benchmark_evidence=false`. TF32 and cuDNN benchmark flags are false.

Raw metadata archive:
`reports/c2_sot_film_runtime_archive_r1_20261104/s{0,1,2,3}/`.
Each shard contains the unchanged remote plan, native status, COMPLETE
receipt, config, precision and progress JSON files. All 24 local SHA256
values were compared with fresh remote `sha256sum` output and matched.
The archive contains no weights or sequence payloads.

The replay plan
`reports/C2_P26_SOT_FILM_REPLAY_ACCEPTANCE_PLAN_R1_20261104.json` binds each
runtime plan, source plan, source review, completion receipt, native status,
precision receipt, executor and manifest. Its canonical read-only preflight
passed at epoch 1793776413.747113:

- Plan SHA256:
  `0219ad7499f7b4477549ad227465c9a7a0dba7bac2f0cbdbd50b2221b7e28b5e`.
- Witness:
  `reports/C2_P26_SOT_FILM_REPLAY_ACCEPTANCE_PREFLIGHT_R2_20261104.json`,
  SHA256 `56a56118c128b613dcd7f8d8c7809b52af12e93c0df9a9eff82f71311c4f3a65`.
- GPU2 had 24,112 MiB free, 15 MiB used and temperature 37 C; its power
  limit was 350 W. The existing zero verifier was on GPU3.
- All 111 pinned remote source files, four collection lineages and exact
  aggregate counts passed the canonical checks.

R1 preflight belongs to the prior draft with the inherited zero review
path; it is preserved as history and does not admit the corrected plan.
The R2 preflight is read-only. Launch requires the separate exact FiLM
review and another live canonical admission. Replay success must precede
any separately reviewed policy FIT plan. No benchmark, selection or goal
completion follows from collection completion or preflight.

## Follow-Up Scope

Apply the supplied review's execution priority: close each executor's
matching TRAIN collection, replay acceptance and policy FIT before adding
LoRA, temporal tokens or new heads. Compare zero and FiLM under the same
sampling, training budget and policy-refresh protocol.

Keep the supplied targets separate by initialization:

| Line | RGBT210 PR / SR | RGBT234 MPR / MSR | LasHeR PR / NPR / SR |
| --- | --- | --- | --- |
| SOT | 88.7 / 62.9 | 90.9 / 65.6 | 75.8 / 71.9 / 60.2 |
| ep20, project label DropMoE | 91.2 / 65.4 | 92.8 / 67.7 | 77.7 / 73.3 / 61.3 |

The supplied final absolute LasHeR target remains 78.74 / 75.16 / 63.84.
These targets do not establish the upstream identity of the project's
ep20 checkpoint. Do not relabel it as independently verified DropMAE.
Retained fusion and query execution costs require measured parameter,
MAC and timing evidence; selective modality execution alone establishes
no FPS advantage. Historical TEST observations remain exploratory and
cannot select branches, epochs, checkpoints or thresholds.

## Replay Launch

The separate exact review
`reports/P2_C2_SOT_FILM_REPLAY_ACCEPTANCE_SOURCE_REVIEW_R1_20261104.json`
is SOURCE_PASS with zero blockers, SHA256
`e76e562c60e5040e88bcf5c69a2508d33d89a80208f4962a6e1bfe8eff124ff8`.
Attribution is UNATTESTED, same-family/provisional; scientific acceptance
is false. It admits the pinned isolated-root sources, not deployment of
the current dirty local working tree.

The unchanged canonical launcher repeated live admission and launched task
`verify_c2_policy_sot_film_full_r1_p26_20261104` on physical GPU2 at epoch
1793776897.8701897 (07:21:37 UTC). Wrapper/actual child are
2535697/2535698. Native status was RUNNING at 07:21:54 UTC; the child was
CPU-active during initialization. At that first observation it had not yet
appeared as a GPU compute process. Do not confuse reserved physical GPU2
with verified CUDA activity before initialization completes.

The local launch witness is
`reports/C2_P26_SOT_FILM_REPLAY_ACCEPTANCE_LAUNCH_R1_20261104.json`.
Raw remote native status and launch/preflight witness are archived under
`reports/c2_sot_film_runtime_archive_r1_20261104/replay_start/`.
The launcher contracts passed 18 local CPU tests. No verifier source,
training recipe, running zero task or model weight was changed.

P25 DropMoE-zero replay on GPU1 and P26 SOT-zero replay R6 on GPU3 remain
live. P25 GPUs0/2/3 remain idle within the 250 W / 85 C limit; its FiLM
replay is still blocked by missing exact original s0/s2 plan bytes.
Do not schedule FIT until the corresponding native replay completes with
a valid PASS receipt and a new FIT plan receives separate review/admission.
