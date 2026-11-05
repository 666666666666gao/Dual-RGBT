# P26 SOT-FiLM Replay Completion and FIT Preflight

## Replay Terminal

Native task `verify_c2_policy_sot_film_full_r1_p26_20261104` completed
with exit code0 at2026-11-05T02:44:29.526356Z. The wrapper and child
(2535697/2535698) exited and GPU2 was released.

The exact matching acceptance is `PASS` for882 sequences,458553 tracked
frames and91091 records. CUDA initialized; `benchmark_evidence=false`
and `policy_fit_admitted=false`. This is full TRAIN collection acceptance,
not FIT admission, tracking accuracy or benchmark success.

At2026-11-05T02:58:05.326485Z the matching native terminal and acceptance
were archived once at
`reports/c2_policy_replay_completed_r1_20261104/sot/film/`. The archive
contains the exact acceptance, native status and run log plus an observation;
all four SHA256SUMS entries verify. The 111 original source hashes and
four exact shard-plan hashes were verified during the archive.

- Acceptance SHA256: `5741c145c01bdc0bf4c41553c1db7747fd2b876e33080b61b7f7ec7a731a9445`
- Native status SHA256: `8e29248fbb144606f728fe625117074d8383768a0a121ba8352585411040ab10`
- Run log SHA256: `33ea4cbf64602df684e73d023de1c55e97aeb10211a67a189fc5cd96976ede6d`
- Shard plan SHA256 values: `870eed231d21359d95a75cf9238809b5a8280261932eb10831640d66ef66137f`, `204a23ae5869f9594d2e61af99a903c4b7bb29b9f4855130095d1206af76c328`, `c8dda162c1fd8c524bf9c5eb993e44af7dfd82c943e1937b061632baa5bee2df`, `2caea0a06c32fbd14a9dd93209321447f4262541ad555ec6b2f06b7537fe7525`

Do not repeat collection, replay verification or terminal archiving.

## Disabled FIT Draft

A separate fixed-recipe P26 SOT-FiLM FIT draft was prepared from those exact
archived bytes:

- Task: `fit_c2_policy_sot_film_full_b512_r1_p26_20261105`
- Independent source root: `/data2/gb/Dual-RGBT/c2_policy_fit_sot_film_r1_20261105`
- Run: `/data2/gb/Dual-RGBT/c2_policy_fit_runs/fit_c2_policy_sot_film_full_b512_r1_p26_20261105`
- Target GPU:2
- Draft SHA256: `614e7cba3b7632f63037c7181c8c1edb414c7cc984370d45a6ddf93ee7b5eb87`
- Canonical draft binding: `70305f1d0ff84a43e1ad8a49a3aefac654ac462e5c1d1c9cf21da0d4c3a47d9f`

The draft remains disabled. Read-only canonical runtime preflight passed at
2026-11-05T03:03:04.245902Z: GPU2 had24112MiB free,15MiB used,40C and a
350W power limit, with no compute process; destination disk had6810257137664
bytes free. It did not deploy files, enable the plan or start a process.

The exact plan/source/input review has been requested and remains pending.
Launch requires that separate review to pass and a fresh canonical runtime
admission at launch time. Recipe remains5 epochs, B512, workers4, seed2027,
preview LR4e-5, quality/gain LR4e-4, cost0.05; retain all five weights.
Do not use dev or TEST for tuning, checkpoint selection or branch selection.

No tracking accuracy, benchmark result or target attainment is established
by the replay PASS or engineering preflight. Do not repeat scoring, replay,
archive or launch without the independent exact review and fresh admission.

## 2026-11-05 04:05 UTC Addendum - FIT review and launch

The exact plan/source/input review subsequently passed with zero blockers:
`reports/c2_policy_fit_r1_20261104/sot/film/P2_C2_POLICY_REFRESH_FIT_PLAN_REVIEW_R1_20261105.json`,
SHA256 `bb6d0df237fbbb54e8487c810fdf527f9f00b919f2036af4bf2a28103f4dd50c`.
Attribution remains UNATTESTED/same-family/provisional; scientific
acceptance is false.

Fresh canonical preflight passed at 03:48:38 UTC, followed by a second live
admission and LAUNCHED at 03:54:38 UTC on GPU2. Native wrapper 86037 and
child 86054 were RUNNING with exact argv and matching CUDA ownership at
04:05:09 UTC. The active config matched the enabled plan and all precision
flags were false. No optimizer updates had flushed; this launch is not FIT
completion, convergence or benchmark evidence. Full startup evidence and
the launch report are recorded at
`reports/c2_policy_fit_r1_20261104/sot/film/start_witness/` and
`reports/C2_SOT_FILM_FIT_LAUNCH_20261105.md`.
