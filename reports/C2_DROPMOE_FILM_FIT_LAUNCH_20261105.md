# P25 DropMoE-FiLM Policy FIT Launch

At 2026-11-05 05:01 UTC, the exact P25 DropMoE-FiLM FIT draft passed its
engineering plan/source review and live canonical launch admission.

- Replay prerequisite: matching P25 DropMoE-FiLM TRAIN replay acceptance
  PASS, 882 sequences / 458,553 tracked frames / 91,091 records; matching
  native task COMPLETE/exit0. Replay acceptance explicitly has
  `benchmark_evidence=false` and `policy_fit_admitted=false`.
- Draft: `reports/c2_policy_fit_r1_20261104/dropmoe/film/plan.json`,
  SHA256 `17b1423a656e38caf1b79f580ad2740c57c38222101c2245dc31ebc9810dd1a0`.
  Canonical plan binding:
  `83764506ca7eb4c3b37a4b5f4fab8f9614d2261a47da36ae1284245a72608153`.
- Exact P25 plan review:
  `reports/fit_c2_policy_dropmoe_film_full_b512_r1_p25_20261105/P2_C2_POLICY_REFRESH_FIT_PLAN_REVIEW_R1.json`,
  SHA256 `7e07e83c2ec8cc229edf261e7e6f77dfa0608ba9a0856c2f37b936c8264f14bb`.
  Status SOURCE_PASS/0, with 112 pinned sources and the exact canonical
  plan binding. This was a same-execution self-review, provisional
  engineering admission, not independent scientific review.
- 29 focused local CPU contract tests passed; CLI help passed.
- Fresh remote admission rechecked source/input/replay bindings and
  capacity. P25 GPU3: 24,247 MiB free / 12 MiB used, 38 C, 250 W limit,
  no compute PIDs; target disk free 929,476,128,768 bytes.
- Native task `fit_c2_policy_dropmoe_film_full_b512_r1_p25_20261105`
  was LAUNCHED at 05:02:19 UTC; wrapper PID 3432670. At 05:02:50 UTC,
  native status was RUNNING and child 3432671 held the exact FIT argv and
  P25 GPU3 UUID `GPU-a9c68382-2377-c377-0faf-e01063f64bec` with about
  1,250 MiB allocated. Four data workers were present. Config, precision
  and flushed optimizer progress were not yet present at that observation;
  this is startup/CUDA ownership only, not proof of an optimizer update or
  epoch completion.
- By 05:11:36 UTC, config and precision files had appeared and the exact
  all-false precision flags were verified. The task remained RUNNING with
  the exact parent/child argv, four workers and child-owned GPU3 context
  (19,422 MiB). GPU utilization was 100%, 69 C and 248.36 W/250 W. `steps.csv`
  remained empty and no COMPLETE receipt existed. This confirms entry into
  compute, not yet a flushed optimizer-step witness.
- At 05:17:53 UTC, steps 1-25 were flushed sequentially in epoch 1, all
  four losses finite. Step25 loss `0.8475197553634644`, quality loss
  `0.38895997405052185`, gain loss `0.06151965260505676`, spatial loss
  `3.970400810241699`. CSV SHA256
  `a14781e736d3f83814494ade9ee4fb0e1facf67c9a2e51ce9f5248345abbd6eb`.
  One-time raw archive is
  `reports/c2_policy_fit_r1_20261104/dropmoe/film/first25_updates/`;
  all 11 files match its `SHA256SUMS`.
- At 05:17:53 UTC, the native task and exact child/parent argv remained
  RUNNING; child 3432671 owned GPU3 at 19,422 MiB, 100% utilization,
  69 C/249.13 W under the 250 W limit. No epoch checkpoint or COMPLETE
  receipt existed. This is TRAIN progress only, not benchmark evidence,
  convergence or target success.
- Recipe unchanged: five epochs, B512, workers4, seed2027, stride5,
  cost0.05, preview LR4e-5, quality/gain LR4e-4, weight decay1e-4,
  gradient clip1.0. dev/TEST disabled. Retain all five checkpoints.
- No benchmark metric, convergence, target success, or checkpoint
  selection is claimed. Observe at 180-300 second boundaries; do not
  duplicate the launch or change the active recipe.
