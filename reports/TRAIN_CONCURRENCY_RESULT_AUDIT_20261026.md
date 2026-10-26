# Training concurrency result audit — 2026-10-26

**RESULT_AUDIT_PASS_WITH_LIMITATIONS — 0 blocking findings.** All 579 deterministic checks pass. Capacity finding: **PARTIAL**. Actual short-run execution supports three total training processes on GPU1; it does not directly establish two extra complete cases on each card or a full-run speedup relative to one formal job per card.

Fresh substantive review of primary artifacts. Same-family/provisional. Requested Astra/max; actual model, effort and routing UNATTESTED. No Torch import, checkpoint deserialization, new NN run, active progress/resource/PID query or source/plan/raw-result edit.

## Fixed work and measured rates

Five arms each perform the same eight TRAIN streams × 32 actual AdamW updates: 256 updates, 1,280 causal track calls and eight initializations per arm. Total: 1,280 updates, 6,400 track calls, 40 initializations and 16 workers, with native exit 0. Model total parameter count is 103,218,887, corroborated by every worker receipt.

| Additional workers | Total GPU1 processes | Training s | Added updates/s | Training track calls/s | Spawn-to-join updates/s | End resident MiB | Existing GPU1 logged updates/s (approx.) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 219.416052 | 1.166733 | 5.833666 | 1.085096 | 4773 | 1.245376 |
| 2 | 3 | 118.105613 | 2.167552 | 10.837758 | 1.895653 | 7127 | 1.256060 |
| 4 | 5 | 72.227028 | 3.544380 | 17.721898 | 2.826138 | 11835 | 1.018062 |
| 8 | 9 | 51.725388 | 4.949214 | 24.746068 | 3.475540 | 21243 | 0.470959 |
| 1 | 2 | 222.878229 | 1.148609 | 5.743046 | 1.067944 | 4773 | 1.226656 |

Rows retain the executed order: additional1 first, additional2, additional4, additional8, additional1 repeat. The last baseline is −1.5534% relative to the first. Additional2 is **1.8578× and 1.8871×** the two additional1 rates. Neither baseline is a lone-formal baseline: each includes two total GPU1 processes.

The NVML column is a single after-work resident-memory snapshot while workers remain alive, **not a continuous peak**. The GPU utilization field is likewise one endpoint. At additional2, the recorded value is 7,127 / 24,576 MiB; at additional8 it is 21,243 MiB. Individual worker allocator peaks are separately preserved in JSON.

## Capacity finding

Use total 3 as a conservative GPU1 operating-cap candidate, not the measured total-9 maximum. Report both-card/full-case adoption as an extrapolated bounded decision after the already registered per-case source/native sanity gates. Do not claim proven full-882 acceleration or zero impact on existing training.

GPU1 total-three short training is directly supported: all child exits are 0, the added-worker rate almost doubles against both baselines, and the coexisting formal logged rate is approximately 1.2561 updates/s versus 1.2454 / 1.2267 in the baseline windows. That is supportive operational evidence, subject to CSV flush resolution and evolving workload. GPU0 has no added-worker arm; simultaneous three full cases on both cards and the other registered objective/K/cost recipes remain extrapolations. The six extra grid cases remain registered, not launched by this audit.

Total nine achieves the highest added-worker rate, but its coexisting formal rate drops to approximately 0.4710 updates/s. These snapshots do not establish a preferred long-run optimum.

## Integrity checks A–F

### A ground truth provenance — PASS

GT comes from the frozen TRAIN manifest, after the causal tracker has executed the current frame. Both sampled teacher queries compare executed boxes to the dataset target through pixel_iou; gains are query IoU minus STOP IoU. The selected decision is replayed/reused under no_grad and asserted equal to the already executed box. This is real-GT supervised training work, not model output masquerading as reference truth. This audit verifies the source and frozen input binding, not a new independent IoU/NN replay; raw teacher labels are not retained in this profile.

Evidence: `scripts/profile_candidate_training_scaling.py:87`, `scripts/profile_candidate_training_scaling.py:96`, `scripts/train_candidate_evidence_policy.py:49`, `scripts/train_candidate_joint_gainpair_trajectory.py:45`.

### B normalization and arithmetic — PASS

All five update rates are 256 / stored training_wall_seconds and all training track rates are 1280 / that denominator. Spawn-to-join rates and approximate coexisting CSV rates recompute exactly. All 1280 seven-component loss rows are finite and satisfy the three FP32 loss-sum identities exactly. There is no performance normalization by a model prediction maximum or mean. Loss decompositions are supervised objectives, not accuracy scores.

Evidence: `scripts/profile_candidate_training_scaling.py:180`, `scripts/train_candidate_joint_gainpair_trajectory.py:88`, `tools/summarize_training_scaling.py:51`.

### C result existence native binding — PASS

COMPLETE/config, five arm summaries, 16 worker records and 40 step payloads exist. Native wrapper PID 1099461 records COMPLETE and exit 0 for the exact GPU1 command. Runtime code matches READY worker PIDs to the processes it created, then requires every owned child exit 0; all archived worker/summary/log records agree. The observer node contains the same five arms in lexical order, so its contents are compared by arm name. All 51 static sources, 28 inputs and 67 finished remote artifact copies hash-match. No new model checkpoint is saved.

Evidence: `runs/pipeline_tasks/profile_candidate_training_scaling_20261026/status.json:12`, `scripts/profile_candidate_training_scaling.py:153`, `scripts/profile_candidate_training_scaling.py:178`, `scripts/profile_candidate_training_scaling.py:207`.

### D execution and gradient protocol — PASS

The actual profile calls the same causal_batch, sampled_targets, selected_target, controller_loss and step_optimizer as the gain-pair formal trainer. Each stream reloads B6250, creates fresh AdamW/tracker/RNG, tracks frames 1–160 and executes 32 updates at frames 5,10,...,160. Preview LR 1e-5, quality/gain LR 1e-4, weight decay 1e-4, gradient clipping 1, seed 2027, max_candidates 8 and cost 0.05 match the base plan. Teacher paths are no_grad; learning preview/loss remains differentiable. Runtime asserts finite loss and clipped norm, 251 frozen state tensors, exactly 14 changed tensors in the permitted prefixes, and no frozen gradients. The profiler passes sanity=False to step_optimizer, so the extra sanity-only per-prefix nonzero-gradient tests are not run here. Its optimizer-step-32/freeze checks concern each worker final stream; no raw gradients or final weights are archived for independent replay.

Evidence: `scripts/profile_candidate_training_scaling.py:57`, `scripts/profile_candidate_training_scaling.py:79`, `scripts/profile_candidate_training_scaling.py:99`, `scripts/profile_candidate_training_scaling.py:123`, `scripts/train_global_evidence_policy.py:40`.

### E scope and capacity — WARN

Measured 1/2/4/8/1 additional workers coexist with one formal GPU1 job: totals are 2/3/5/9/2, not 1/2/4/8/1. All arms use the same eight selected TRAIN prefixes with independent resets; they do not reproduce uninterrupted full-882 evolving state or checkpoint I/O. Only the gain-pair/K8/cost0.05 recipe is measured. GPU0 has no added-worker arm. No isolated formal-only baseline exists. Evidence supports short-run GPU1 three-process feasibility and a conservative capacity choice, but per-card two-extra full-case deployment and total completion-speed gain are not directly established.

Evidence: `reports/TRAIN_CONCURRENCY_PLAN_20261026.json`, `reports/TRAIN_CONCURRENCY_CANDIDATE_GRID_20261026.json`, `scripts/profile_candidate_training_scaling.py:77`, `scripts/profile_candidate_training_scaling.py:164`.

### F evaluation type — PASS

The workload performs real-GT supervised TRAIN updates; the reported metrics are empirical system timings, not an accuracy benchmark. Cross-arm output comparisons are a model-output consistency subtest. No dev/test metric or accuracy improvement is inferred.

Evidence: `scripts/train_candidate_evidence_policy.py:57`, `scripts/profile_candidate_training_scaling.py:117`.

## Cross-arm outputs and actual learning

All 40 stream payloads hash-match their receipts. Saved update-frame boxes and all non-timing route fields, including confidence, are equal across arms. Their losses are not uniformly bit-exact: 89 rows in 22 nonreference stream/arm pairs differ, with maximum absolute difference 5.960464477539063e−8. Every total/quality/gain decomposition still satisfies its FP32 addition identity. No final weights or optimizer tensors were saved, so no bit-exact training-state claim is made.

The profile uses the same native forward/teacher/replay/loss/backward/AdamW path as the active gain-pair trainer. It resets the original B6250 model, optimizer, tracker and sampler for each stream. Teacher evaluation is under no_grad; learned preview, quality and gain paths retain gradients. The runtime requires finite loss/gradient norm, frozen tensor equality and expected changed prefixes. Because sanity=False, per-prefix nonzero-gradient sanity checks are not repeated in this profile; optimizer step count and final frozen-state assertions apply at each worker’s last stream.

The raw route log covers only the 32 update frames per stream, not every tracked frame. STOP 120, local-main 126 and global-wide 10 are first-arm **update-frame counts**; no whole-frame auxiliary fraction is inferred.

## Limits that must accompany claims

- **Three processes was measured only on GPU1 for a short workload.** additional2 completed 256 updates in 118.105612639 s: 2.167551518 updates/s, versus both additional1 baselines 1.166733235 and 1.148609269. These baselines themselves already include a coexisting formal job, so they are not the current one-process-only baseline. Applying the result to extra full cases on both cards is a resource-planning extrapolation, not measured full-run speedup.
- **NVML endpoint is not a measured peak.** gpu_after_work_csv is sampled once after all worker result messages, while workers are held resident before finish release. At total 3 processes it reads 7127 / 24576 MiB and 15% utilization. Neither field is a continuous peak/mean. The total-9 endpoint is 21243 MiB and 8% utilization. Worker allocator peaks (559048704–561272320 allocated bytes; 622854144–627048448 reserved bytes across all arms) exclude driver/context/coexisting-process memory and pre-barrier initialization peaks.
- **Timing definitions and evidence boundary.** training_wall spans the parent release timestamp through the latest synchronized worker finish. It includes per-stream state reload, optimizer construction, initialization, all causal frames, teacher queries, backward/AdamW and step JSON writing/hashing. Initial imports/model construction/checkpoint loading are before READY and excluded. Post-work CPU state validation and result IPC add about 1.05–1.18 s; spawn-to-join includes load, observation and shutdown. Saved worker finish counters/durations permit consistency checks, but parent release and spawn/join raw endpoints are not independently archived. Thus rate arithmetic is independently reproduced while those denominator values retain their source-defined runtime-witness boundary. Training track frames/s is not online inference FPS, and no per-frame service mean/P95 is available.
- **Coexisting formal rates are approximate.** Formal trajectory CSVs explicitly flush on the first update and every 25 updates; read snapshots are not exact instantaneous counters and the two jobs are read sequentially. At additional1 first / additional2 / additional4 / additional8 / additional1 repeat, GPU1 logged rates are 1.245376 / 1.256060 / 1.018062 / 0.470959 / 1.226656 updates/s. GPU0 rates are also preserved. The formal trajectories keep evolving through different sequences and routes, so these numbers are a contention warning rather than a controlled causal slowdown estimate. Highest added-worker throughput at total 9 is not proof that it is the preferred operational configuration.
- **Saved update-frame equality is not weight equality.** All five arms have equal saved frame numbers, boxes and non-timing routes, including confidence, at the 32 update frames per stream. Raw payload hashes include route.seconds and need not agree. Across nonreference arms, 89 loss rows in 22 stream/arm pairs differ, maximum absolute difference 5.960464477539063e-8. Loss identities still hold exactly after FP32 addition. No model checkpoint/final tensor digest is saved, so bit-exact weights, optimizer-state equality and uninterrupted full-trajectory equality are not established. The audit imports no Torch and performs no neural replay.
- **Only update-frame routes are logged.** Each arm tracks 1280 frames but saves route data for only 256 update frames. The first arm update-frame counts are STOP 120, local-main 126 and global-wide 10. These counts cannot be divided by all tracked frames or presented as a whole-frame auxiliary-query fraction. The teacher still cycles all four request families equally (8 times per stream) independently of the saved selected actions.
- **Short prefix/reset/cache and recipe limits.** Eight selected TRAIN streams, 32 updates each, one seed, one base checkpoint and one gain-pair/K8/cost0.05 recipe were tested. Each stream starts fresh from B6250, unlike the active formal process whose model/optimizer evolve across 882 streams and two rounds. Arm order is fixed and repeated filesystem access/cache state may differ. additional1 repeat drift is -1.553394084%; both baselines are retained. No full checkpoint-save cost, final accuracy, full-882 completion time or other grid-case behavior is measured. The six registered extra grid rows remain PLANNED_NOT_LAUNCHED, including the two reserved GPU2/3 rows.
- **Assertions, dependencies and requested environment.** Freeze, changed-state and optimizer counts are source-enforced runtime assertions, not independently recovered tensors. The profile never saves teacher query labels or raw gradients; this audit cannot regenerate their numerical targets. The 51-file source map does not seal every imported library/config/package or image byte. OpenCV thread count is requested by OPENCV_FOR_THREADS_NUM=1 and asserted as an environment value; actual cv2.getNumThreads is not recorded. Torch one-thread and TF32-disabled settings are source enforced. No AMP/half conversion is introduced.
- **Unchanged parent task and devices.** The archived formal startup and per-arm records identify the coexisting GPU1 wrapper 1054384/child 1054385. The audit reads only saved observations and static files; it does not query any active training progress, current PID, GPU resources, temperature or power, and launches no work on any GPU.

## Evidence and trace

All 51 source hashes (including unchanged inherited 50), 28 frozen inputs and 67 remote finished-file copies match. Native task `profile_candidate_training_scaling_20261026` exited 0; its whole command lasted 796.875218 seconds. Existing formal identities are corroborated from saved launch/startup/arm snapshots, with no new active query.

- Primary result: `profile_output/candidate_training_scaling_20261026/COMPLETE.json`
- Native result: `runs/pipeline_tasks/profile_candidate_training_scaling_20261026/status.json`
- Independent checks and comparisons: `.aris/traces/experiment-audit/2026-10-26_training_concurrency_result_review/RECOMPUTATION.json`
- Static remote bindings: `.aris/traces/experiment-audit/2026-10-26_training_concurrency_result_review/REMOTE_STATIC_BINDINGS.json`
- Full machine-readable assessment: `reports/TRAIN_CONCURRENCY_RESULT_AUDIT_20261026.json`

Instrumentation changelog: the existing standalone profiler and offline summarizer were inspected without edits; this reviewer created only the two result-audit reports and trace artifacts. No model checkpoint, source, plan, raw experiment result, HANDOFF or Git state was changed.
