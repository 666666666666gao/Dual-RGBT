# VRAM18 training source review — R1

Status: **SOURCE_FAIL — 3 blockers**. Fresh same-family review; acceptance provisional. Requested Astra/max; actual model and reasoning effort **UNATTESTED**.

Only local source, JSON, hashes and AST were inspected. No SSH, NN/GPU execution, active log/PID/weight/environment query or implementation edit occurred.

## T1 — Later waves are admitted without actual throughput, original-task slowdown, or swap evidence

Phases 6/8 accept a previous FORMAL_STARTUP_PASS after one startup CSV observation. next_observation_epoch is written but never consumed or enforced. The launcher neither computes delta-update throughput nor compares the original eight jobs against a measured baseline; it checks only MemAvailable and does not examine system/process swap before expansion. The supplied historical capacity records already contain nonzero swap (43,344, 471,876 and 668,000 KiB for three original children), so this is an evidenced capacity concern. A completed four-update allocator profile is not a sustained wave-health measurement.

Evidence: `tools/launch_vram18_training_wave.py:33`, `tools/launch_vram18_training_wave.py:36`, `tools/launch_vram18_training_wave.py:81`, `tools/launch_vram18_training_wave.py:111`, `tools/launch_vram18_training_wave.py:127`, `reports/VRAM18_RAM_AND_DISK_CAPACITY_20261027.json`, `reports/VRAM18_HOST_MEMORY_BREAKDOWN_20261027.json`.

Required change: Before phase 6 or 8, require a sealed observation of the previous wave at its scheduled diagnostic node, with actual aggregate update rate, original-task slowdown, RAM and swap measurements and an explicit admission decision. Preserve the original eight tasks.

## T2 — The ownership guard conflates a running native child with CUDA readiness

The wrapper writes RUNNING before spawning its child. A child then imports code and hashes source/input files before its first .cuda(). The next same-card launch includes every RUNNING child in expected GPU PIDs and requires actual==expected after only a fixed two-second sleep. Phase 4 starts three consecutive sanities on GPU 2 and three on GPU 3, so legitimate CPU initialization can make the next launch abort. The same transition applies to sequential same-card formal launches. Static string expansion reproduces an admissible RUNNING/no-CUDA state rejected by the guard; this is a source state-machine counterexample, not an observed runtime failure.

Evidence: `tools/launch_vram18_training_wave.py:68`, `tools/launch_vram18_training_wave.py:74`, `tools/launch_vram18_training_wave.py:79`, `tools/launch_vram18_training_wave.py:88`, `tools/launch_vram18_training_wave.py:90`, `scripts/run_task_four_gpu.py:24`, `scripts/run_task_four_gpu.py:28`, `scripts/train_candidate_joint_quality_trajectory.py:258`, `scripts/train_candidate_joint_quality_trajectory.py:307`.

Required change: Represent registered-but-initializing children separately from GPU-ready owners. Require all known ready children to be present and every actual CUDA process to belong to registered children. Limit slots by registered live children, including initializing children. Do not weaken foreign-process rejection or restart any job.

## T3 — Partial sanity launches are not durably represented in the wave state

The initial report is saved before launching, but each sanity PID/command remains only in memory until all eight launch calls succeed. A T2 guard failure during the second same-card sanity leaves a launched child absent from the durable cases and the report still SOURCE_ADMITTED_HOST_PROFILE_PASSED. The report then prevents re-entry with assert not report_path.exists(). Formal receipts are written per case, but the state label remains SANITIES_LAUNCHED until the entire loop completes. This does not stop the old eight, but makes the new wave lifecycle incomplete and unreviewable after a partial launch.

Evidence: `tools/launch_vram18_training_wave.py:40`, `tools/launch_vram18_training_wave.py:55`, `tools/launch_vram18_training_wave.py:89`, `tools/launch_vram18_training_wave.py:91`, `tools/launch_vram18_training_wave.py:106`, `tools/launch_vram18_training_wave.py:109`.

Required change: Persist each returned sanity/formal launch receipt immediately, with explicit partial-launch/admission/formal transition states. Preserve existing identities and do not add automatic retries, fallbacks, or duplicate launches.

## Verified source properties

- 57 source hashes match; all 32 configuration identities are unique, with 24 new cases and eight original plan hashes unchanged. All original input maps and sequence order are retained. Declared phase placement yields 4→6→8 tasks per card.
- The objective-specific training code, architecture/layers, seed2027, learning rates, batch behavior, original B6250/SOTep14 initialization and full882×2/182182 updates/917106 tracked frames/1764 initializes remain unchanged.
- Native0, actual4-update sanity, selected replay, teacher/tracker-state equality, 251 frozen+14 changed tensors and strict reload precede formal reset to original B. No kill, preemption, fallback or retry path is added.
- All 96 generated remote programs parse; all 48 training argv vectors match their plans. The source-only STARTING-state counterexample is saved in the trace and is not a runtime experiment.
- The allocator recipe compares all265 state tensors, recursive optimizer state and both CSV byte streams; the launcher requires native0, equality flags and lower compact sampled peak PSS. Actual allocator results remain unverified and mandatory before any new formal run; four steps do not establish formal steady-state memory.

No per-case SOURCE_PASS files were issued. The original eight tasks were not queried, changed, abandoned or restarted.

## Sealed inputs

- index: `d1e7acbca0cfae197496d353df0a69c27dc7ab1d70f2a9ba783836cfeb313047`
- launcher: `b2c2e964f9d5dfe1f686ac60c2439afecf2792693d2bc85e05dfa6a5e4bfbdc4`
- registration: `551ef19b38c2d0bc7b5b5973455de3437a0884dc4458a14625e4e145cab31824`
- extended evaluation protocol: `e47ebe248fc8893ebb5a89322428effe7498f7757000f231fac164c9c1e4cee2`

Full57-source union, all32 plan hashes, all available non-weight input hashes and limitations are in the JSON report. Trace: `.aris/traces/experiment-audit/2026-10-27_vram18_source_review`.

# VRAM18 evaluation source review — R1

Status: **SOURCE_PASS — 0 evaluation-source blockers**. Fresh same-family review; acceptance provisional. Requested Astra/max; actual model and reasoning effort **UNATTESTED**.

The extension retains the original protocol's full evaluation contract. Only the source map, registered plan map, review path and scope text differ. All57 source hashes and all32 plan hashes match; the original eight plan bindings are unchanged.

Dataset GT and the supplied toolkit remain authoritative, with no prediction-statistic normalization. The source requires completed full882×2 training/native0 and an exact265-tensor witness; actual11-frame smoke and strict reload; full97dev/54646 frames at costs0/.05/.15; a fresh result audit binding the plan/checkpoint/exact three reports; and strict dev SR>56.43314371195134. Eligible models freeze one checkpoint/config across all three full tests: LasHeR245/220703, RGBT210210/104706, RGBT234234/116650.

The result schema preserves actual total parameters103218887, GPU/input dimensions, decode+initialization-inclusive frame-service FPS and mean/P95 latency. Parallel aggregate throughput and unpaired baseline speedup are excluded. Relevant source: `scripts/joint_evaluation_contract.py:21,32,68,124,147,240,266`; `scripts/run_joint_candidate_evaluation.py:36,77,103,115,143`; `tools/joint_candidate_results.py:16,45,79,124`; `scripts/benchmark_inputs.py:102`; `dual_rgbt_global/candidate_tracker.py:21,65`.

This verdict does not grant training admission: the training source review is SOURCE_FAIL/T1–T3. No runtime/model/accuracy/FPS result is certified, and no per-case training PASS is issued. The actual allocator proof and all runtime ownership/capacity, completion, smoke, dev-audit and test-freeze gates remain mandatory.

Protocol SHA256: `e47ebe248fc8893ebb5a89322428effe7498f7757000f231fac164c9c1e4cee2`. Original protocol SHA256: `9ad6d14fd611dbd8226f187d77ed523c646f344388dfbb1138ac5750df3d83f0`. Exact source and input maps are in the JSON report. Trace: `.aris/traces/experiment-audit/2026-10-27_vram18_source_review`.
