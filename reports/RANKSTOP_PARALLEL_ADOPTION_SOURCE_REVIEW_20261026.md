# RankSTOP parallel adoption source review

**SOURCE_PASS; 0 blocking findings.**

Same-family / provisional. Requested gpt-6-astra, reasoning max; actual routing UNATTESTED.

This review covers only the sealed one-shot operational handoff. Reviewer did not stop a real process, perform SSH/GPU work, launch a real queue/model, or edit project source.

- Plan SHA256: `5ee1bf5fdb3983b6b2118272fd6a192f48e23fe8ee73c474112d99aa1aebcd6f`
- Full51-source map and3 local-input hashes are in the companion JSON.
- Original50-source map and both original plans remain unchanged.
- Validation:13 embedded-code CPU cases,6 copied PowerShell CPU mock cases,0 PowerShell syntax errors.

## Resolved R1 finding

Initial preflight read wrapper argv without asserting equality; initial post-retirement block trusted status.json command and /proc/cmdline existence without re-reading live wrapper/child argv and parentage.

Final source validates native RUNNING/pid/command/physicalGPU/start, exact wrapper argv, sole child PID, child argv and PPid both before retirement and before GPU1 launch.

Final embedded-code CPU tests reject wrong wrapper, wrong child argv, wrong PPid, different child PID and completed status in both stages. R1 was a static finding; no R1 live operation or fabricated red test is claimed.

## Checks

### exact_source_and_plan_binding: PASS

Final plan SHA matches. All51 source and3 local-input hashes match. Removing only the new ops script entry yields the prior reviewed50-source map exactly; dev48-source plan and queue plan retain a8b76928... and f866be18... hashes. Architecture, weights, train/eval algorithms, seed, manifests, costs and scorer are unchanged.

Evidence: `reports/RANKSTOP_PARALLEL_ADOPTION_PLAN_20261026.json`; `reports/rankstop_dev_plan_server26_20261026.json`; `reports/rankstop_dev_queue_plan_server26_20261026.json`.

### windows_owner_and_sleep_guard: PASS

Initial owner check requires exact PID30652, parent launcher41216, full command and birth timestamp. Queue must remain FULL_DEV_RUNNING with2 tasks, same GPU0 wrapper PID and0 metrics, with next action at least300s away. Immediately before retirement PID presence and exact command are checked again. Stop-Process targets only that owner.ProcessId; no remote stop command exists.

Evidence: `scripts/adopt_rankstop_dev_parallel.ps1:17`; `scripts/adopt_rankstop_dev_parallel.ps1:20`; `scripts/adopt_rankstop_dev_parallel.ps1:65`.

### gpu0_live_identity_before_and_after: PASS

Both native check blocks require the registered RUNNING native pid/argv/start/physicalGPU0, exact wrapper command, sole child439647, exact child command and PPid439618. Post-retirement verification completes before new launch; current source resolves R1.

Evidence: `scripts/adopt_rankstop_dev_parallel.ps1:33`; `scripts/adopt_rankstop_dev_parallel.ps1:84`.

### gpu1_admission_precedes_retirement: PASS

Before any Stop-Process, GPU1 memory must be below500MiB and a seed2027 8x8 CUDA matmul must be finite and synchronized. The remote helper returns only after this short subprocess ends. Original launch helper rechecks GPU1 below500MiB and target-task nonexistence before Popen. No GPU2/3, temperature or power query is introduced.

Evidence: `scripts/adopt_rankstop_dev_parallel.ps1:42`; `scripts/adopt_rankstop_dev_parallel.ps1:63`; `scripts/complete_pipeline.py:52`.

### native_exit_preserved: PASS

Existing scheduler records the native Python launcher exit. Ops script requires matching launcher PID and copies the actual exit value unchanged to OWNER_RETIRED_NN_CONTINUES with an explicit operator-retirement reason. It does not write a synthetic successful NN exit or alter original remote status. CPU fixture value99 is preserved exactly.

Evidence: `scripts/adopt_rankstop_dev_parallel.ps1:70`; `scripts/run_scheduled_controller.ps1:15`.

### single_launch_and_manual_ownership: PASS

The one-shot script launches only the pre-registered cost0.05 full-dev command on physicalGPU1 using unchanged launch/run_rankstop_dev. Original cost0 task stays detached and is never relaunched. The new execution receipt names root operator, pending cost0.15, and first observation+900s. There is no replacement background observer, polling loop, training or test command.

Evidence: `scripts/adopt_rankstop_dev_parallel.ps1:96`; `scripts/adopt_rankstop_dev_parallel.ps1:99`; `scripts/complete_pipeline.py:60`.

### failure_and_task_preservation: PASS

Preflight failure throws before observer retirement. After retirement, a failed identity check/launch stops the ops script without retry; original GPU0 process is detached by the original helper and is not sent a signal. Success re-entry is prevented by the execution-state guard; a post-retirement failure also cannot blindly replay the old-owner check because that owner is absent. Root must inspect receipts and assume monitoring after retirement.

Evidence: `scripts/adopt_rankstop_dev_parallel.ps1:15`; `scripts/adopt_rankstop_dev_parallel.ps1:64`; `scripts/adopt_rankstop_dev_parallel.ps1:104`; `scripts/complete_pipeline.py:60`.

### throughput_measurement_scope: PASS

Preflight counts written trace frame indices+initialization across completed and current sequences, and stores last256 recorded per-frame service times. A CPU fixture with an unfinished current sequence and a short partial CSV row verifies count inclusion and arithmetic. These buffered trace counts and short service windows are operational diagnostics; final full-dev FPS remains the original full timing computation, with mixed solo/parallel load disclosed.

Evidence: `scripts/adopt_rankstop_dev_parallel.ps1:48`; `scripts/adopt_rankstop_dev_parallel.ps1:99`; `scripts/score_tracking.py:72`.

### three_cost_and_test_boundary: PASS

Plan explicitly retains all3 full97/54646 dev runs, original toolkit scoring, independent result audit and bestSR strictly above originalE1 dev56.43314371195134 before later tests. This ops script neither completes those obligations nor launches tests. It records remaining0.15 for the manual owner.

Evidence: `reports/RANKSTOP_PARALLEL_ADOPTION_PLAN_20261026.json:105`; `scripts/adopt_rankstop_dev_parallel.ps1:99`.

### syntax_templates_and_cpu_cases: PASS

PowerShell parse has0 errors; both local Python templates and their expanded remote programs parse as Python3.7. Single-quoted PowerShell here-strings preserve Python %r/chr(0) code; no reserved $PID/$HOME variable is assigned.13 embedded remote identity/resource cases and6 complete copied-script CPU mock cases pass on final source. No real process stop, remote launch, SSH or GPU call was made by reviewer.

Evidence: `.aris\traces\experiment-audit\2026-10-26_parallel_adoption\cpu_templates_5ee1bf5fdb39.json`; `.aris\traces\experiment-audit\2026-10-26_parallel_adoption\shell_cpu_8b4a7d9ef93a.json`.

## Scope and operational warnings

- **W_SOURCE_ONLY**: This review does not attest current live GPU0 identity, current Windows ownership, GPU1 availability, CUDA witness success or actual retirement/parallel launch. Those must be evidenced by execution receipts. Existing documented-launch R2 is historical source-bound input, not a new observation by this reviewer.
- **W_MANUAL_CONTINUATION**: After observer retirement the root operator owns both observations and collection. Cost0.15, all3 scores, result audit, original parent-training exit warning and test gate remain mandatory. The script deliberately creates no replacement background watcher.
- **W_MIXED_LOAD_FPS**: Cost0 spans solo and parallel load. Recent256-frame service FPS and15-minute combined throughput cannot establish paired speedup. Trace counting sees flushed rows and can lag real progress; keep timestamps and this load disclosure with final capacity/FPS reporting.
- **W_RETIREMENT_FAILURE_RECOVERY**: The native exit receipt is read after a2-second settle. Missing/late receipt or later SSH/identity/launch failure aborts the script after retirement; original detached NN continues. Inspect original native receipt, before/retirement records and actual task state before any manual continuation; do not relaunch the old serial observer or retry the ops script blindly.
- **W_INVOCATION_ENVIRONMENT**: Use the documented existing PowerShell7 environment. Historical R1 documented launch failed from a different PowerShell module environment, while R2 used direct invocation successfully. No project compatibility layer was added.

No new full-dev score, formal FPS, paired speedup, successful live handoff, or achieved target is claimed.
