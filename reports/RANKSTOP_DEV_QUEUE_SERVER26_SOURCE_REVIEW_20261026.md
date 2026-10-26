# RankSTOP server26 queue source review

**Verdict: SOURCE_PASS; 0 blocking findings.**

Independent source review under experiment-audit; same-family / provisional. Requested gpt-6-astra, reasoning max; actual model routing UNATTESTED.

Scope: server26 source readiness for the same final C882 checkpoint, one train smoke and three complete dev costs. This is not a completed-dev result audit. No reviewer GPU, SSH, real queue, training or model inference was run.

- Plan: `reports/rankstop_dev_queue_plan_server26_20261026.json`
- Plan SHA256: `f866be18a9ae3c47d21a26774a95b5933438eeee461e48b21a0c861fda2921d9`
- Final checkpoint SHA256: `527a3bc0f56f6a26e31adcc5e656f971b6bd34bbcd220251151d7db1cc228753`
- Full sealed source map: companion JSON (50 entries).
- Validation: 17 static/isolated CPU check groups, 2 real CPU scheduler exit cases, path/report-only AST comparisons, PowerShell parse with 0 errors.

## Checks

### sealed_review_and_entrypoint: PASS

All50 queue source hashes match and queue plan binds the exact revised dev plan. Startup rejects stale review/plan/source/input/toolkit before resource polling or launch. Scheduler uses the available Python3.12 scorer environment and refuses a preexisting invocation log directory.

Evidence: `scripts/queue_rankstop_dev.py:19`; `scripts/run_scheduled_controller.ps1:8`; `scripts/run_scheduled_controller.ps1:15`.

### server26_gpu_admission: PASS

Remote execution uses SSH2026 and /data2/gb/Dual-RGBT conda only. free_gpu queries only physical0/1 memory and admits strictly below500MiB. launch rechecks selected GPU below500MiB and refuses an existing native task directory. No temperature/power queries. Mock boundary500 rejected/499 admitted on allowedGPU1.

Evidence: `scripts/complete_pipeline.py:17`; `scripts/complete_pipeline.py:52`; `scripts/queue_rankstop_dev.py:43`.

### one_owned_nn_and_no_c_restart: PASS

Loop is exactly smoke,0.0,0.05,0.15, sequentially launch->finish->collect->score. No new launch occurs until prior native task terminal success; failures abort without retry. Only run_rankstop_dev.py is launched. complete_pipeline is imported as helpers; its historical training main is not executed.

Evidence: `scripts/queue_rankstop_dev.py:91`; `scripts/queue_rankstop_dev.py:99`; `scripts/queue_rankstop_dev.py:103`; `scripts/complete_pipeline.py:182`.

### native_identity_and_exit: PASS

Finish validates launch PID/task/GPU; RUNNING native wrapper command and child command are checked against /proc. Terminal status must be COMPLETE/exit0. run_task retains actual child returncode and command; scheduler retains actual process ExitCode,PID,plan hash. Isolated native CPU and scheduler cases0/7 both passed.

Evidence: `scripts/queue_rankstop_dev.py:58`; `scripts/queue_rankstop_dev.py:70`; `scripts/run_task.py:24`; `scripts/run_scheduled_controller.ps1:15`.

### three_cost_closure_and_test_gate: PASS

Each of the three full-dev native completions must pass the unchanged local scorer before best selection. Max(SR,PR,NPR,-mean_ms) matches plan; original E1 reference56.43314371195134 exactly matches existing artifact. Queue ends FULL_DEV_COMPLETE_PENDING_RESULT_AUDIT and test_launched=false whether SR improves or not. No test launch exists in this entrypoint.

Evidence: `scripts/queue_rankstop_dev.py:114`; `scripts/queue_rankstop_dev.py:119`; `reports/candidate_complete_adopted_20261016/e1_dev_cost0p15_freeze_recomputed.json`.

### observation_estimates: PASS

Busy first node1800s then resource polls300s. Smoke first180s, full-dev first1800s. Subsequent full-dev observation is estimated from finished-sequence frame counts and elapsed time, scheduling approximately180s before finish, bounded180..3600s. No progress or smoke uses300s. next_observation_epoch is persisted; helper60s sleep chunks do not poll resources.

Evidence: `reports/rankstop_dev_queue_plan_server26_20261026.json`; `scripts/queue_rankstop_dev.py:38`; `scripts/queue_rankstop_dev.py:55`; `scripts/queue_rankstop_dev.py:82`; `scripts/complete_pipeline.py:30`.

### migration_and_result_boundary: WARN

Runtime validates registered source/input bytes and full tracker smoke before dev. Migration must finish before the real controller is invoked; this source review did not contact either server and cannot attest installed files, live GPU idleness or task cancellation. Full-dev result and later test review remain separate.

Evidence: `scripts/run_rankstop_dev.py:36`; `scripts/run_rankstop_dev.py:49`; `reports/SERVER26_TRAIN_DEV_MIGRATION_20261026.json`.

## Scope warnings

- **W_PARENT_EXIT_UNKNOWN**: The original direct C882 training launch has no native terminal receipt. Its actual process exit code remains null; artifact checks do not establish that exit code or replay training-time causal checks.
- **W_RUNTIME_PENDING**: This is SOURCE review, not a completed-dev result audit. The provided seeded CUDA witness is a small generic kernel/update/reload check, not the full tracker. Reviewer ran no GPU, SSH, real queue, training or inference. Real 11-frame smoke/strict reload, native terminal receipts, full97x3 predictions, metrics and measured FPS remain pending.
- **W_REMOTE_ARTIFACT_SCOPE**: Train/dev migration report establishes all referenced files nonempty and float32 annotations equal, and explicitly does not hash all image contents. Original/full train manifests and large checkpoints are remote-only in this reviewer workspace; their bytes and installed environment were not independently rehashed here. Exact local dev/smoke manifests are verified to differ only in image paths. Runtime preflight hashes every registered input and source before model construction.
- **W_HISTORICAL_ENTRYPOINTS**: SOURCE_PASS applies to the sealed new RankSTOP dev/queue entrypoints. Imported helpers are used, but historical complete_pipeline.main, run_benchmark_tracking.main, global policy build_policy, and the old training audit entrypoint retain old campaign/path assumptions and are not approved as new server26 campaigns. Do not invoke those historical mains to restart C or bypass the later result/test audit.
- **W_INHERITED_RGBT234_PROTOCOL**: Separate test inventory flags afterrain visible GT. Local supplied toolkit and preexisting RGBT234 manifest independently agree on first box [144,53,10,80] and all 123 visible boxes. The executor report says raw data use [144,53,24,45] and raw25/raw26 SHA are equal; remote raw bytes were not reverified by this reviewer. This is a disclosed inherited toolkit/raw protocol issue, outside current dev-only approval. RGBT210 migration and complete test acceptance are separate; no test metrics are approved.

## Deterministic evidence

- `.aris/traces/experiment-audit/2026-10-26_server26_source/cpu_verification.json`
- `.aris/traces/experiment-audit/2026-10-26_server26_source/scheduler_cpu_verification.json`
- `.aris/traces/experiment-audit/2026-10-26_server26_source/reporting_only_verification.json`

The initial scheduler fixture inherited PowerShell7 module paths into Windows PowerShell5.1 and failed module discovery. Direct native invocation worked; final isolated fixtures used Windows PowerShell default module discovery and preserved exits0/7. No project source was changed by the reviewer.

No new accuracy/FPS value, completed full-model GPU smoke, original training native exit code, or target achievement is asserted.
