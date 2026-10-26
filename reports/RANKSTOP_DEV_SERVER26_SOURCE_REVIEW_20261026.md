# RankSTOP server26 dev source review

**Verdict: SOURCE_PASS; 0 blocking findings.**

Independent source review under experiment-audit; same-family / provisional. Requested gpt-6-astra, reasoning max; actual model routing UNATTESTED.

Scope: server26 source readiness for the same final C882 checkpoint, one train smoke and three complete dev costs. This is not a completed-dev result audit. No reviewer GPU, SSH, real queue, training or model inference was run.

- Plan: `reports/rankstop_dev_plan_server26_20261026.json`
- Plan SHA256: `a8b76928f4e4ebe795d46f8c0ac9ed8257f291a3e78579a7a99c543480dbb82f`
- Final checkpoint SHA256: `527a3bc0f56f6a26e31adcc5e656f971b6bd34bbcd220251151d7db1cc228753`
- Full sealed source map: companion JSON (48 entries).
- Validation: 17 static/isolated CPU check groups, 2 real CPU scheduler exit cases, path/report-only AST comparisons, PowerShell parse with 0 errors.

## Checks

### migration_only_diff: PASS

All eight path-only source changes exactly reproduce git02d6689 with explicit path substitutions. Stripping only the added reporting fields/summary and relocated protected path makes runner/scorer AST equal to baseline. No network/module/optimizer/loss/update algorithm change; local evidence fusion remains present.

Evidence: `dual_rgbt/model.py:34`; `dual_rgbt_global/candidate_policy.py:87`; `dual_rgbt_evidence/readout.py:108`; `git diff 02d6689 --`.

### source_binding: PASS

All 48 source digests match; corresponding Python files parse with Python3.7 syntax. Plan/review/source/input digests are checked before model construction and sources/inputs again after inference.

Evidence: `scripts/run_rankstop_dev.py:30`; `scripts/run_rankstop_dev.py:141`.

### same_final_checkpoint: PASS

One registered final C882 checkpoint SHA is used for smoke and all three dev costs. Runtime validates saved step182182/stagecandidate_c/original training plan and strict-loads the final state into the same candidate architecture. Original unknown parent exit code is asserted and warning is propagated.

Evidence: `scripts/run_rankstop_dev.py:56`; `scripts/run_rankstop_dev.py:71`; `tools/score_rankstop_dev.py:38`.

### smoke_and_strict_reload: PASS

Train smoke uses first11 frames of one176-frame train sequence. Full dev requires native smoke COMPLETE/exit0, same plan/weight SHA and reload PASS. Reload separately strict-loads the checkpoint, replays the final saved causal state and requires box/route/templates/history bit equality; no optimizer/backward call is made.

Evidence: `scripts/run_rankstop_dev.py:49`; `scripts/run_rankstop_dev.py:102`; `scripts/run_rankstop_dev.py:138`; `scripts/run_benchmark_tracking.py:105`.

### A_ground_truth_provenance: PASS

GT comes from original dataset manifests/init.txt, never model outputs. All dev/smoke non-path fields and frame ordering equal the original manifests. Dev97/54646 is disjoint from C882 train names. Only frame0 initial_box enters initialize; subsequent track calls receive current image only; dataset boxes are written for separate scoring after tracking.

Evidence: `tools/migrate_server26_inputs.py:18`; `scripts/run_rankstop_dev.py:103`; `scripts/run_rankstop_dev.py:119`; `scripts/run_rankstop_dev.py:128`; `dual_rgbt/data.py:15`.

### B_score_normalization: PASS

All seven original toolkit code digests match. Scoring uses existing PR_LasHeR/NPR/SR_LasHeR and sequence averages. NPR normalizes center error by GT width/height; no model-output maximum/mean rescales reported accuracy. Existing toolkit rounding, initialization and invalid-GT conventions remain unchanged.

Evidence: `tools/score_rankstop_dev.py:23`; `scripts/score_tracking.py:58`; `scripts/score_tracking.py:70`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/utils.py:141`.

### C_result_existence_and_native: PASS

No new dev result is claimed. Scorer requires real COMPLETE/exit0 native status, matching task/GPU/native-command/plan/source/checkpoint/cost, all97 sequences and54646 frames, and exact float32 manifest GT. Runner binds native wrapper PID to os.getppid and actual executable/argv. CPU wrappers preserved actual0/7 exits.

Evidence: `scripts/run_rankstop_dev.py:41`; `tools/score_rankstop_dev.py:29`; `tools/score_rankstop_dev.py:49`; `scripts/run_task.py:24`.

### D_metric_live_path: PASS

score_rankstop_dev calls the unchanged score_tracking.main; PR/NPR/SR curves, per-sequence scores, mean/P95 and FPS are written to JSON/NPZ. Queue invokes this scorer for each completed cost.

Evidence: `tools/score_rankstop_dev.py:55`; `scripts/score_tracking.py:58`; `scripts/score_tracking.py:75`; `scripts/queue_rankstop_dev.py:114`.

### E_scope: WARN

This is prospective source approval. Three full dev runs and downstream result audit remain required; the smoke and seeded kernel witness do not establish accuracy. No test metrics, FPS or target achievement have been observed by this review.

Evidence: `reports/rankstop_dev_plan_server26_20261026.json`; `reports/SERVER26_GPU_KERNEL_WITNESS_20261026.json`.

### F_evaluation_type: PASS

Planned evaluation is real_gt. Isolated test fixtures are explicitly synthetic control-flow/arithmetic checks and are not model metrics.

Evidence: `scripts/run_rankstop_dev.py:128`; `.aris/traces/experiment-audit/2026-10-26_server26_source/cpu_verification.json`.

### parameters_timing_hardware_inputs: PASS

Runtime counts total parameters, records actual CUDA device name, template128/main256/local128/wide256/main-preview64/global-preview256. Service timing includes decode, synchronized initialize/track, crop/previews/query/template updates, and reports FPS/mean/P95. Model loading, result I/O, source hashing and reload-probe overhead are excluded. The supplied CPU artifact records103218887 total parameters; no trainable count is required or claimed.

Evidence: `scripts/run_rankstop_dev.py:90`; `scripts/run_rankstop_dev.py:104`; `scripts/run_rankstop_dev.py:147`; `dual_rgbt_global/candidate_tracker.py:21`; `dual_rgbt_global/candidate_tracker.py:65`; `dual_rgbt_global/tracker.py:23`; `reports/SERVER26_MODEL_TOTAL_PARAMETERS_20261026.json`.

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
