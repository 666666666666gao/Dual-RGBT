# Baseline / historical E0 paired-latency source audit — R1

Verdict: **FAIL — two source-readiness blockers.** Requested reviewer: Astra / max. Actual model/effort: UNATTESTED. Same-family, provisional. No SSH, GPU, project-module import, model execution or checkpoint/active-weight read.

1. **F1 (P1): the profiler does not verify its own native wrapper.** `tools/profile_baseline_e0_pair.py:24-35` checks its plan, source and environment, and `:42-66` checks/signals the two training groups, but there is no profiler task/status, expected command or parent-PID admission. `scripts/run_benchmark_tracking.py:160-165` shows the existing native binding pattern. Bind the profile's own task, RUNNING state, GPU, parent PID and exact command before any SIGSTOP, then execute through the detached registered wrapper.
2. **F2 (P2): six actual package initializers are outside the source hash map.** Missing: `dual_rgbt/__init__.py`, `lib/__init__.py`, `lib/models/__init__.py`, `lib/models/layers/__init__.py`, `lib/test/tracker/__init__.py`, `lib/utils/__init__.py`. The model and utility initializers contain executed imports (`lib/models/__init__.py:1`, `lib/utils/__init__.py:1`). Add their current hashes only; no neural source edit is needed.

All 113 declared hashes match and all 112 Python files pass offline AST/compile checks. The 112 inherited source entries are unchanged. The original baseline uses strict SOTep14 loading and its complete causal tracker. Historical E0 uses the exact ee60704e... checkpoint, strict loading, frozen eval, and the official benchmark factories. The prior model source differs only by a default checkpoint path relocation, verified byte-for-byte. Total parameter assertions are 103,138,437 and 103,374,215.

The E0 crop binding is correct on the reviewed import path: earlier imports load global/evidence trackers, never `dual_rgbt.tracker`; `build_model('e0')` sets `data.crop_pair` before `build_tracker` first imports that tracker. Its local binding therefore resolves to `six_channel_crop` (`scripts/run_benchmark_tracking.py:21-35`, `dual_rgbt/tracker.py:6`).

Each phase resets causal state/templates, warms TRAIN frames 1–20 and measures 21–220 for the same three sequences: 600 rows per phase, 2,400 overall. Order is baseline-before / E0 cost0 / E0 cost0.05 / baseline-after. CUDA synchronization correctly brackets decode-inclusive and service timers; initialization is separate. No accuracy, optimizer or trainable-parameter evaluation is called.

Trainer ownership is restricted to the two hash-bound own GPU0 groups, with exact live wrapper/child commands, UID/PGID and compute-client equality before pause. The try/finally restores both groups after ordinary import/load/forward/assertion exceptions and records all four states (`tools/profile_baseline_e0_pair.py:42-68`, `:129-135`). Native execution and post-resume progress remain pending; this audit does not claim current liveness or resilience to uncatchable termination.

Claim ceiling: a loaded-host TRAIN timing sample. Preserve baseline-before/after values and explicitly define any bracket-mean FPS comparison; arithmetic mean FPS is distinct from pooled throughput. Other GPU and CPU jobs remain active. This is not full-test FPS or a result for the current 32 training models.

Plan SHA256: `b484194f296dfd5a9fdebe98fe5a4b0eda6ae92250cfafe0cb1f00fb9c32c4d6`

Profiler SHA256: `0d737c4a83a7f841657b5282921b3fbb2fd82bd61dade2f032c590839fde228b`

See `round1.review.json` for complete evidence and exact input/source hashes; `deterministic_source_checks.json` for compilation and import-closure results. The source guard payload is deliberately SOURCE_FAIL and is written only inside this trace.
