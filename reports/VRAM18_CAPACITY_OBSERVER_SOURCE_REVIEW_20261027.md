# VRAM18 capacity observer source review

**SOURCE_PASS; blocking issues:0.** Same-family / provisional; actual reviewer model and effort UNATTESTED.

Observer SHA-256: `9b62c9609df49ef580e7d849b992d27593ceff392fb3576b34b41f366f8efda5`.

## bound_delayed_single_measurement — PASS

Requires a new output path, exact previously reviewed serial-controller hash and observer hash bound by the node plan. wait_until precedes every remote call. The main path performs CPU-before, ten-second local delay, one measurement of the registered task list, CPU-after, then one output write. wait_until only sleeps locally; it does not poll the server. There is no retry/restart loop.

Evidence: `tools/observe_vram18_capacity.py:15`, `tools/observe_vram18_capacity.py:16`, `tools/observe_vram18_capacity.py:17`, `tools/observe_vram18_capacity.py:19`, `tools/observe_vram18_capacity.py:22`, `scripts/complete_pipeline.py:30`.

## cpu_program_and_formula — PASS

The exact generated CPU program compiles. busy_percent uses the delta of the first eight aggregate /proc/stat fields and subtracts idle plus iowait; guest counters are not counted again. Nonnegative deltas and positive total are required. Exact formula fixtures give0% idle-only,100% busy-only and50% mixed with large excluded guest deltas. This is aggregate logical-CPU activity, not per-process utilization.

Evidence: `tools/observe_vram18_capacity.py:20`, `tools/observe_vram18_capacity.py:24`, `tools/observe_vram18_capacity.py:25`, `tools/observe_vram18_capacity.py:26`.

## raw_endpoint_preservation — PASS

The observer writes the complete measurement object and appends both raw CPU endpoints/timestamps, actual interval, calculated busy percent, node-plan path and node-plan hash. Original task identity/loss/progress/host-memory records, full meminfo_kib and gpu_csv remain present. CPU sampling brackets the measurement, so the saved interval includes SSH/snapshot overhead and may exceed10 seconds. Host/GPU fields are from one measurement call; they are not claimed simultaneous hardware samples.

Evidence: `tools/observe_vram18_capacity.py:22`, `tools/observe_vram18_capacity.py:26`, `tools/observe_vram18_capacity.py:27`, `tools/observe_vram18_capacity.py:28`, `tools/launch_vram18_serial_training_wave.py:27`, `tools/launch_vram18_serial_training_wave.py:29`, `tools/launch_vram18_serial_training_wave.py:31`.

## unchanged_measurement_and_read_only_scope — PASS

The serial controller and complete_pipeline retain their sealed hashes. measurement still validates each registered RUNNING native PID, child command, contiguous completed CSV rows and finite logged losses, and reads smaps_rollup/meminfo and restricted nvidia-smi fields. remote is unchanged checked SSH with120-second timeout and JSON decoding. The new observer does not launch/stop training, read weights, modify source/plans remotely, or query temperature/power.

Evidence: `tools/launch_vram18_serial_training_wave.py:20`, `tools/launch_vram18_serial_training_wave.py:23`, `tools/launch_vram18_serial_training_wave.py:26`, `scripts/complete_pipeline.py:24`, `tools/observe_vram18_capacity.py:28`.

## inert_main_flow — PASS

Exact main AST executed locally with inert wait/remote/measurement functions and a clearly marked synthetic24-task node. It made exactly two CPU reads and one measurement after the wait, retained raw fields exactly, stored the matching node hash, and recorded the actual13-second synthetic bracket interval. No remote command, NN invocation or wall-time wait was executed by this review.

Evidence: `tools/observe_vram18_capacity.py:13`, `tools/observe_vram18_capacity.py:29`.

The actual node plan must register the intended24 owners and observer hash. CPU is measured across a bracket of at least the requested10-second delay plus snapshot overhead; the actual timestamps are retained. Full raw meminfo/GPU data are available in the same task snapshot. This review does not change score_health or certify a runtime health outcome.

No source was edited, no remote command was executed, and no NN or weight was accessed. Local verification is recorded in `.aris/traces/experiment-audit/2026-10-27_vram18_capacity_observer_source_review/verification.json`.
