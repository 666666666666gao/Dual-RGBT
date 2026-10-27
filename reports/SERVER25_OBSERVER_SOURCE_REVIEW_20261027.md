# Server25 observer source review

**SOURCE_PASS. Blocking issues: 0.**

Reviewer: /root/audit_server25_training_launch_20261027. Same-family, provisional. Requested gpt-6-astra/max; actual model/reasoning UNATTESTED.

Observer SHA256: d2109d0b8a75eace5361df5d9e1b9e5d55ccd037ece00dc8bd2982de52b2ba90

The observer and actual generated remote code parse with Python3.7 grammar. All54 active source files still match the prior review. No remote, NN or training-control call was made.

## schedule_and_single_query: PASS

Binds its exact source hash to the registered node, refuses an existing output, waits locally until observe_at_epoch and then invokes remote exactly once. No periodic remote polling. Node registration itself was not present yet and is outside this source review.

Evidence: tools/observe_server25_training.py:16, tools/observe_server25_training.py:19, tools/observe_server25_training.py:21, tools/observe_server25_training.py:44.

## true_controller_receipt: PASS

Reads the controller native status and requires its PID to equal the node controller_pid. For EIGHT_FORMALS_STARTUP_PASS, requires native COMPLETE and actual exit_code0. Reviewed launch receipt records wrapper209569. Reading a record is not an assertion that the record is already complete at review time.

Evidence: tools/observe_server25_training.py:27, tools/observe_server25_training.py:32, scripts/run_task_four_gpu.py:28.

## eight_job_contract: PASS

For the success execution state, applies unchanged startup to every case. The reviewed launcher produces exactly8 cases. startup verifies wrapper PID and full argv, exactly one native child and its full argv, physical GPU environment, non-sanity config with exact plan/source/input maps, nonempty sequential CSV updates and finite loss values. This validates config source identity; observer does not independently rehash remote source files.

Evidence: tools/observe_server25_training.py:34, tools/launch_server25_training.py:41, tools/launch_server25_training.py:62, tools/launch_server25_training.py:75.

## raw_cpu_and_same_visit_resources: PASS

Returns raw first8 aggregate CPU counters and before/after epochs, samples a minimum10s interval including job-check time, calculates busy percentage excluding idle+iowait, then reads RAM and GPU index/memory/utilization in the same visit. This is a bounded observation interval, not an atomic simultaneous snapshot. No temperature or power query. meminfo_kib also contains kernel count-valued fields; MemAvailable is KiB.

Evidence: tools/observe_server25_training.py:31, tools/observe_server25_training.py:35, tools/observe_server25_training.py:38.

## raw_execution_and_no_overwrite: PASS

Remote returns the exact execution file bytes as base64 plus SHA256. Local decode must match that SHA256. Only a success execution state copies those raw bytes to the canonical local execution file; an existing differing canonical file triggers assertion and is never overwritten. Observer result output is new-only. For a non-success state, the local result retains state/hash but does not retain full raw execution bytes after removing execution_base64.

Evidence: tools/observe_server25_training.py:30, tools/observe_server25_training.py:40, tools/observe_server25_training.py:45, tools/observe_server25_training.py:48.

## read_only_training_isolation: PASS

Generated code performs reads, a10s sleep and read-only nvidia-smi. Importing reviewed helpers defines functions but calls neither launch, worker nor admission. No torch/NN call, process signal, restart, queue control or remote write. All54 active source hashes are unchanged from the previous review. No server26 reference or operation.

Evidence: tools/observe_server25_training.py:23, tools/server25_runtime.py:8, tools/launch_server25_training.py:62.

## generated_code_parse: PASS

Actual formatted remote code for /home/gaob/Dual-RGBT and controller209569 compiles and parses with Python3.7 grammar. The observer module also compiles/parses. Neither the observer main nor generated remote code was executed.

## failure_diagnostic_scope: PASS_WITH_LIMITS

If native and execution JSON exist and execution is not success, tasks stays empty; raw controller/state/resources are returned, with the final7000 log characters when native state is not COMPLETE. Empty tasks does not attest any child job. Missing/malformed JSON, incomplete execution writes, unavailable /proc evidence, startup assertions or subprocess failures abort the observation and may leave no local result JSON; the SSH helper surfaces nonzero stderr. No comprehensive failure recovery or partial-job diagnostic is claimed.

Evidence: tools/observe_server25_training.py:27, tools/observe_server25_training.py:30, tools/observe_server25_training.py:34, tools/observe_server25_training.py:41, tools/server25_runtime.py:15.

## Limits

- SOURCE_PASS authorizes this source interface only; no live controller or training state was observed.
- Non-success execution yields zero validated tasks, not eight-job acceptance.
- Missing files and failed job assertions can prevent any result JSON; diagnostics are fail-fast and limited as described.
- Raw execution bytes are transferred for every successful SSH response but persisted canonically only for EIGHT_FORMALS_STARTUP_PASS.
- Actual model/reasoning are UNATTESTED; review remains same-family/provisional.
