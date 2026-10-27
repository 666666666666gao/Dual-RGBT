# VRAM18 wave4 and host-memory result audit

**Verdict: WARN. Deterministic saved-evidence checks: PASS. No blocking issues.**

Date: 2026-10-27. Fresh finishing reviewer: same-family, provisional; requested gpt-6-astra/max, actual model UNATTESTED. This audit reads local saved evidence only: no remote calls, active-run access, weights, or new experiments. The existing CPU audit is cited evidence; this reviewer did not execute it.

The records support eight native0 wave4 admissions and a historical staged-capacity decision. They do not establish completed formal training, full-training equivalence, accuracy, online FPS, a controlled aggregate speedup, or material malloc_trim benefit.

## Gt Provenance: PASS

TRAIN admission uses data/manifest.json train boxes; current-frame boxes enter label generation after track advances. Saved admissions are one sequence (11runthree), four updates and twenty tracked frames. No benchmark accuracy is inferred; dataset pixels/annotations were not reloaded in this finishing audit.

Evidence: `scripts/train_candidate_joint_quality_trajectory.py:186`; `scripts/train_candidate_joint_quality_trajectory.py:297`; `scripts/train_candidate_joint_gainpair_trajectory.py:182`; `scripts/train_candidate_joint_gainpair_trajectory.py:294`; `runs/joint_quality_k4_cost0p0125_vram18_20261027_sanity/COMPLETE.json:2`.

## Score Normalization: PASS

Throughput is logged update delta divided by saved elapsed time; capacity ratio uses the earlier eight-task aggregate. PSS max is a sampled resource maximum, not a prediction-derived score denominator. Numerical comparison is explicitly differential verification, not ground-truth accuracy.

Evidence: `tools/launch_vram18_training_wave.py:39`; `tools/launch_vram18_training_wave.py:46`; `tools/launch_vram18_training_wave.py:50`; `tools/profile_host_allocator_r4.py:72`.

## Result Existence: PASS

All 88 collected raw metadata files match SHA-256 and byte counts (270491 bytes total). Eight wave4 admissions and four later memory probes have native COMPLETE/0. The R3 run remains failed despite its child COMPLETE/SANITY files. Embedded wave4 admission receipts and formal admission seals match raw files. All wave4 rates, retention and swap growth recompute.

Evidence: `reports/VRAM18_COMPLETED_ADMISSION_METADATA_COLLECTION_20261027.json:2`; `reports/VRAM18_COMPLETED_ADMISSION_METADATA_COLLECTION_20261027.json:357`; `reports/VRAM18_WAVE4_HEALTH_20261027.json:8`; `reports/HOST_ALLOCATOR_DEFAULT_SSH_20261027.json:3`; `reports/HOST_ALLOCATOR_R3_RUNTIME_FAILURE_DIAGNOSIS_20261027.json:109`.

## Dead Code: PASS

Saved native commands, emitted startup/warm memory records and saved numerical-comparison output demonstrate use of the audited paths. This audit inspects those records and source; it does not re-execute training or a weight comparison.

Evidence: `scripts/train_candidate_host_compact.py:17`; `scripts/train_candidate_host_compact.py:24`; `scripts/train_candidate_host_warm_compact.py:17`; `scripts/train_candidate_host_warm_compact.py:26`; `tools/profile_host_allocator_trim.py:116`; `tools/profile_host_allocator_warm.py:116`; `.aris/traces/experiment-audit/2026-10-27_vram18_wave4_result_review/completed_cpu_verification.json:153`.

## Scope: WARN

Single-seed four-update TRAIN admissions and historical capacity samples only. Numerical tolerance is not optimizer bit equality or full-training equivalence. Reused default controls, sequential profiling, evolving training states, two-second PSS samples and reload-clone peaks preclude clean-room aggregate, steady-state peak or malloc_trim benefit claims. MemAvailable is an extra score-time read with no exact endpoint timestamp.

Evidence: `reports/VRAM18_WAVE4_HEALTH_20261027.json:43`; `tools/launch_vram18_training_wave.py:58`; `tools/observe_vram18_training.py:55`; `reports/HOST_ALLOCATOR_TRIM_PROFILE_PLAN_20261027.json:233`; `reports/HOST_ALLOCATOR_WARM_PROFILE_PLAN_20261027.json:339`; `tools/profile_host_allocator_r4.py:72`.

## Evaluation Type: PASS

Dataset-GT-backed TRAIN sanity plus engineering resource instrumentation and saved numerical regression checks; no completed benchmark, accuracy evaluation, online FPS measurement, or paper novelty evaluation.

Evidence: `scripts/train_candidate_joint_quality_trajectory.py:187`; `scripts/train_candidate_joint_quality_trajectory.py:332`; `reports/VRAM18_WAVE4_HEALTH_20261027.json:43`.

## Recomputed wave4

All 16 saved task identities match between the start and end (wrapper PID, child PID, GPU and command), with four per GPU, positive update deltas and finite logged losses. These are historical observations, not a current live-state assertion.

| Quantity | Recomputed value |
|---|---:|
| Interval | 1011.1337263584137 s |
| Total update delta | 15,875 |
| Aggregate | 15.700198288483193 updates/s |
| Earlier eight-task reference | 168,725 / 17019.99653530121 = 9.91333926831578 updates/s |
| Historical aggregate/reference ratio | 1.5837446760914262 |
| Minimum original-task rate retention | 0.8754537034556423 (87.5454%) |
| Sum of trainer Swap, start / end | 6,932,716 / 6,959,732 KiB |
| Swap increase | 27,016 KiB |
| Extra score-time MemAvailable | 51,165,932 KiB; exact timestamp unavailable |

The operational gate recomputes to ADMIT_NEXT_WAVE using the recorded MemAvailable. That value came from a separate read during scoring, not the timed raw health node. The exact read time and an independent raw endpoint receipt are absent. This is a reporting limitation, not an invented blocker.

Evidence: `reports/VRAM18_WAVE4_EXECUTION_20261027.json:1934`; `reports/VRAM18_WAVE4_HEALTH_NODE_20261027.json:2`; `reports/VRAM18_WAVE4_HEALTH_20261027.json:8`; `tools/launch_vram18_training_wave.py:35`; `tools/launch_vram18_training_wave.py:58`.

Measurement-start GPU use was 9447, 9484, 9439 and 9598 MiB. Wave4 does not establish that each card reached about 18 GiB. The 1.5837 ratio compares different intervals and counts of evolving training jobs. It is an operational aggregate ratio, not a paired clean-room speedup.

## Completed raw admission evidence

All 88 collected metadata files match the manifest SHA-256 and byte length, totaling 270,491 bytes. Eight wave4 admissions and four later memory probes have native COMPLETE/0. Each records four optimizer updates, twenty tracked frames, one TRAIN sequence (11runthree), seed 2027, 251 frozen and 14 changed tensors, plus strict reload/replay checks. The eight wave4 admission seals match the formal launch arguments. All 63 unique source files in the saved source-check inventory still match their expected hashes. This does not claim their formal runs completed.

R3 is explicitly excluded from successful admissions: its child wrote COMPLETE and SANITY, but the observer encountered ProcessLookupError, SSH returned 1, its owned wrapper was reaped with -9, and native RUNNING remained stale. Later probes do not erase that failure.

Evidence: `reports/VRAM18_COMPLETED_ADMISSION_METADATA_COLLECTION_20261027.json:2`; `reports/VRAM18_COMPLETED_ADMISSION_METADATA_COLLECTION_20261027.json:357`; `reports/HOST_ALLOCATOR_DEFAULT_SSH_20261027.json:3`; `reports/HOST_ALLOCATOR_R3_RUNTIME_FAILURE_DIAGNOSIS_20261027.json:109`. Per-run raw evidence and all 88 hashes are in the JSON report.

## Memory samples and numerical comparison

| Probe | Samples | Sampled peak PSS (GiB) | Last sampled PSS (GiB) | Native elapsed (s) |
|---|---:|---:|---:|---:|
| R4 default | 28 | 6.823671 | 5.376710 | 59.861522 |
| R4 compact | 35 | 5.548796 | 3.757719 | 74.134095 |
| Post-load trim | 32 | 5.546911 | 3.758902 | 68.062565 |
| Post-update warm trim | 53 | 5.838279 | 2.994295 | 110.497651 |

Samples are about two seconds apart and include load/reload clones. They are neither continuous peaks nor formal steady-state measurements. The later probes explicitly reuse the completed R4 default as their reference; they do not provide a new contemporaneous control. Elapsed times are recorded probe durations, not training speedup measurements.

Immediate post-load malloc_trim released only 82 KiB PSS/96 KiB RSS; after four updates it released 116 KiB of each. Return value 1 alone does not establish useful memory savings. Differences in last sampled or peak PSS cannot be attributed to a material malloc_trim benefit.

The saved CPU audit reports all 265 network tensors and both trajectory/query-label CSV files bit equal against R4 default. The optimizer state (42 tensors checked) is **not bit equal**: maximum absolute differences are 4.656612873077393e-10 (compact), 9.313225746154785e-10 (post-load trim), and 4.656612873077393e-10 (warm trim). All pass atol 1e-8/rtol 1e-6. These are four-update numerical checks, not full-training equivalence. The original R4 exact-optimizer requirement remains failed.

The saved CPU SSH receipt exits 0 and its parsed stdout exactly matches completed_cpu_verification.json. Its native statuses and metadata hashes match the collected raw records, and its stored weight/label hashes match the COMPLETE records. This finishing audit checked that receipt consistency without loading any weights or private query-label files. Stored state-element count includes buffers and is not substituted for the model parameter count.

Evidence: `reports/HOST_ALLOCATOR_R4_DEFAULT_SAMPLES_20261027.json:114`; `reports/HOST_ALLOCATOR_R4_DEFAULT_SAMPLES_20261027.json:340`; `reports/HOST_ALLOCATOR_R4_DEFAULT_SAMPLES_20261027.json:342`.
Evidence: `reports/HOST_ALLOCATOR_R4_COMPACT_SAMPLES_20261027.json:114`; `reports/HOST_ALLOCATOR_R4_COMPACT_SAMPLES_20261027.json:396`; `reports/HOST_ALLOCATOR_R4_COMPACT_SAMPLES_20261027.json:398`.
Evidence: `reports/HOST_ALLOCATOR_TRIM_COMPACT_SAMPLES_20261027.json:115`; `reports/HOST_ALLOCATOR_TRIM_COMPACT_SAMPLES_20261027.json:373`; `reports/HOST_ALLOCATOR_TRIM_COMPACT_SAMPLES_20261027.json:375`.
Evidence: `reports/HOST_ALLOCATOR_WARM_COMPACT_SAMPLES_20261027.json:115`; `reports/HOST_ALLOCATOR_WARM_COMPACT_SAMPLES_20261027.json:541`; `reports/HOST_ALLOCATOR_WARM_COMPACT_SAMPLES_20261027.json:543`.
Evidence: `reports/HOST_ALLOCATOR_TRIM_STARTUP_MEMORY_20261027.json:5`; `reports/HOST_ALLOCATOR_TRIM_STARTUP_MEMORY_20261027.json:29`; `reports/HOST_ALLOCATOR_TRIM_STARTUP_MEMORY_20261027.json:51`.
Evidence: `reports/HOST_ALLOCATOR_WARM_MEMORY_RELEASE_20261027.json:5`; `reports/HOST_ALLOCATOR_WARM_MEMORY_RELEASE_20261027.json:29`; `reports/HOST_ALLOCATOR_WARM_MEMORY_RELEASE_20261027.json:51`.

Additional evidence: `.aris/traces/experiment-audit/2026-10-27_vram18_wave4_result_review/completed_cpu_verification.json:153`; `.aris/traces/experiment-audit/2026-10-27_vram18_wave4_result_review/completed_cpu_verification.json:295`; `tools/profile_host_allocator_r4.py:102`; `reports/HOST_ALLOCATOR_R4_COMPARISON_DIAGNOSIS_20261027.json:2`; `reports/HOST_ALLOCATOR_TRIM_PROFILE_PLAN_20261027.json:233`; `reports/HOST_ALLOCATOR_WARM_PROFILE_PLAN_20261027.json:339`.

## Claim handling

- Supported: completed four-update admissions, preserved native failures, exact raw metadata seals, recomputed historical wave4 counts/rates, and the explicitly saved numerical-tolerance results.
- Qualified: the historical 1.5837 aggregate/reference ratio, sampled PSS values, and score-time MemAvailable without an exact timestamp.
- Unsupported by this evidence: full-training equivalence, optimizer bit equality, material malloc_trim benefit, controlled aggregate speedup, approximately 18 GiB on all cards at wave4, benchmark accuracy, or online FPS.

No new run or approval is required to finish this saved-evidence audit. Retain these qualifications whenever the numbers are reused.
