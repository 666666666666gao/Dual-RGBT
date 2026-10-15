# Recovered candidate paired-latency source review

**SOURCE_PASS — 0 blocking issues.** Plan SHA256 `2d24df1dfc781e4c9a7478e3280575278e6333b91a47d5ed67116395b39b9d25`.

Requested gpt-6-astra/max; actual routing and effort UNATTESTED. Same-family/provisional.

The recovered latency plan binds actual reference worker41568 and candidate worker33196. The first dependency read is exactly reference next+120 seconds. All64 source files and19 local input bindings match;23 narrow CPU checks passed with actual exit0 and89 input seals unchanged. Original waiting, full-queue/native completion, strict-reload and paired timing semantics remain intact.

## LAT-R01 — PASS

The plan differs only in ten recovery keys: source_review, prior_status, prior_owner_pid, e1_status, e1_owner_pid, first_dependency_observation_epoch, status_path, private_directory, local_input_sha256 and recovery. All64 source-map entries and actual source bytes match the original reviewed latency plan. Nineteen local seals retain the original13 common inputs and replace only the candidate/reference plan-review-owner triples. The entire remote input_sha256 map, expected_full counts and original timing revision remain unchanged.

Evidence: `reports/e1_paired_latency_recovered_plan_20261015.json:70`; `reports/e1_paired_latency_recovered_plan_20261015.json:91`; `reports/e1_paired_latency_recovered_plan_20261015.json:109`; `reports/e1_paired_latency_recovered_plan_20261015.json:117`; `reports/e1_paired_latency_recovered_plan_20261015.json:125`.

## LAT-R02 — PASS

Reference dependency binds actual worker41568 and preserves launcher43612 separately. Saved launch receipt identifies WAIT_E1_PIPELINE/tasks[] and the exact reference script/recovered plan command. Producer launcher reads the os.getpid-written state, queries that worker, validates ParentProcessId against its launcher and validates command/state before writing the receipt. The saved receipt hashes match reference plan cee35452... and SOURCE_PASS ae70013a.... No new CIM or current-state read was made.

Evidence: `reports/reference_completion_recovered_launch_20261015.json:2`; `reports/reference_completion_recovered_launch_20261015.json:5`; `reports/reference_completion_recovered_launch_20261015.json:8`; `reports/reference_completion_recovered_launch_20261015.json:10`; `C:/Users/gb/.codex_tmp/dual_rgbt_launch_recovered_downstream_20261015.ps1:29`; `C:/Users/gb/.codex_tmp/dual_rgbt_launch_recovered_downstream_20261015.ps1:31`.

## LAT-R03 — PASS

Candidate dependency remains actual worker33196, linked to recovered R2 plan803c18a5... and SOURCE_PASS233209f8.... Both new status paths match their actual owner receipts and reviewed plans. Latency first read1792082095.116463 equals saved reference next1792081975.116463+120 exactly.

Evidence: `reports/candidate_completion_recovered_owner_20261015.json:2`; `reports/candidate_completion_recovered_owner_20261015.json:10`; `reports/e1_paired_latency_recovered_plan_20261015.json:109`; `reports/e1_paired_latency_recovered_plan_20261015.json:113`; `reports/reference_completion_recovered_launch_20261015.json:12`.

## LAT-R04 — PASS

Exact original source104/117/118/121 executes against fields taken from the saved reference launch receipt and a virtual clock. Its WAIT_E1 state keeps terminal branch105 false, tolerates absent last_observation, and schedules the registered node. These receipt-backed CPU objects are not a claim about current live state. Exact E1 gate123 also rejects the historical saved FULL_C882 state if reached; it requires COMPLETE_E1_THREE_BENCHMARKS and owner33196.

Evidence: `scripts/complete_candidate_latency.py:104`; `scripts/complete_candidate_latency.py:105`; `scripts/complete_candidate_latency.py:117`; `scripts/complete_candidate_latency.py:118`; `scripts/complete_candidate_latency.py:121`; `scripts/complete_candidate_latency.py:123`.

## LAT-R05 — PASS

Unchanged main waits locally until COMPLETE_E0_THREE_BENCHMARKS with all three dataset counts, then requires terminal E1. Its first remote admission checks the union of both task lists for COMPLETE/exit0/exact PID plus source/input seals before timing launch. Consequently no SSH/GPU work is reached while the reference dependency is still waiting, and no timing job is admitted before both queues and native tasks are complete.

Evidence: `scripts/complete_candidate_latency.py:105`; `scripts/complete_candidate_latency.py:107`; `scripts/complete_candidate_latency.py:123`; `scripts/complete_candidate_latency.py:124`; `scripts/complete_candidate_latency.py:129`; `scripts/complete_candidate_latency.py:134`; `scripts/complete_candidate_latency.py:148`.

## LAT-R06 — PASS

Existing strict checkpoint/selection binding, protected CAD baseline identity, canonical argv, config/CSV hashes, real smoke-reload prerequisite and paired full/warm timing validation are byte-identical and remain covered by the old SOURCE_PASS. Full scope stays97 sequences/54646 frames/53773 warm/54549 tracked. Timing task/run names, GPU0, alternating independent online state/decoding, service/tracking measurements and no-baseline-accuracy protocol are unchanged. NN math and full timing/result audits were not repeated.

Evidence: `scripts/complete_candidate_latency.py:28`; `scripts/complete_candidate_latency.py:30`; `scripts/complete_candidate_latency.py:35`; `scripts/complete_candidate_latency.py:39`; `scripts/complete_candidate_latency.py:74`; `scripts/complete_candidate_latency.py:144`; `scripts/complete_candidate_latency.py:156`; `reports/e1_paired_latency_recovered_plan_20261015.json:117`.

## LAT-R07 — PASS

New status and private collection paths are unused locally and are consumed by unchanged source93-95. Exact source87-90 review-admission assertions pass on this SOURCE_PASS/0, exact recovered plan2d24df1d... and exact64-entry map. The runtime forwards the recovered plan path into timing argv; that same plan must be deployed at the corresponding remote relative path before the future timing job. Remote deployment or availability is not newly attested by this review.

Evidence: `scripts/complete_candidate_latency.py:87`; `scripts/complete_candidate_latency.py:88`; `scripts/complete_candidate_latency.py:93`; `scripts/complete_candidate_latency.py:95`; `scripts/complete_candidate_latency.py:146`; `scripts/complete_candidate_latency.py:148`.

No SSH, GPU, NN, CIM, current-state query, controller launch, experiment-input mutation or Git write occurred. Prior sealed protocol evidence is reused. Real future smoke/full timing and deployment of the recovered plan remain runtime work; this source pass makes no timing or performance claim.

Complete hashes and terminal receipt: `reports/RECOVERED_CANDIDATE_LATENCY_SOURCE_REVIEW.json`; private checks: `C:\Users\gb\.codex_tmp\dual_rgbt_controller_recovery_source_review_20261015\latency`.
