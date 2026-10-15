# Recovered candidate completion source review — R1

**SOURCE_FAIL — 1 blocking issue.** Exact R1 plan SHA256: `aa68797aa5b9e8477073fbf4133f4ff6033ba0d31ee9215afb5557fc2d3c8fa3`.

Requested reviewer: gpt-6-astra / max. Actual routing and effort: UNATTESTED. Fresh same-family review; acceptance provisional.

The unchanged controller cannot consume the recovered dependency: it reads `last_observation` before its terminal-state branch, while that field is absent. Exact source assertions reproduced exit1 with `KeyError: last_observation`. This report must not admit an R1 launch.

57 deterministic checks completed; 869 input files remained byte-identical, including all45 candidate sources and all743 collected predecessor run files. The union of64 sealed source files was rehashed;63 Python files parsed using the actual Python3.7 interpreter. Checker terminal: run_03 exit0. Expected source-failure witness: dependency_gate exit1.

## R1-F01 — FAIL: Recovered predecessor omits the exact field consumed before the terminal branch

The recovered receipt has no last_observation. The unchanged candidate controller reads prior['last_observation'] at line 123 before testing COMPLETE_ALL_TRAIN_PIPELINE at line 127. Exact AST extraction of source lines 121-125 against the actual R1 plan/receipt terminates with KeyError: 'last_observation', exit 1. A launch would create local controller output paths before this failure, but would not reach the remote dependency check or formal training launch.

Evidence: `scripts/complete_candidate_benchmarks.py:106`; `scripts/complete_candidate_benchmarks.py:123`; `scripts/complete_candidate_benchmarks.py:127`; `reports/all_train_recovered_status_20261015.json:54`; `C:/Users/gb/.codex_tmp/dual_rgbt_prepare_recovered_candidate_20261015.py:18`; `C:\Users\gb\.codex_tmp\dual_rgbt_controller_recovery_source_review_20261015\dependency_gate.stderr.bin:6`.

Required correction: Create a separate corrected recovery receipt containing an actual saved original COMPLETE/exit0 native observation; bind its new SHA and actual writer identity in a separately reviewed plan. Preserve R1 evidence. No controller or NN source change is needed.

## R1-S01 — PASS: Only recovery metadata and output destinations differ

The original and recovered plan differ at exactly nine top-level keys: source_review, original_status, original_owner_pid, first_dependency_observation_epoch, status_path, private_directory, metrics_directory, local_input_sha256 and recovery. All 45 source seals and the original 13 local input seals are unchanged; only the recovery dependency hash is added. Actual local bytes match all maps in the candidate, reference and latency plans.

Evidence: `reports/candidate_completion_plan.json:51`; `reports/candidate_completion_recovered_plan_20261015.json:51`; `reports/candidate_completion_recovered_plan_20261015.json:65`; `reports/candidate_completion_recovered_plan_20261015.json:68`.

## R1-S02 — PASS: Recovery attribution preserves original ownership and stale state

The preparation source records os.getpid() as recovery writer 22160, and the recovered plan binds that identity. Original owner 16936 is separately retained. The original status remains byte-identical to its saved FULL_TEST snapshot, rather than being rewritten as original-owner completion. All eight original task records and the actual collected full-test object are preserved. Recovery writer process birth/execution was not independently observed by this reviewer; source-level attribution and artifact identity are verified.

Evidence: `C:/Users/gb/.codex_tmp/dual_rgbt_prepare_recovered_candidate_20261015.py:18`; `reports/all_train_recovered_status_20261015.json:3`; `reports/all_train_recovered_status_20261015.json:73`; `reports/all_train_recovered_status_20261015.json:75`; `C:/Users/gb/.codex_tmp/dual_rgbt_recovery_admission_20261015/original_old_status.json:2`.

## R1-S03 — PASS: Old local controllers are absent, including reused PID

The saved admission CIM output contains no matching controller. A single independent local CIM snapshot at 2026-10-15T14:17:34.0520995Z finds no matching Python controller; 16936/33868/21656 are absent, while 31556 is a cmd.exe born 2026-10-15T08:02:35.107516-06:00 running scripts\launch_code_review_mcp.cmd. Numeric PID reuse is not interpreted as the original reference owner surviving. This is a dated local snapshot, not a continuing liveness guarantee.

Evidence: `C:/Users/gb/.codex_tmp/dual_rgbt_recovery_admission_20261015.py:21`; `C:\Users\gb\.codex_tmp\dual_rgbt_controller_recovery_source_review_20261015\local_cim_snapshot.json:2`.

## R1-S04 — PASS: Prior queues are empty and original tasks have exact saved terminal evidence

Saved candidate/reference/latency statuses each have tasks=[] and match their untouched current status bytes and old plan hashes. Eight original native rows match every original task name, PID and physical GPU, with COMPLETE/exit0. These are saved native observations; this review did not query the server or infer current process state from stale status files.

Evidence: `C:/Users/gb/.codex_tmp/dual_rgbt_recovery_admission_20261015/candidate_old_status.json:5`; `C:/Users/gb/.codex_tmp/dual_rgbt_recovery_admission_20261015/reference_old_status.json:5`; `C:/Users/gb/.codex_tmp/dual_rgbt_recovery_admission_20261015/latency_old_status.json:4`; `C:/Users/gb/.codex_tmp/dual_rgbt_recovery_admission_20261015.py:41`.

## R1-S05 — PASS: Saved admission establishes no duplicate formal E1 task or checkpoint

The helper-generated remote program exactly matches the saved remote.py bytes after the observed Windows CRLF encoding. Saved transport exit0 and raw stdout match report.json. That program asserts absent formal native status and absent formal checkpoint directory, source/input equality, protected baseline SHA and unused new metric directories. Both formal paths are also absent locally at this review. Runtime launch still checks the native task directory and GPU memory; remote source/input and terminal-task gates remain. Historical GPU-idle values are not presented as current availability.

Evidence: `C:/Users/gb/.codex_tmp/dual_rgbt_recovery_admission_20261015.py:42`; `C:/Users/gb/.codex_tmp/dual_rgbt_recovery_admission_20261015.py:47`; `C:/Users/gb/.codex_tmp/dual_rgbt_recovery_admission_20261015.py:49`; `scripts/complete_candidate_benchmarks.py:138`; `scripts/complete_pipeline.py:56`; `scripts/complete_pipeline.py:58`.

## R1-S06 — PASS: Renamed operational fields are consumed by the unchanged source

AST-derived use sites: source_review99; local_input_sha256103; status_path106; private_directory108; metrics_directory110/148; first_dependency_observation_epoch114; original_status119; original_owner_pid121. recovery is explanatory metadata and has no control-flow use. Metrics paths flow through dev scoring and uploaded dev receipts; formal run/task/checkpoint and frozen-selection identities remain canonical.

Evidence: `scripts/complete_candidate_benchmarks.py:99`; `scripts/complete_candidate_benchmarks.py:103`; `scripts/complete_candidate_benchmarks.py:106`; `scripts/complete_candidate_benchmarks.py:114`; `scripts/complete_candidate_benchmarks.py:119`; `scripts/complete_candidate_benchmarks.py:148`; `scripts/complete_candidate_benchmarks.py:215`; `scripts/complete_candidate_benchmarks.py:224`.

## R1-S07 — PASS: Neural initialization, training counts and evaluation protocol remain bound

The unchanged training/evaluation plans retain B checkpoint b84d0045..., seed2027, 882 distinct sequences x2 rounds, 917106 tracked frames, 182182 updates and 1764 initializes. Full dev is97/54646 at costs0/.05/.15; test scopes are LasHeR245/220703, RGBT210210/104706 and RGBT234234/116650. Manifest counts were independently read and summed. The existing dev-only selection code remains unchanged. No baseline accuracy rerun, architecture/module change, test selector, environment rebuild or new NN action was performed.

Evidence: `reports/global_candidate_trajectory_plan.json:3`; `reports/global_candidate_trajectory_plan.json:6`; `reports/global_candidate_trajectory_plan.json:951`; `reports/benchmark_evaluation_plan.json:118`; `reports/benchmark_evaluation_plan.json:123`; `scripts/complete_candidate_benchmarks.py:151`; `scripts/complete_candidate_benchmarks.py:212`; `scripts/benchmark_inputs.py:92`.

## R1-S08 — PASS: Collected predecessor completion is byte-bound

All743 copied run files match the saved inventory. Embedded config and COMPLETE objects match local copied files. Two test native records equal the admission rows; their scopes sum to245 sequences/220703 frames. Receipt scorer, metric, curve and frozen-selection SHA values match actual files. This checks completion/admission provenance; raw GT metric recomputation and performance claims remain the separate full-test result audit.

Evidence: `C:/Users/gb/.codex_tmp/dual_rgbt_all882_test_closed_20261015/receipt.json:3`; `C:/Users/gb/.codex_tmp/dual_rgbt_all882_test_closed_20261015/receipt.json:7`; `reports/all_train_recovered_status_20261015.json:55`; `reports/all_train_recovered_status_20261015.json:79`.

## R1-S09 — PASS: Review and source maps bind the exact plan

This report binds the exact R1 plan SHA and reproduces its exact45-entry source map. The existing main requires SOURCE_PASS/0 plus exact plan and map equality before creating outputs. This SOURCE_FAIL/1 therefore correctly blocks R1 at the review gate. The old SOURCE_PASS applies only to the old plan hash and cannot admit this new plan.

Evidence: `scripts/complete_candidate_benchmarks.py:100`; `scripts/complete_candidate_benchmarks.py:101`; `reports/candidate_completion_recovered_plan_20261015.json:3`.

## R1-S10 — PENDING: Downstream recovery plans require later actual-owner rebinding

The original reference and latency plans still bind owners33868/31556 and original paths. They have not been approved as recovered plans. Review must follow the actual new candidate launch and then actual new reference launch, using each real PID/status/plan/review/launch evidence; no future reference/latency pass was written.

Evidence: `reports/reference_completion_plan.json:71`; `reports/e1_paired_latency_plan.json:109`; `scripts/complete_reference_benchmarks.py:44`; `scripts/complete_candidate_latency.py:104`; `scripts/complete_candidate_latency.py:123`.

## Checker provenance and limits

Two auditor-only failures are preserved with stdout, stderr, exact checker snapshots and terminal receipts: run_01 omitted Windows CRLF expansion; run_02 requested an AST option unavailable in the installed Python3.7. Both were corrected solely in private checker code. The substantive missing-field traceback is separately preserved.

No SSH, GPU query, torch import, NN execution, controller launch, experiment-input mutation, Git commit or push occurred. One scoped local CIM observation independently resolved PID reuse. Saved remote state is attributed to its saved observation only. Full-test result audit and later reference/latency actual-owner rebinding remain separate.

Complete input hashes, exact source map, findings and terminal receipts: `reports/RECOVERED_CANDIDATE_COMPLETION_SOURCE_REVIEW.json`.

Private checks: `C:\Users\gb\.codex_tmp\dual_rgbt_controller_recovery_source_review_20261015`.
