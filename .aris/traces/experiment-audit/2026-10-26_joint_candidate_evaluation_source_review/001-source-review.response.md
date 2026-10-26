# Joint candidate evaluation source review — R1

**Verdict: SOURCE_FAIL — 2 blockers.** Protocol SHA256 `630f8eadb7d66a1ed337432207c413a73416c9591be312338d302dac9349896f`.

Fresh same-family review, provisional. Requested Astra/max; actual model and effort are UNATTESTED. No active jobs, active final checkpoints, waiters or GPUs were inspected. No NN was executed.

## Blocking findings

1. **DEV receipt cost is not bound to executed cost.** `tools/joint_candidate_results.py:133–140` verifies reports and their run artifacts, heavy-token means and FPS but omits report-cost versus config-cost comparison. `scripts/joint_evaluation_contract.py:240–259` has the same omission and is reused by `selected_configuration`. The synthetic CPU fixture returned verified cost0 configs for all three nominal cost0/.05/.15 reports. Actual freeze code accepted them, wrote ELIGIBLE and froze0.15. Legacy `scripts/benchmark_inputs.py:62` already requires this exact binding. Bind each report to its referenced runtime configs in the shared checker.

2. **DEV curve dimensions are not bound to the registered toolkit.** `scripts/joint_evaluation_contract.py:254` checks only97 rows. The unchanged toolkit produces PR97×51, NPR97×51 and SR97×21. A synthetic report carrying PR/NPR97×21 and SR97×1 passes the checker with matching hashes and means. Require those exact shapes; do not change metric math.

The fixture mocks only upstream readiness and the valid cost0 verifier result. Its receipts, curves and audit JSON are explicitly synthetic source-gate inputs. They are not training evidence, accuracy measurements or an independent result audit. Evidence: `.aris/traces/experiment-audit/2026-10-26_joint_candidate_evaluation_source_review/r1_counterexample_result.json`.

## Verified scope

- All55 local source hashes and Python3.7 grammar parses pass. All8 registered plan hashes and their current source-review bindings pass.
- All10 immutable input hashes pass on server26;4 also exist and pass locally. Static SSH read only the explicitly named manifests, historical B/factory/pretrained checkpoints and2 sealed reports. It exited0. No live status, current training checkpoint or PID was read.
- All7 supplied toolkit code hashes match the old evaluation plan. LasHeR delegates to unchanged `score_tracking`;210/234 preserve the legacy metric equations. Dataset GT remains separate from predictions, and saved GT is checked against registered annotations.
- Source checks cover native identity, non-sanity/full counts, exact tensor witness requirements, shard completeness and duplicate rejection, one frozen configuration, total parameters and frame service timing. A broader dynamic mutation suite was deferred after the two decisive counterexamples.

## Deferred runtime and claim boundaries

This source review does not establish final C completion, GPU reload, accuracy, FPS or target achievement. After a corrected source PASS, preparation still requires actual native COMPLETE/0 and full882×2 counts182182updates/917106tracks/1764initializes,265 finite same-shape/dtype states with251 frozen and14 changed versus originalB. Then the final checkpoint needs real11-frame TRAIN smoke with strict reload and frozen-state checks, complete97DEV at3 costs, and a fresh result audit binding plan/checkpoint/exact3 report hashes. Eligibility remains strictly above oldE1DEV SR56.43314371195134; otherwise NO_TEST. One frozen checkpoint/cost/maxK applies across the three full test sets. Existing metric quirks remain unchanged. Runtime reports must retain total103218887 parameters and measured serviceFPS/mean/P95, with initialization and decode included and model load/file writes/verification excluded.
