# Joint candidate evaluation source review — R2

**Verdict: SOURCE_PASS — 0 blocking issues.** This is source readiness only; runtime and accuracy remain deferred.

Current protocol SHA256: `9ad6d14fd611dbd8226f187d77ed523c646f344388dfbb1138ac5750df3d83f0`.
Current contract SHA256: `990523e1bacaefb348cf344f216845b64aca0f7b895f74f5ec0e29c03b67cf40`.

Requested Astra/max; actual model and effort are UNATTESTED. Review remains same-family/provisional.

## R1 blockers resolved

1. `scripts/joint_evaluation_contract.py:250–253` now binds every referenced runtime config hash to the report artifact map and requires its cost to equal the report cost. Both freeze and frozen-selection validation call this shared checker. The prior counterexample is rejected before an eligibility decision is written.
2. `scripts/joint_evaluation_contract.py:258` now requires exact PR97×51, NPR97×51 and SR97×21 curves. The prior malformed97×21/97×21/97×1 receipt is rejected; a matching-cost full-shape receipt remains accepted.

Eight targeted CPU checks passed, including reproduction of the old acceptance under sealed R1 code, R2 rejection of both mismatched costs, rejection through freeze and selected_configuration, and malformed-curve rejection. Upstream readiness and the valid cost0 verifier result are mocked; all fixtures and audit receipts are explicitly synthetic. No NN, training completion, accuracy, timing or fresh result-audit claim follows from them.

## Binding and review scope

All55 current source hashes match the R2 protocol. Only the contract source digest changed; all other protocol fields, including8 training-plan,10 input and7 toolkit maps, are unchanged. The modified source passes Python3.7 grammar parsing. The actual protocol source-review guard also passes against the authoritative R2 report. R1 deterministic evidence for the unchanged plans, static server26 inputs and toolkit is reused; unrelated suites and remote queries were not repeated.

R1 remains sealed under `.aris/traces/experiment-audit/2026-10-26_joint_candidate_evaluation_source_review/R1_sealed_source/`. Its report SHA is `6944aa51fbca2a6e8d838dd71725fa7be2de48b20ec279abd4f1b29d307d9daf`; protocol SHA is `630f8eadb7d66a1ed337432207c413a73416c9591be312338d302dac9349896f`. R2 regression evidence is `r2_regression_result.json` in the same trace directory.

## Deferred runtime requirements

No active task/log/PID/weight/GPU/waiter was queried, and no audited source or plan was edited by this reviewer. Source PASS does not make any training checkpoint or evaluation plan ready. Preparation still requires actual full-C native COMPLETE/0 and882×2 coverage,182182updates/917106tracks/1764initializes,265 finite same-shape/dtype states with251 frozen and14 changed versus originalB. Then require real11-frame TRAIN strictreload/frozen smoke, full97DEV at3 costs and fresh DEV_RESULT_AUDIT_PASS binding exact plan/checkpoint/three report hashes. Test eligibility remains strictly above oldE1DEV SR56.43314371195134; otherwise NO_TEST. Freeze one checkpoint/cost/maxK across full LasHeR245/220703,RGBT210210/104706 andRGBT234234/116650. Preserve supplied metric math, dataset GT and total103218887 parameters; report actual serviceFPS/mean/P95 with initialization/decode included and model load/file writes/verification excluded.
