R1 SOURCE_FAIL — 1 blocking issue, preserved after correction.

Plan SHA256: `9716abf0388b5637cfbf2c5e056b1d5b826ff83912235806611989ceb48d3ebd`.
Runner SHA256: `8a17f686d892cb7e1ca47b2f1648bc042ffc14e106d73a642249b60f726496e6`.

The non-daemon sampler was left running when readiness timed out, the barrier failed, or the first joined worker exited1. Remaining workers were not reaped. AST-isolated CPU fixtures confirmed all three paths while the success fixture passed. This could prevent run_task from recording a native failure.

Evidence: `.aris/traces/system-profile/2026-10-26_rankstop_scaling_source_review/R1_CPU_LIFECYCLE_FIXTURES.json`; exact source and plan snapshots are retained alongside it. The final separately bound review verifies the narrow author fix. No GPU/NN/SSH action was performed.

Requested reviewer: gpt-6-astra / max. Actual route: UNATTESTED. Independent context, same-family, provisional acceptance.
