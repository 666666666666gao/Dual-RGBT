# Executed candidate-query utility

2026-10-14. Native formal task is COMPLETE/exit0. Independent result audit completed524392 checks: deterministic PASS, semantic WARN, zero blocking issues, same-family/provisional. See reports/GLOBAL_CANDIDATE_UTILITY_RESULT_AUDIT.md/.json. This is a fixed-state train20 diagnosis, not deployed tracking performance, a trajectory upper bound or measured inference speed.

The old full-B primary-only tracker supplies8854 tracking steps and1752 labelled states. A separately fitted preview model proposes before current-frame GT. Its259 non-preview tensors match the reference model; both265-tensor models stay unchanged. The run executes31384 raw auxiliary crops, including17444 local128 queries and13940 wide256 queries, totaling8733592 auxiliary tokens. There are no optimizer/backward calls. Counterfactual results never update the reference trajectory.

Native artifacts are `reports/global_candidate_utility_formal_completion.json` and `reports/full_pipeline/global_candidate_utility_formal_20261014/`. Private raw CSV SHA256 is `dd85510eb7e05b2e9c2a7eff2b530f592a097057b0a5323697493ba9107dac90`. CPU analysis independently recomputes every executed clip, IoU, primary IoU, region coverage and signed gain; all maximum errors are zero. It also reproduces the previous main-query boxes and the proposal lists. Magnitude summaries consume only this verified CSV. Raw boxes, GT, coordinates and CSV remain private.

## Candidate choice can matter for actual localization

In the239 states whose GT centre lies outside the main search region, primary IoU is zero. Each row below concerns one query of the indicated family, on the same state. Top1 follows the cheap cosine proposal order. Best-of8 uses dataset GT afterward and is not a deployable selection rule.

| Query choice | Mean executed IoU | Positive gain states | IoU at least0.5 |
|---|---:|---:|---:|
| Main local |0.0006172943|1|0|
| Main alternate local |0|0|0|
| Global local Top1 |0.0337360558|21|8|
| Global local best-of8 |0.2301839757|125|58|
| Global wide Top1 |0.0408293935|20|10|
| Global wide best-of8 |0.3377776673|129|100|

Of the129 positive wide-oracle gains, all129 are at least0.01,125 are at least0.05 and123 are at least0.10. For local best-of8 the corresponding counts are123,120 and115. The opportunity therefore includes substantial IoU changes. This does not show that the currently available policy features can learn which query to select.

Within each family the query resolution and encoder token count are fixed: local128/193 tokens or wide256/385 tokens. The diagnostic actually executes every counterfactual to measure them. It does not constitute a matched deployment-latency comparison or simultaneous multi-query fusion.

## Whether to query also matters

Across all1752 states, mean primary IoU is0.5179512621. Even the best global candidate query, if forced on every state, has negative mean signed gain: local best-of8 −0.0643111569 and wide best-of8 −0.0609063972. Their GT-dependent STOP-inclusive oracle gains are+0.1008473943 and+0.1021804621 respectively. The useful diagnostic question is which request has positive localization value, rather than whether a second modality always helps.

Including both main-search local requests and all available global local/wide requests gives best executed IoU0.6644843554 and mean signed gain+0.1465330933. Its STOP-inclusive oracle gain is+0.1614323696. This mixes193/385-token actions and cannot be used as a same-cost comparison or a full-tracking bound.

## Coverage is not the same selection objective

Among the239 outside states, choosing the wide candidate with highest GT-area coverage yields mean executed IoU0.2347693618; choosing by executed IoU gives0.3377776673. Even the coverage comparator uses GT. The difference establishes that geometric coverage is an imperfect proxy for executed localization value, rather than demonstrating a learned policy or causal reason for its mistakes.

## Next action and remaining evidence

The action set has enough executed oracle opportunity to attempt a single-query benefit selector. The integrity review accepts this diagnostic opportunity with stated limits. A separate candidate policy/tracker/controller-fit implementation now exists; fresh SOURCE, real M0 and formal training remain pending. Reuse the existing955-input scalar benefit head with variable request count and explicit four request-family ids. Score cheap candidate-region features before encoding only the selected raw auxiliary region. Keep the fitted CNN and shared executor frozen while calibrating the quality/gain heads with current proposals and actual signed-IoU labels, covering all882 training sequences; refresh current closed-loop states afterward. See CANDIDATE_POLICY_PROTOCOL.md for the GT-jittered offline fit and planned causal refresh distinction.

Full97-dev calibration,245-test PR/NPR/SR, paired complete-system cost and the three core mechanism controls remain unfinished. Current target78/72/62 and a clear measured cost reduction are not achieved by these diagnostics. The long new global C waits until the original reference pipeline terminates and required GPUs release; no original GPU0 source, owner or checkpoint is changed.
