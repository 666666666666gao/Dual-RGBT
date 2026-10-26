# RankSTOP saved TRAIN decision audit

Date: 2026-10-26. **Overall verdict: PASS within the saved TRAIN diagnostic scope; 0 blocking issues.**

Fresh delegated experiment-audit review. Requested reviewer: `gpt-6-astra`, reasoning `max`; actual model and effort route: **UNATTESTED**. Review independence: **same-family**. Acceptance: **provisional**. Reviewer owns this report and the independent recount; it did not modify or rerun the diagnostic producer. The deterministic arithmetic checks below are separate from semantic reviewer acceptance.

## Verified results

All **45/45 numerical fields** in `RANKSTOP_TRAIN_DECISION_DIAGNOSTIC_20261026.json` match the independently vectorized recount exactly. All **24 conditional groups**, including **120 counts and 24 means**, match `RANKSTOP_CONDITIONAL_QUERY_DIAGNOSTIC_20261026.json` exactly.

The census covers **182,182 update states**, **399,401 label rows**, **882 TRAIN sequences**, two rounds of **91,091** updates, seed **2027**, and valid frames at stride **5**. Both rounds match the hash-bound plan's exact sequence order and complete valid stride-5 frame order. There are 364,364 sampled labels and 35,037 selected-extra labels. The archived manifest has 459,435 TRAIN annotation frames; its expected two-round tracking count is 917,106 and initialization count is 1,764.

All 882 installed original TRAIN `init.txt` files match every original-manifest annotation after float32 conversion, without coordinate transformation. Independently computed **798,802 IoUs** and **399,401 gains** match saved values **exactly** (maximum absolute error 0). The verifier uses the original-image xywh intersection/union formula in NumPy float32, without importing Torch. Checking the saved float32 delta using float64 subtraction produces at most 2.980232238769531e-08 rounding residual; this is not a GT mismatch.

| Quantity | All | Round 1 | Round 2 |
|---|---:|---:|---:|
| Update states | 182182 | 91091 | 91091 |
| Executed queries | 44443 | 23260 | 21183 |
| Selected-extra outcomes | 35037 | 18249 | 16788 |
| Reused sampled outcomes | 9406 | 5011 | 4395 |
| STOP states | 137739 | 67831 | 69908 |
| Helpful executed queries, gain > .01 | 14535 | 7553 | 6982 |
| Harmful executed queries, gain < -.01 | 11057 | 5849 | 5208 |
| STOP with helpful sampled same-role alternative | 28630 | 14324 | 14306 |
| Mean primary routing regret | .04924217114643316 | .05064848304219745 | .04783585925066887 |
| Mean gain per executed query | .016101530431867832 | .014529244591109765 | .017827979407746258 |
| Mean reconstructed final IoU | .5211649696710093 | .5211670728809975 | .5211628664610211 |

The executed-query split is exact: selected-extra contributes 8,522 harmful / 11,110 helpful / 15,405 neutral outcomes; reused-sampled contributes 2,535 / 3,425 / 3,446. Every selected-extra flag, chosen modality, action kind, rank, same-role primary box/geometry/IoU, and reused-sampled match agrees. Both CSVs are fully exhausted, with contiguous step IDs and no duplicate round/sequence/frame keys. Every sampled pair has exactly one role-0 and one role-1 label and identical primary geometry.

## A–F checks

**A. Ground-truth provenance — PASS.** TRAIN rows come from `data/manifest.json` (`scripts/train_candidate_rankstop_trajectory.py:285`). Frame-0 GT initializes each trajectory (`:167`); current-frame GT is supplied to the teacher only after `tracker.track(image)` (`:173`, `:181`). `sampled_targets` computes executed primary and query IoU against that target (`scripts/train_candidate_evidence_policy.py:57`, `:72`, `:77`); `write_label` records those results (`scripts/train_candidate_rankstop_trajectory.py:141`). The original manifest SHA matches the training config, migrated TRAIN arrays/order/valid frames are identical, and all raw TRAIN text annotations and all saved IoUs/gains were independently checked. `dual_rgbt_evidence/regions.py:43` uses uncensored original-image GT.

**B. Score normalization — PASS.** Standard IoU uses geometric intersection divided by union (`dual_rgbt_evidence/regions.py:45`). There is no rescaling by prediction extrema or prediction-score statistics. The diagnostic divides gain by executed-query count for `mean_gain_per_executed_query`, and all other sums by update count (`tools/analyze_rankstop_training_decisions.py:72`, `:75`). Regret uses the better of two same-state primary IoUs as a comparison, not as a normalization denominator (`:36`, `:37`).

**C. Result existence and numbers — PASS for the audited diagnostic.** Both raw CSVs, config, and COMPLETE receipt exist and match their reported SHA-256 values. All 45 diagnostic values and 144 conditional-group statistics match exactly. `COMPLETE` counts and the older completion audit agree with the CSV/manifest census. The older completion audit's checkpoint/tensor/optimizer assertions were not re-audited because checkpoint loading was excluded; its missing native exit receipt remains unknown (`reports/RANKSTOP_C882_COMPLETION_AUDIT_20261026.json:36`). Full-frame route totals are internally consistent (917,106 frames and 223,466 queries), but this audit does not possess an independent per-frame runtime log for them.

**D. Dead code and derivation — PASS within the inspected paths.** The producer invokes `main` (`tools/analyze_rankstop_training_decisions.py:90`), reaches every reported aggregation, and its outputs are independently reproduced. The training path invokes sampled targets, selected replay, optimizer update, and label writing in order (`scripts/train_candidate_rankstop_trajectory.py:184`, `:186`, `:191`, `:202`, `:209`). Labels describe the pre-update decision even though they are written after that update. No original producer invocation receipt was supplied; numerical reproduction and static reachability do not attest that historical process invocation.

**E. Scope — PASS only as explicitly qualified TRAIN diagnostics.** The output itself restricts its claim (`reports/RANKSTOP_TRAIN_DECISION_DIAGNOSTIC_20261026.json:60`). This is one seed and one training trajectory schedule over two evolving rounds, with update-weighted, temporally dependent observations. It is neither final fixed-weight performance nor official PR/NPR/SR or an estimate over all tracked frames. No learning-improvement claim follows from round differences alone.

**F. Evaluation type — PASS, `real_gt`.** Dataset-provided TRAIN annotations determine the target; model outputs are predictions. This classification does not turn in-training descriptive measurements into held-out performance evidence.

## Denominators and permitted interpretation

- Primary routing comparisons use all 182,182 update states: the other primary is better by > .01 on 59,246 (32.520227%) and by > .1 on 29,479 (16.181072%). Mean same-state routing regret is .04924217114643316 IoU.
- Harmful/helpful executed-query fractions use 44,443 queries: 11,057/44,443 = 24.879059%, and 14,535/44,443 = 32.704813%. Mean gain is .016101530431867832 per executed query; .003927941931604122 averages the same gain, including STOP zeros, across all updates.
- Helpful sampled STOP alternatives use 137,739 STOP states: 28,630/137,739 = 20.785689%. This examines one sampled request for the chosen role, not every available query.
- Same-state primary comparisons, selected-query gains, and sampled STOP alternatives are descriptive GT oracle diagnostics. They do not establish a deployable GT gate, a causal module benefit, a full candidate-pool oracle, a trajectory upper bound, or a reason to disable a query family based on its unconditional average.

## Preserved provenance findings and limits

1. **Current-source mismatch resolved, not concealed.** The current training scripts differ from config hashes. Exact historical evidence-policy bytes were recovered from local Git history; trajectory bytes were recovered from commit `4898736ce0ee77f829d8014d2b4faf8b1671f079` with the historical CRLF representation. Their hashes exactly match the training config. Compared as text, each current file differs only in two protected-checkpoint path substitutions from `/home/gaob/CADTrack_ep0014.pth.tar` to the project-local checkpoint path. The label, decision, training-order, and arithmetic logic is unchanged. Source-history evidence is preserved in the trace. No Git state was changed.
2. **Same-history correspondence is source-backed.** Pre-track state/templates/history are copied at `scripts/train_candidate_rankstop_trajectory.py:172`; teachers use those saved values; selected replay asserts route/geometry/executed-box equality (`:47`, `:54`, `:64`, `:67`, `:82`); teacher-state preservation is asserted (`:188`, `:189`). Saved keys, primary geometries, and selected-role boxes align for every update. The CSVs do not archive per-step templates/history/weights, and this audit did not rerun neural replay, so it does not independently attest those runtime tensors or assertions.
3. **Historical terminal status remains unknown.** No checkpoint was loaded, no native exit status invented, and no prior frozen-tensor/optimizer claim was silently upgraded to a fresh verification. The new PASS is for the saved decision diagnostic, not a replacement for the older completion audit.
4. **Scope stays TRAIN only.** No images, active dev results/logs, resource/PID queries, neural runs, checkpoint loads, or job operations occurred. Source files and raw artifacts were read only. Review routing is UNATTESTED and same-family/provisional.

No code fix is required for these saved diagnostics. Keep these qualifiers whenever reporting the numbers; final fixed-weight evaluation requires separate evidence.

## Hash-bound evidence

- Audited diagnostic: `2dc047b90a5003104e3b35ed3614726f050bca93c8f9e07859c79f68a3b51d5c`.
- Diagnostic producer: `cc0530cb1d35c7aeb51d3047f062be46ab0ac30d2b5a9c3b6269ec5c9c6ae9b5` (local and server bytes equal).
- Raw labels: `6ddfdb92feae3294ea85211a672fe6bf1b07cdcdd5a8abc1dfdf38b908bbfa47`.
- Raw decisions: `2f22103097fdcaeb4483841cf5456160938a2ba0baadf393a7d70317f0a14e50`.
- COMPLETE: `fd2c8c534a5b22107971aa90133db0866db1fbc863494678769c36fcb6d17171`.
- Config: `5b8baf3f1a6d29c9f7d17ea06086fadb4673e4cc8584c81facfb9efe66f9c190`.
- Original manifest: `389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc`; migrated manifest: `dfc64432210b053fc9db517bae0771fdeb750a5d373b732949ef43e8b65b1a7a`.
- Plan: `a9a537a734511d0e7639ac050f01f97698af45881d29c38aca839ff59b875c08`.

Full inputs, original-GT file hashes, check counts, reconstructed groups, source-history comparisons, and audit-source hashes: `.aris/traces/experiment-audit/2026-10-26_run01/`. Machine-readable findings: `reports/RANKSTOP_TRAIN_DECISION_AUDIT_20261026.json`. Audit-output hashes are sealed in the trace's `output_seals.json` to avoid self-referential hashes.
