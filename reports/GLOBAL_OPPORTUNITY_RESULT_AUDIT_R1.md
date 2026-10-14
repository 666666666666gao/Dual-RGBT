# Global opportunity result audit

Date: 2026-10-14T03:24:04.605395+00:00. Overall **WARN**; deterministic arithmetic **PASS**; blocking issues **0**.
Independent delegated file review under experiment-audit / local-codex-policy. `review_independence=same-family`, `acceptance_status=provisional`, `actualRouting=UNATTESTED`. The policy requests gpt-6-astra / max; this audit has no actual routing attestation.

The 24,528 stored IoUs, all deltas and all 395 reported numeric fields match independent computation exactly. The report preserves the limited train-state interpretation and records two nonblocking document corrections.

## Deterministic evidence

- 621 named checks passed, plus 6 execution-scope checks: **627 / 627**.
- 20 sequences; 8,854 tracked frames; 1,752 valid labelled states; 24,528 action rows; 1,752 complete 14-combination matrices.
- Every state has both primary roles × five executable actions and four controls only for the selected role. Selected RGB/TIR roles: 1,501 / 251.
- All 24,528 scalar float32 IoUs and all 24,528 Python delta subtractions are exact; maximum error is 0. The 395 compared native/analysis numeric fields are also exact.
- Actual scope is the first20 entries of the identical frozen all_train/evidence/global 882-sequence execution orders, whose membership equals canonical train882. It is not the first20 entries of the manifest original ordering.
- All31 sealed source hashes match. Five original formal/sanity JSON byte hashes and the private CSV hash match their collector bindings. Postseal: 63 recorded audited files unchanged.

The checker uses standard library plus cached NumPy1.26.4 and explicit scalar float32 arithmetic; it never calls/imports the author analyzer, torch or a model. Main checker native exit0: `cmd_1791947675359097600_checker_r2.exit.json`; source/order addendum native exit0: `cmd_1791947748347951900_scope_addendum.exit.json`.

## Selected-role means on all 1,752 states

| Action | Auxiliary tokens | Mean IoU | Mean ΔIoU | Positive / negative / zero |
|---|---:|---:|---:|---:|
| STOP | 0 | 0.438021928912 | +0.000000000000 | 0 / 0 / 1752 |
| LOCAL_MAIN | 193 | 0.480857090006 | +0.042835161094 | 830 / 502 / 420 |
| LOCAL_ALTERNATE | 193 | 0.474299520965 | +0.036277592053 | 752 / 598 / 402 |
| GLOBAL_LOCAL | 193 | 0.159507260931 | -0.278514667981 | 397 / 988 / 367 |
| GLOBAL_WIDE | 385 | 0.224177654849 | -0.213844274063 | 427 / 967 / 358 |
| DENSE_SAME_SEARCH | 385 | 0.410003380465 | -0.028018548447 | 498 / 853 / 401 |
| RANDOM_LOCAL | 193 | 0.081193709403 | -0.356828219509 | 207 / 1162 / 383 |
| GT_CENTER_LOCAL | 193 | 0.594217657433 | +0.156195728521 | 1184 / 512 / 56 |
| GT_CENTER_WIDE | 385 | 0.500305812443 | +0.062283883531 | 794 / 827 / 131 |

The native all-role summary includes 3,504 rows for each executable action and 1,752 for each control. The table above deliberately uses the selected role only, so it is paired with those controls.

Coverage-derived groups contain271 main-centre-outside and239 main/GT-disjoint states. The independently recomputed STOP-IoU-zero group contains430 states; the complementary main-centre-inside group contains1,481. Coverage-based memberships retain native geometric provenance.

On the271 centre-outside states, STOP mean IoU is0.000092482126, GT_CENTER_LOCAL is0.528493161035 and GT_CENTER_WIDE is0.570011270167. These are responses to a correct-location prior, not evidence isolating auxiliary-pixel causality.

## Same-token and GT-selected comparisons

| Paired comparison | Tokens | Mean IoU difference | Positive / negative / ties |
|---|---:|---:|---:|
| LOCAL_MAIN_minus_RANDOM_LOCAL | 193 | +0.399663380603 | 1222 / 143 / 387 |
| LOCAL_ALTERNATE_minus_RANDOM_LOCAL | 193 | +0.393105811562 | 1227 / 134 / 391 |
| GLOBAL_LOCAL_minus_RANDOM_LOCAL | 193 | +0.078313551528 | 503 / 212 / 1037 |
| GLOBAL_WIDE_minus_DENSE_SAME_SEARCH | 385 | -0.185825725616 | 374 / 1019 / 359 |

All four comparisons pair the same1,752 states and selected role. Equal token counts do not establish equal latency; the random baseline has one realized query per state.

GT-selected observed best-primary mean is0.542823526464; observed best among the five selected-role executable actions is0.550987690299, with mean gain0.112965761387 over selected STOP. The action maximum excludes all GT-centred/random/dense controls. Neither maximum is deployable selection or a trajectory upper bound.

## Checks and proof boundaries

### A_gt_provenance_and_causal_reference: WARN

prepare_data.py loads trainingset/<sequence>/init.txt; canonical LasHeR loader also uses init.txt. The copied manifest hash matches the sealed training plan, canonical train-name order matches all 882 entries, and only train entries were accessed for scoring. No target is generated from a prediction. Reference initialization uses frame0 GT; later PrimaryOnlyTracker.track accepts only image, derives state/history/template updates from predictions. Counterfactuals consume target only for IoU/labels; GT-centred controls are downstream offline queries and do not alter the reference tracker.

Limit: No original remote init.txt or image bytes were opened in this audit. Real-dataset origin is established by source trace and the bound manifest, not fresh independent dataset-file attestation. Whole manifest bytes were hashed; dev/test entries were not queried or used.

Evidence: `scripts/prepare_data.py:7-25`; `lib/train/dataset/lasher.py:61-84`; `scripts/diagnose_global_opportunity.py:30-57`; `scripts/diagnose_global_opportunity.py:84-120`; `scripts/diagnose_global_opportunity.py:122-151`; `dual_rgbt_global/training.py:102-125`.

### B_executed_boxes_and_metric: PASS

Source maps query proposals to original-image xywh, calls clip_box margin=10, then converts to float32. Every stored prediction is exactly representable as float32, has nonnegative x/y and width/height >=10. Explicit scalar float32 IoU against uncensored train GT matches every CSV value exactly; every Python delta subtraction is exact. IoU is intersection/union; model confidence normalization occurs only in predictor features, not in reported performance.

Limit: The CSV omits raw pre-execution proposals and image dimensions, so margin clipping is verified by source and stored-box invariants, not independently replayed from raw proposals.

Evidence: `dual_rgbt_evidence/readout.py:49-58`; `dual_rgbt_evidence/regions.py:35-49`; `lib/utils/box_ops.py:97-106`; `scripts/diagnose_global_opportunity.py:143-151`; `dual_rgbt_evidence/policy.py:52-61`.

### C_native_receipts_weights_source_and_hardware: WARN

Formal A input receipt records COMPLETE/exit0, 50000 samples/6250 updates, sanity=false. Its checkpoint hash matches launch, sanity, formal config and COMPLETE. Native task launch/first/terminal identity and commands match; GPU1 lies within the permitted physical0/1 set. All 31 local source bytes match the source map; source30 is an exact subset. All five native formal/sanity JSON artifact byte hashes match collector receipts and parsed payloads. The private formal CSV SHA matches both completion levels. Diagnostics record 265 unchanged state tensors, zero backward/optimizer; sanity records seven bit-equal strict probes.

Limit: Native status, remote checkpoint hashes, frozen-state assertions, physical GPU binding and collector completion are inherited parent-collected evidence. This audit did not SSH, inspect /proc, query a GPU, load weights or rerun NN. The supplied collector uses subprocess check=True for remote/scp and verifies raw bytes, but its parent tool receipt is not a newly independently observed receipt. Source31 covers exactly those files, not every dependency, binary or runtime package. Current clip_box/LasHeR loader source was read separately and is outside source31.

Evidence: `scripts/run_task.py:14-32`; `scripts/diagnose_global_opportunity.py:175-237`; `scripts/complete_pipeline.py:24-27`; `C:/Users/gb/.codex_tmp/dual_rgbt_observe_global_opportunity_formal_a_20261014.py:35-55`; `C:/Users/gb/.codex_tmp/dual_rgbt_observe_global_opportunity_formal_a_20261014.py:74-90`; `reports/global_opportunity_formal_a_completion.json:2-187`.

### D_planned_scope_matrix_and_dead_code: PASS

Actual scope is exactly the first20 entries of the identical all_train/evidence/global execution orders, all belonging to the canonical 882 train set. 8854 tracked frames, 1752 valid every-fifth-frame states and 24528 rows match the plan. Each state has 2 primary roles x5 executable actions, plus four controls only for the recorded selected role. There are 1501 selected RGB-role and 251 selected TIR-role states. The state matrix, own-role STOP baseline, token labels and matching source coverage fields are consistent. Metric functions are called on every recorded action and native summary is actually emitted.

Evidence: `reports/all_train_plan.json:1`; `reports/evidence_policy_training_plan.json:1`; `reports/global_evidence_training_plan.json:1`; `reports/global_evidence_opportunity_plan.json:1`; `scripts/diagnose_global_opportunity.py:106-164`; `scripts/diagnose_global_opportunity.py:229-238`; `order_chain.json`; `scope_addendum_result.json`.

### E_independent_arithmetic_and_report_match: PASS

Independent checker imports only standard library and cached NumPy. It does not import/call the author analyzer. Explicit scalar float32 arithmetic recomputes all24528 IoUs and all24528 Python deltas. All395 numeric fields agree exactly:72 native action fields,292 analysis group fields,20 pair fields,4 GT-selected conditional fields,3 scope counts and4 numerical-check fields. Selected-role all/outside/disjoint/zero-IoU groups and complementary centre-inside aggregates were independently regrouped from CSV records. Four193/385-token comparisons are paired by identical state and selected role.

Evidence: `checker_r2.py`; `checker_r2_numeric_field_comparisons.json`; `checker_r2_checks.json`; `checker_r2_result.json`.

### F_coverage_and_GT_control_semantics: WARN

Source computes coverage from executed query geometry and uncensored GT. This checker verifies range, repeated-source consistency and aggregate arithmetic only. Full historical query geometry is absent; centre-outside/disjoint memberships therefore retain native measurement provenance. The independently recomputed zero-primary-IoU grouping does not depend on coverage geometry. GT-centred controls supply a correct location prior to query selection and coordinate conversion. Their improvement is conditional on that prior and does not isolate auxiliary-pixel causal contribution, establish an optimal region, or bound a whole trajectory. Neither suppressed/blank/swapped auxiliary-pixel controls nor deployable GT-centred selection were run here.

Evidence: `scripts/diagnose_global_opportunity.py:60-65`; `scripts/diagnose_global_opportunity.py:90-91`; `scripts/diagnose_global_opportunity.py:119-151`; `dual_rgbt_evidence/readout.py:100-119`; `tools/analyze_global_opportunity.py:124-127`.

### G_scientific_claim_scope: WARN

One seed2027 and20 training sequences, using a cheap-scheduler forced-STOP reference and state-weighted IoU. These are not official toolkit PR/NPR/SR, whole882 deployment-policy/trajectory results, an independent RGB or TIR rollout, efficiency measurements, or a three-component effectiveness result. Observed best primary and best selected-role action use GT selection on the fixed visited states; the latter maximizes only the five recorded executable actions and excludes GT controls. One random query per state supplies a single realized random comparison. Stage A trains executor decode/box_head while global preview/quality/query_gain remain frozen; the requested future stage B is not evidenced by this result.

Evidence: `scripts/train_global_evidence_policy.py:117-151`; `scripts/train_global_evidence_policy.py:279-285`; `scripts/diagnose_global_opportunity.py:30-57`; `tools/analyze_global_opportunity.py:83-88`; `docs/GLOBAL_OPPORTUNITY_PROTOCOL.md:18-53`.

## Nonblocking documentation findings

- `docs/GLOBAL_OPPORTUNITY_PROTOCOL.md:18`: replace “原train顺序前20条” with the first20 entries of the already frozen882-sequence execution plans. Do not alter the plans or results.
- `docs/GLOBAL_OPPORTUNITY_PROTOCOL.md:3-9`: the dated launch-time status still says no complete result. Update it for completed-diagnostic publication, preserving the historical timestamp distinction.

No numerical or matrix defect requires rerunning NN. Future stageB and matched causal/deployment/efficiency evidence remain separate work; this result cannot pre-approve three effective innovations.

## Preserved audit history

The first bare-python command actually exited1 (`No pyvenv.cfg file`) before the checker started. The first cached-interpreter source read exited1 because of GBK stdout encoding. Both failures and their outputs remain. Later commands use the existing interpreter with explicit UTF8; nothing was installed.

The original checker actually exited1 with two extra assumptions equating the execution-plan order and canonical manifest order. Its619 other checks already passed. The order was then read directly from the plan files; R2 corrects only those checker assumptions and writes separate results. `checker.py`, `checker_result.json`, the original exit1 and `checker_r2_revision.json` remain. The imprecise protocol wording remains a WARN. No author result was edited and no failure was overwritten into PASS.

Saved sources, exact argv, raw stdout/stderr and process return codes are under `cmd_*.command.json`, `cmd_*.stdout.txt`, `cmd_*.stderr.txt`, `cmd_*.exit.json`; early PowerShell/interpreter receipts have separate named files. `original_seal.json` preserves the initial seal, `input_seal.json` records later first reads, and `postseal.json` checks the same bytes. Private raw CSV/manifest were only read in place; no GT coordinates or per-frame query data are in this report.

## Key original-byte SHA256 bindings

| Artifact | SHA256 |
|---|---|
| csv | `3521958631fe7d86029b820158d19be86ebf5c5cbed6c8701e41f2aeb7a6e62a` |
| manifest | `389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc` |
| plan | `a0accbdea3c6c8742103484e7a6c1db8bc5507647f1975f5f048477c79ef84e7` |
| training | `b5cf7d472bb1a09787d57a1a15f95cdd9989f0b102908189a74c975dbe7daafd` |
| completion | `be5ea78be9c4340f1c680bfb4578a1a24206c75767771cbebfd99fd288fc793e` |
| analysis | `ad9c86892ea819614afd4962007c1ac1db5298f22b0a8bc532af581d9b06e426` |
| analyzer | `881b3540c13a97852d2315ee8baf639e59c933169a6c716431550f1512eadb16` |

Checkpoint hash recorded consistently in the parent-collected native chain: `4836ade9b7afc417f90a995d90ef70a4681347fd000e3a4f0f1de313af9fe06e`. This audit did not load or freshly hash the remote checkpoint.
