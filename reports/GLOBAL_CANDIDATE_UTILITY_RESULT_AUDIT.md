# Global candidate utility result-integrity audit

**RESULT_INTEGRITY_PASS_WITH_WARNINGS — 0 blocking issues.** Deterministic verification **PASS**; semantic verdict **WARN**, fresh independent-context Codex review, **same-family/provisional**. Requested `gpt-6-astra` / `max`; actual model, effort and routing **UNATTESTED**. Generated 2026-10-14T13:43:33.778605+00:00.

The artifacts support executed single-query opportunity on the fixed training states. They do not certify the proposed selector, closed-loop improvement, final PR78/NPR72/SR62, three causal mechanism benefits or actual system acceleration.

| Audit check | Verdict | Evidence and implication |
|---|---|---|
| A. GT provenance | PASS | `scripts/prepare_data.py:11–30` reads dataset `init.txt`; the sealed manifest matches train882/dev97 lists and is disjoint from test245 names. `scripts/diagnose_global_candidate_utility.py:67–130` forms candidates before consuming current GT coordinates. GT is neither generated from predictions nor passed into the advancing tracker. Initial GT and valid-frame eligibility are explicit. Remote original annotation/image bytes were not recaptured. |
| B. Metric normalization | PASS | `dual_rgbt_evidence/regions.py:43–49` uses ordinary image-coordinate IoU; `scripts/diagnose_global_opportunity.py:60–65` divides overlap by GT area. Gain is query IoU minus primary IoU. Prediction-normalized model features are not evaluation denominators. STOP flooring is a disclosed GT oracle. |
| C. Existing results and numbers | PASS | Actual native COMPLETE/exit0, raw metadata, private CSV, CPU outputs and 17 transport receipts bind by hash. All 31,384 formal rows and 36 M0 rows recompute with **zero** clip/IoU/main-IoU/coverage/gain error. All aggregate and magnitude claims reproduce (`docs/GLOBAL_CANDIDATE_UTILITY_RESULTS.md:3–34`). |
| D. Metric execution | PASS | Producer calls IoU/coverage and aggregate (`scripts/diagnose_global_candidate_utility.py:120–154,220–242`); actual CPU and choice stdout match their files. Independent NumPy recomputation confirms outputs. Official tracking metrics were not executed by this diagnosis. |
| E. Scope | WARN | One seed2027 and checkpoint pair; 20 training sequences, 8,854 reference tracking steps, 1,752 valid stride5 states, including239 outside states. State-weighted, correlated training observations and GT oracles establish no held-out selector effectiveness, closed-loop bound or speed. |
| F. Evaluation type | PASS | **real_gt — train-only fixed-state executed counterfactual/oracle diagnosis**. Model-produced trajectories do not make the labels model-generated. Legacy GT-centred controls are separate conditions. |

The reviewer independently reconstructed all **1,752 candidate lists, 13,940 indices/points and 14,016 prefix/family geometry records** from the preserved score maps, image headers, predecessor content rows and dataset manifest. Candidate counts are2→2,3→7,4→1,5→4,6→4,7→5,8→1729; all23 short lists genuinely exhaust valid cells. Exact query action order, geometry and tokens reproduce. All **3,504 legacy main-search local queries** retain identical geometry, executed boxes and IoU. Main-outside membership now recomputes directly from saved main geometry, closing the older geometry-only audit's inherited-label limitation.

The complete diagnostic executes **17,444 local128/193-token queries and13,940 wide256/385-token queries**, totaling **8,733,592 auxiliary tokens**. It has8,854 trajectory primary calls and1,752 additional diagnostic primary calls. These operation counts are not inference latency. Within each query family Top1 and oracle-selected query share the same resolution/token budget; the diagnostic physically executes the whole candidate pool. Mixed `ALL_ACTIONS` uses both193/385-token actions.

The current driver is byte-bound to R2. Undoing only the two candidate argument complements at `scripts/diagnose_global_candidate_utility.py:58,108` reconstructs the preserved R1 driver exactly; its plan changes only the driver digest and R2 path. Candidate generation consumes the auxiliary stream; the query reader receives the primary role and complements internally (`dual_rgbt_evidence/readout.py:100–118`). R1 FAIL remains preserved, not overwritten by later success.

Actual M0 and formal native tasks reached COMPLETE/exit0 under that source. M0 records10 tracking frames,2 states,36 queries plus2 reload query probes. The first36 formal rows equal M0 byte-for-byte at the parsed row level. The eight strict-reload fields are `query_box`, `query_pixel_box`, `query_score`, `query_size`, `query_offset`, `candidate_scores`, `candidate_indices`, `candidate_points`. Both separately loaded models have265 frozen state entries; exactly six preview entries differ and259 shared entries match. Zero optimizer/backward and absent-gradient assertions apply. The source checks precede and follow the executed diagnosis (`scripts/diagnose_global_candidate_utility.py:164–240`). These neural-state facts are supported by archived runtime execution, not re-created by CPU source checks.

Formal command/config uses M0 COMPLETE SHA256 `234d6fc6da80c841d9d8078ad5abdc807272e834f685c1dec9d087d2dc097a0b` before model construction. The formal metadata gate checks length8, while the actual sealed producer, collector and saved SANITY establish exact names. It does not itself rehash the M0 CSV; the actual collector verified CSV/config/SANITY/COMPLETE hashes. Seven of10 plan inputs were rehashed locally. The three checkpoint inputs and protected CAD checkpoint remain bound by archived predecessor/native evidence and required runtime assertions; no weight reload was performed here.

The original CPU precision failure is real and reproducible. Applying the old NumPy-scalar clipping to the same M0 rows produces maximum error **3.0517578125e-05**, exceeding the unchanged1e-5 criterion. The one-line conversion to Python floats matches `execute_boxes.tolist()` → `clip_box` → float32 (`dual_rgbt_evidence/regions.py:35–40`, `lib/utils/box_ops.py:97–106`) and produces zero error for both runs. The sanity checker is otherwise byte-identical; no tolerance was relaxed, producer changed or M0 rerun. Original exit1 and corrected exit0 receipts remain intact.

The main supported numerical findings are:

| Outside-state query | Mean executed IoU | Positive gain | IoU≥0.5 |
|---|---:|---:|---:|
| Main local |0.0006172943|1|0|
| Main alternate local |0|0|0|
| Global local Top1 |0.0337360558|21|8|
| Global local best-of8 |0.2301839757|125|58|
| Global wide Top1 |0.0408293935|20|10|
| Global wide best-of8 |0.3377776673|129|100|

All239 outside states have zero primary IoU. Wide best8 gains reach≥0.01/0.05/0.10 in129/125/123 states; local counts are123/120/115. These are substantial measured oracle opportunities, with no proof that available cheap features can learn the choices (`docs/GLOBAL_CANDIDATE_UTILITY_RESULTS.md:11–24`).

Across all1,752 states, primary mean IoU is0.5179512621. Forced best8 global-local/wide signed gains are **−0.0643111569/−0.0609063972**; their GT STOP-inclusive gains are+0.1008473943/+0.1021804621. Mixed all-action best IoU is0.6644843554, signed gain+0.1465330933 and STOP-inclusive gain+0.1614323696. Maximum-GT-coverage wide choice outside yields0.2347693618 versus executed-IoU oracle0.3377776673. This supports the stated distinction between geometry and actual localization value, without establishing a cause or a deployment policy (`docs/GLOBAL_CANDIDATE_UTILITY_RESULTS.md:28–34`). Lower row order resolves ties; `coverage_choice_equals_best` compares action identities, not equal-valued IoUs.

The existing gain path still hardcodes four slots (`dual_rgbt_evidence/policy.py:70–88`). Extending it is future work, not an audited implementation. A new selector needs its own source/M0 evidence, refreshed labels and closed-loop training, dev calibration, full test results and paired full-system timing. No present result certifies those future outcomes.

Historical pending statements in the20:50 HANDOFF/protocol and earlier R2 review were preserved as historical snapshots. `HANDOFF.md:22` uses20:38:20.832287 BJT, the M0 collection time; native M0 completed at20:35:37.490006 BJT. Formal native duration is2027.236982seconds (about33.79minutes), ending21:21:03.994471 BJT. This is total diagnostic task wall time, not online latency. Formal observations were one scheduled RUNNING capture and one COMPLETE capture about241.36seconds later, of the same wrapper.

Verification completed **524,392 deterministic checks**:520,013 independent NumPy checks,3,789 stdlib provenance/claim checks and590 first/last preservation seals. The NumPy run used the exact requested interpreter and NumPy1.26.4, actual exit0/empty stderr, with no torch import. The stdlib provenance run also exited0 with empty stderr. Both scripts and all actual stdout/stderr/exit receipts are private at `D:/Program Files/UserCache/gb/codex_tmp/dual_rgbt_candidate_utility_independent_audit_20261014_7b2e`.

All34 gated sources and the plan/results/private inputs are unchanged under the initial/final seals. The raw initial preservation inventory contains589 files, including two reviewer-owned helpers inadvertently included through a directory alias. The final-seal attempt correctly detected the intentionally edited reader; only these two identified review helpers were excluded, leaving587 unchanged experiment-preservation entries plus3 supplemental files (two split lists and predecessor ranking). The original initial seal remains intact. This broad inventory is larger than the directly interpreted evidence. Three reviewer-only failures were retained: the initial GBK read, a provenance glob that included a CSV SCP receipt, and the self-inventory seal assertion. Corrections affected private review helpers only. No NN/GPU/SSH/process query, training, source change or prior-document rewrite occurred.

- Plan SHA256: `36dfb23499c693deb5d62b31c2a8e257cbf863162443ca8b5e4f180a89e4d689`.
- Driver SHA256: `8a067918dc77994f486215e9439133175581c97a3339e8efc76179d58431c6fb`.
- Formal CSV SHA256: `dd85510eb7e05b2e9c2a7eff2b530f592a097057b0a5323697493ba9107dac90`.
- Initial seal SHA256: `01709764df5a9ab999d605af6a695c5c086d1764801b73cb995748b8b905e6e1`.
- Final seal SHA256: `23290fbc45d97d508756a91c0ea663f24632c9a0afdb7c0441631d9e96e7167a`.

Only this safe aggregate report, its JSON companion and the private audit directory were written. Original GT/image bytes and checkpoint tensors were not independently reloaded. Deterministic checks can be accepted; the semantic assessment remains same-family/provisional with UNATTESTED routing.
