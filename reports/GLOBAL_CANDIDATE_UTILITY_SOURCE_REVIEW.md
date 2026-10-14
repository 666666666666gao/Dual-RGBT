# Global candidate utility source review

**SOURCE_FAIL — 1 blocking issue.** Fresh independent-context Codex review; same-family/provisional. Requested `gpt-6-astra` / `max`; actual model, effort and routing are **UNATTESTED**. This driver has not run real M0 or formal NN execution.

**R1 — wrong modality at the candidate-generation calls.** `scripts/diagnose_global_candidate_utility.py:58,108` passes the reference primary modality to `candidate_predictions`. The sealed predecessor uses the auxiliary modality, `1 - primary` (`scripts/diagnose_global_candidates.py:35–41,60,73`). Its `predictions` slices both initial template and global pixels using that argument. The new driver's own line96 asserts the saved auxiliary role is opposite, and all1752 archived rows confirm that contract.

Consequently, the new driver does not reproduce the intended candidate mechanism. Its line110 exact-point assertion will reject whenever the two roles produce different peaks; a coincidental point match does not correct the role. No neural execution was performed to claim a particular failing frame. Change only the two candidate-generation role arguments to their complements; keep `model.query`'s primary-role arguments unchanged. Then reseal the driver/plan and seek fresh SOURCE admission before M0.

| Check | Finding |
|---|---|
| Source and predecessor evidence | All34 files /234665bytes read and AST parsed/compiled without imports; source32 predecessor remains exact. Three raw config/COMPLETE pairs match archived native exit0 reports and hashes. Remote weights were not loaded; their identities are bound by archived evidence and driver checks. |
| Reference trajectory | Old fullB `PrimaryOnlyTracker` alone advances state/history/modality/templates. The diagnostic uses pre-track clones and asserts reproduced primary output/IoU/old scope. Slot0 remains the initial factor2/128px template; `Preview.forward` resizes to64 and uses central2:6 descriptor. Counterfactual outputs never update the tracker. |
| Frozen model compatibility | Two separately loaded/eval/frozen models; changed-tensor set must be exactly the six fitted preview CNN entries. The remaining259 tensors match before shared reference primary features are reused. The driver checks all model-state tensors unchanged and gradients absent after execution; no optimizer/backward call exists. Runtime verification is still pending. |
| GT and executed boxes | Proposals/regions are formed before current-frame GT label use. Fitted query output is mapped to original-image coordinates, clipped with actual `clip_box(margin=10)`, converted tofloat32 and then compared with uncensored trainingGT. Geometric coverage is a separate GT-area fraction. |
| Genuine candidate exhaustion | Independent stable NumPy reconstruction exactly matches all1752 lists and13940 candidates/points;23 lists truly exhaust valid cells. Histogram:2→2,3→7,4→1,5→4,6→4,7→5,8→1729. No candidate padding. |
| Aggregation | Numerical cases cover negative/zero/positive gains, coverage/IoU disagreement, exact ties and short lists across24scope-K-region groups. `ALL_ACTIONS` includes both main requests(rank−1) and both global local/wide prefixes. Means are state-weighted. STOP-inclusive gain is floored at0; best-query signed gain retains losses. |
| Formal admission | Driver-internal gate precedes model construction: source-review/plan/source/input/protected hashes, same-plan realM0 config, native COMPLETE/exit0/GPU1, exact supplied COMPLETE digest,10tracks/2states/265+265frozen/zero updates/backward, and SANITY PASS/bit-equal/eight fields.32 isolated corruption cases reject correctly. These tests use synthetic metadata and are not M0. |
| Scope and privacy | Raw state/frame/coordinate/box records remain private in the intended runCSV; public output is aggregate metadata plus hashes. Fixed-state query/coverage oracles establish no deployable selector, closed-loop bound, officialPR/NPR/SR, generalization, latency or ultimate-goal completion. |

Independently derived formal budget: **8854 tracking frames,1752 labelled states,31384 auxiliary queries** =17444 local128px/193-token executions +13940 wide256px/385-token executions, totaling **8733592 auxiliary tokens**. The normal trajectory also makes8854 primary calls; the diagnostic re-executes1752 additional primary calls. Sanity is10tracking frames/2states/36auxiliary queries plus2additional strict-reload query probes. These are counts, not measured throughput or end-to-end latency.

The all-state group has1752 states; the earlier native main-outside group has239. Those239 labels were checked against the sealed CSV and candidates, not independently reconstructed from unavailable old main geometry. Tie selection follows first row order: main requests precede ranked local/wide requests, and lower rank precedes later rank. `coverage_choice_equals_best` counts identical selected actions, so equal-IoU alternative actions do not count as equal choices.

The M0 probe code names five query fields (`box,pixel_box,score,size,offset`) and three candidate fields (`scores,indices,points`). The formal metadata gate checks list length8 rather than exact names; the sealed probe code defines them. It does not rehash the raw sanityCSV itself. Real saved-output/true-native-exit validation remains part of actual M0 collection, which has not happened. All runtime freeze/reload assertions remain unexecuted for this driver.

Pure CPU checker: **8978 checks, actual exit0, empty stderr**. Native receipt started `2026-10-14T12:14:59.858983+00:00` and finished `2026-10-14T12:15:03.076445+00:00`. Exact interpreter: `C:/Users/gb/AppData/Local/uv/cache/archive-v0/7oLWbeZ6v0oUodecKf4d1/Scripts/python.exe`. Source AST extraction executes only the aggregate function, original pure-Python `clip_box`, and the pre-model gate with isolated metadata; no torch/project imports or neural calls occurred.

Two reviewer errors are preserved: an initial default-GBK Markdown read failed after saving the before seal; the first checker attempt incorrectly expected `manifest['test']` instead of the actual `test_sequences` name list. The first checker script/stdout/stderr/true exit1 are archived. Correcting the checker schema produced the successful receipt; no source/plan input was changed or failed result hidden.

All **62 audited inputs** remained unchanged:52 original entries plus10 instruction/provenance dependencies, including source34 and the exact plan. Initial/final manifests, checkers, receipts, errors and numerical details are private under `C:/Users/gb/.codex_tmp/dual_rgbt_global_candidate_utility_source_review_20261014/`. Only this report, its JSON companion and that private directory were written; no Git, SSH, GPU, process/progress query, install, source/plan edit or nested agent was used.

Exact gate identities:

- Plan SHA256: `690a797df457b2880a09a4e1e847961e2992772cc2c57f93a793bcdabf8147ff`.
- Driver SHA256: `83bd405011f5902a7cf55cc60489b60fdd9b9ac175daf90f169747e8a8cf49fb`.
- Before/final52-input manifest SHA256: `ba3e65d1950c200954d521c8ac0199fe350c3ad4c97c02dcc383559a63b859bd`.
- Additional before/final10-input manifest SHA256: `e79b3c205f0a237f649c9432924823c7e38c0bd280f2b06a26d921a33dd41200`.
- Successful checker SHA256: `204f370750780298fa3f2e9cbd9ede4acf505d5924ae834f7195b1d48b07fbf4`.
- Successful stdout SHA256: `ab672a30796e8b6cc63fd6b3951bcc809c4fa62b930d56acfd526c8183bf436d`; stderr SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

**No SOURCE_PASS is issued.** Fix R1 and obtain fresh source admission; realM0/formal/results acceptance remains pending.
