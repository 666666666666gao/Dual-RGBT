# Global descriptor-scale result audit

**Verdict: WARN; frozen-result integrity PASS; 0 blocking integrity findings.** Fresh same-family/provisional review, 2026-10-14. Requested `gpt-6-astra` / `max`; actual model and effort **UNATTESTED**. No cross-family or blind acceptance.

The saved comparison is complete and internally consistent: 20 train sequences, 1,752 frozen states, 3,504 paired T64/T32 rows, including 239 native-main-outside states. **323/323 independent local checks passed.** All 73 sealed audit inputs and all 32 plan-bound sources remained unchanged. T64 exactly equals every prior-after field for all 1,752 rows. The float32 ROI recomputation gives maximum absolute coverage error **0** across 7,008 ROI evaluations. A separate float64 rectangle formula differs by at most 1.246059263049215e-6 due to numerical precision.

| Scope | Descriptor | Local centre | Wide centre | Mean local GT coverage | Mean wide GT coverage | Mean nearest-cell rank |
|---|---:|---:|---:|---:|---:|---:|
| All 1,752 | T64 | 348 | 720 | 0.18783659 | 0.41281550 | 144.42523 |
| All 1,752 | T32 | 324 | 711 | 0.17843404 | 0.40250532 | 143.10674 |
| Native-main-outside 239 | T64 | 25 | 55 | 0.10059388 | 0.22847299 | 100.58577 |
| Native-main-outside 239 | T32 | 34 | 56 | 0.13868382 | 0.24161485 | 78.26360 |

T32 changes 1,346 overall peaks and 132 in the outside subset. In that subset, local centre inclusion gains 17 states and loses 8; wide inclusion gains 21 and loses 20. Overall, local gains 146 and loses 170, while wide gains 213 and loses 222. The subgroup improvement is supported; overall improvement is unsupported. No significance or generalization inference is made.

**A. GT provenance — PASS (`real_gt`).** The manifest declares dataset `init.txt` annotations; source loads those files directly (`scripts/prepare_data.py:17`) and the diagnostic uses the train split (`scripts/diagnose_global_descriptor_scale.py:50`). Current and first-template GT boxes are finite, positive and in `valid_frames`; sequence and frame identities, official local train split membership, split disjointness and all paired roles were checked. Current GT is used for rank/margin and ROI evaluation, without entering cosine scores or argmax (`scripts/diagnose_global_descriptor_scale.py:71`, `:76`, `:77`, `:83`, `:89`). Initial GT initializes the shared frame0 template. Raw remote annotation/image bytes were not reopened.

**B. Score normalization — PASS.** ROI intersection is divided by full GT area, never the model's own maximum (`scripts/diagnose_global_descriptor_scale.py:89`). Cosine similarity is the proposal score; margin is target-nearest score minus peak score, and rank counts strictly higher scores (`:76`, `:83`). The audit verified finite nonpositive margins, valid rank ranges and rank-zero/margin-zero consistency. These checks validate stored numbers and definitions; they do not independently regenerate CNN score maps.

**C. Result existence and provenance — PASS within the saved evidence.** Source32, plan, prior-checkpoint identities, launch/config/completion bindings, private-result SHA, all copied artifacts and original CPU-analysis inputs match. Prior preview fit records 6,250 updates and 882 seen sequences. Formal saved SSH transport reports exit0 with actual stdout/stderr hashes; parsed stdout exactly equals `reports/global_descriptor_scale_formal_completion.json:2`. Copied config/COMPLETE hashes match its `artifact_sha256` (`:133`) and comparisons match COMPLETE. CPU transport records exit0, empty stderr and stdout equal to the existing analysis. Sanity config/SANITY/COMPLETE match their native-status report, and its completion precedes formal launch; the first four formal rows serialize to the sanity private-result SHA.

Native job durations are 40.90194 s sanity and 177.76081 s formal; original CPU analysis is 2.14730 s. These are whole-job elapsed times, not inference latency. The separate sanity SSH transport was outside the supplied evidence. Per-copy SCP exit receipts are absent from the listed observer directory; artifact integrity is established by hashes, not inferred from empty SCP streams.

**D. Metric execution — PASS.** `diagnose` is called from main (`scripts/diagnose_global_descriptor_scale.py:149`), `predictions` is called for sanity reload (`:155`, `:158`), and groups flow into COMPLETE (`:165`). Saved actual CPU stdout proves the analyzer ran. This audit independently reads all raw rows and redoes geometry and aggregation; the original analyzer is not imported or used as the checker. All 31,536 old CSV rows were read; the 1,752 GLOBAL_WIDE/TRUE states each occur exactly once per resolution, with no missing/duplicate state IDs. Existing aggregate groups agree exactly with independent recomputation and prior-after T64 groups.

**E. Scope — WARN.** One frozen seed2027 checkpoint is compared on 20 already-seen training sequences after a train882 preview fit. This supports fixed-state proposal geometry only (`reports/global_descriptor_scale_formal_analysis.json:76`). Shared current pixels/grid/ROI sides and shared CNN weights are preserved. T64 central4x4 and T32 central2x2 descriptors retain the same frame0 template (`scripts/diagnose_global_descriptor_scale.py:22`, `:63`). Resizing also changes receptive-field/context effects; scale alone, template ageing and distractor identity are not isolated. No new query execution, controller, trajectory, held-out test, PR/NPR/SR, latency, stability, novelty or final-method target claim follows.

**F. Evaluation type — real_gt, training diagnostic.** The GT source is dataset annotation, while main-outside membership and numerical CNN rank/margin remain native-derived. Grid membership, ROI geometry, counts and T64 identity are independently checked. Score maps were not saved or reconstructed by this audit.

The native code asserts all copied state tensors are unchanged and all parameter gradients absent (`scripts/diagnose_global_descriptor_scale.py:151`). The reported count265 is a literal field corroborated by the prior259 frozen plus6 preview tensors; this auditor did not open or recount a checkpoint. Sanity reload checks four score/point fields on the final sanity probe (`:153`), not every formal state.

The supplementary audited protocol is the exact `docs/GLOBAL_DESCRIPTOR_SCALE_PROTOCOL.md` snapshot SHA256 `79e7109125381e355ba96dfb2bc0ee2e885df79e856898b1bf2dc8b1f946a7a6`. Its SOURCE_PENDING/no-GPU wording is historical; actual completion is established by the later bound artifacts. Source-review runtime-admission text is likewise pre-execution. No claim is made about later edits to these documents or unaudited current HANDOFF/tracker pages. The JSON enumerates every audited mutable document and its private snapshot.

No NN, torch/project import, SSH, process polling, Git mutation, installation, environment change or nested agent was performed. The independent checker exited0 on its first execution with empty stderr. Two preparatory command errors are retained as explicit native-tool transcriptions: a generic Python alias lacked pyvenv.cfg, and a Windows wildcard rg discovery failed. The already-recorded CPU interpreter and `rg --files` resolved those tool-selection issues without changing inputs. These transcriptions are not fabricated raw-stream receipts.

Key SHA256 bindings:

- Plan: `146c29016fbb9a513c35c8ed738433106e6788c15294ee7023d04b1c1d8de52b`
- Diagnostic: `abdf76b9b18f80ac352cc3e7e1cfdcf612f4fc8a9653071270a025ff8c04f4c8`
- Analyzer: `3938c3ef03a3776b62ce40971ef8902ac6ad478c0771d2d3f06a4f918215694e`
- Private comparisons: `fc1dcda9968e30063faa33e97759d939877cd28723aba7d312d155ff96993d65`
- Prior-after baseline: `2de0a53c514ff19f4fe45172b00f6f61bce1e9b4b841502ed86fb8ff72273a6c`

Public report contains aggregate results only. Private checker, exact input snapshots, first/closing seals, actual checker stdout/stderr/exit receipt and trace are under `C:/Users/gb/.codex_tmp/dual_rgbt_global_descriptor_scale_result_audit_20261014/`. Frozen source32 and plan were not edited.
