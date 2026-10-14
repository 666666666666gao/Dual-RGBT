# Global content diagnostic: independent result audit

**Substantive verdict: WARN. Deterministic result checks: PASS. Blocking issues: 0.**

Requested reviewer: `gpt-6-astra` / `max`; actual routing **UNATTESTED**. Fresh same-family review; acceptance **provisional**. The request included executor summaries, so this was not a blinded path-only prompt. Primary artifacts were read and an independent checker was written; supplied numbers were treated as assertions to verify.

The complete frozen train 20 diagnostic has 8,854 tracked frames, 1,752 labelled states and 31,536 comparisons. All 216 result checks, 35 provenance/array checks and 8 native-binding checks passed in actual local processes. IoU, query GT coverage, query-centre membership, GT-centred geometry and every native/analyzer aggregate matched exactly; maximum numeric errors were **0**. Python 3.12.12 and NumPy 1.26.4 were used. No PyTorch/project import, NN, GPU, optimizer, backward or SSH execution occurred.

## Findings

- **GT provenance — PASS:** the frozen dataset manifest uses `init.txt`, matching the preparation source and CADTrack loader. Repository train 882/dev 97 split lists match; the selected 20 sequences are exactly the frozen 882-sequence execution prefix, distinct from manifest order. Labels are dataset annotations, with uncensored original-image IoU. Original annotation files were not recaptured. Evidence: `scripts/prepare_data.py:11`, `scripts/prepare_data.py:17`, `lib/train/dataset/lasher.py:70`, `dual_rgbt_evidence/regions.py:43`.
- **Source and read-only admission — PASS:** 32 sealed sources match the plan and preserve the 30-file training and 31-file opportunity seals. Local helper snapshots parse and remain unchanged. Full B precedes actual M0 and formal execution. M0 records 10 tracks/2 states/36 comparisons, 265 unchanged state tensors, seven strict-reload fields and zero backward/updates. Formal records the same frozen checkpoint, 265 unchanged tensors and zero backward/updates. These tensor/remote facts derive from completed native assertions, not a local neural replay. Evidence: `scripts/diagnose_global_content.py:142`, `scripts/diagnose_global_content.py:181`, `reports/global_content_sanity_b_completion.json`.
- **Controls and dataflow — PASS:** every state has one selected heavy primary role, six regions and three contents. Coordinates, token count and native main scalar are identical across content variants. The source keeps templates and primary features fixed, asserts factual STOP equality, and modifies only auxiliary search channels. Four synthetic NumPy cases verify both modalities at 128/256 pixels; they are array semantics tests, not PyTorch or image tests. Current GT affects current IoU/coverage and oracle regions only after initialization; counterfactual output never updates the future tracker. Evidence: `scripts/diagnose_global_content.py:23`, `scripts/diagnose_global_content.py:75`, `scripts/diagnose_global_content.py:95`, `dual_rgbt_evidence/readout.py:100`.
- **Metrics and existence — PASS:** all scheduled valid stride-five states are present exactly once with all 18 comparisons; no cherry-picked content subset or self-normalization appears. Means weight states equally. Raw native config/COMPLETE hashes, copied original tool stdout and private comparison bytes agree. Evidence: `scripts/diagnose_global_content.py:36`, `reports/global_content_formal_b_completion.json`, `reports/global_content_formal_b_analysis.json`.

## Safe aggregate results

| Region | TRUE IoU | ZERO_AUX IoU | SHIFT_AUX IoU | TRUE − ZERO |
|---|---:|---:|---:|---:|
| LOCAL_MAIN | 0.527540361287 | 0.464625591258 | 0.435487417751 | +0.062914770030 |
| LOCAL_ALTERNATE | 0.521659486305 | 0.452318036834 | 0.412245716839 | +0.069341449471 |
| GLOBAL_LOCAL | 0.106520094533 | 0.091417218869 | 0.066516228390 | +0.015102875664 |
| GLOBAL_WIDE | 0.192226896777 | 0.195201769992 | 0.091890006834 | -0.002974873215 |
| GT_CENTER_LOCAL | 0.644672370340 | 0.494620602053 | 0.470554546680 | +0.150051768287 |
| GT_CENTER_WIDE | 0.497800534364 | 0.432380008356 | 0.371911485169 | +0.065420526008 |

The native main STOP mean is **0.517951262113**. LOCAL_MAIN TRUE minus STOP is **+0.009589099174** on this visited-state distribution. This is a one-step diagnostic difference, not a trajectory performance gain. GLOBAL_WIDE TRUE is slightly below ZERO_AUX on the full population, so the results do not support a universal content advantage.

Of the **239 native-labelled main-centre-outside states**, the learned GLOBAL_LOCAL region includes the target centre in **11** and GLOBAL_WIDE in **49**. Those query memberships and coverage fractions were independently reconstructed from stored query geometry and dataset GT. The subgroup membership itself remains native-derived.

## Nonblocking limits and warnings

1. **Native provenance:** the parent supplied the original final tool result, actual exit 0/chunk `4e1e77`, from persistent tool-result storage. Its stdout exactly equals the public completion summary. Separate SSH stdout/stderr receipts were not retained. The original parent CPU analysis exit 0 is transcript-only; it is not presented here as a persisted independent receipt. This reviewer performed its own recorded computation.
2. **Reconstruction boundary:** main executed boxes and main search geometries were not saved, so main IoUs and main-outside labels cannot be independently rebuilt. Primary pixel/feature equality and the factual primary match rely on source assertions and the native run. Learned proposal coordinates were checked for identity and GT coverage, not regenerated from model features. The analyzer boolean about primary checks must be read with this boundary.
3. **Historical protocol status:** `docs/GLOBAL_CONTENT_PROTOCOL.md:3` still says SOURCE_PENDING / NOT_RUN. Completion artifacts supersede that historical status. No frozen protocol or source was edited.
4. **Inference scope:** GT-centred controls supply a correct-location prior. TRUE−ZERO isolates only the current auxiliary search-pixel intervention while auxiliary templates, both cheap previews, coordinates and learned priors remain. SHIFT is cyclic displacement, not independent distractor identity or natural corruption. Within-region comparisons have equal budgets; local 193 versus wide 385 is not an equal-budget comparison. One seed and one training reference distribution establish no stability, generalization, official PR/NPR/SR, latency, trajectory bound, A/B paired gain or completion of all three proposed innovations.

## Actual execution and input seal

| Local audit step | Actual exit |
|---|---:|
| snapshot_v1 | 0 |
| independent_results_v1 | 0 |
| provenance_semantics_v1 | 0 |
| native_binding_v1 | 0 |
| final_seal_v1 | 0 |

All 69 reviewed input files are byte-identical between their recorded first and final hashes; no changed input was found. Exact first/end hashes, stdout/stderr hashes, checker versions and safe aggregates are in the companion JSON. Private snapshots, scripts, receipts and the full request are retained privately. No computational check failed. Two discovery searches referenced absent paths; their observed errors and exits were preserved, and the correct paths were subsequently used.

| Bound artifact | First SHA256 = final SHA256 |
|---|---|
| Private comparisons | `ad261bfca634b205df559ed6d1b680c6c618268479b923f9f751a22cf44e3ffc` |
| Private dataset manifest | `389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc` |
| Frozen diagnostic plan | `7797bae9b9196467d4a56194927a9a1b01477a1d002df266dce2bb3b121afd75` |
| Original native config | `2090497304c408044fbaf6e407c84535fdef4753ffdf75e7ef3507ec6736e8b1` |
| Original native COMPLETE | `dffc3706c2fca704f1b4898a0821869782ac1b0ef352f536ee62fa583b5d74f8` |
| Parent analyzer | `f01a047ab9d7ccf48f44d74f76e4d1582d3147e1036a85f24b7d593e304534b5` |
| Original final native tool result | `d6235d3bbaa391d77f45bb8f4a63e6006b8e54ea9b7f911e86bdda95da258efd` |

This audit covers result integrity only. HANDOFF and subsequently written human-facing result documents were not reviewed. No raw GT, coordinates, row records, private paths or input snapshots are included in the public report.
