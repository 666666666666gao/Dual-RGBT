# Candidate B result integrity audit

**Overall: WARN — 0 blocking findings for the narrowly scoped completed offline B fit.** The available artifact checks pass. Raw-box clipping and checkpoint tensor equivalence remain explicitly unverified; neither is converted into a PASS. Requested reviewer: Astra/max; actual model/effort routing: **UNATTESTED**. Fresh context, same-family, provisional.

The sealed formal run records native exit 0 after **6,747.707935 seconds**, from 2026-10-14 22:44:35.351304 to 2026-10-15 00:37:03.059239 Asia/Shanghai. Its original observer records the same wrapper/command at the running and completed observations. This review made **0 NN calls, 0 GPU queries and 0 SSH calls**.

## Independently checked result

| Check | Result |
|---|---|
| Contiguous optimizer-step rows | 6,250; correct epoch and four-family cycle |
| Sampled examples / query labels | 50,000 / 100,000, both primary roles |
| Train sequence exposure | All 882 in every epoch; 36–79 samples per sequence overall |
| Sequence/frame/role/family/main-geometry reconstruction | Exact for all 100,000 rows using seed 2027 and epoch/sample offsets |
| Candidate rank sampling | Exact RNG replay within each recorded valid pool count |
| Earlier-template ordering | 100,000 reconstructed earlier-frame checks pass |
| Saved executed-box main and query IoU | Exact float32 agreement with dataset GT on every row |
| Signed query-minus-main benefit | Exact float32 agreement on every row |
| Explicit finiteness checks | 300,000 IoU/gain scalars; 2,200,000 box/geometry values; 18,750 loss scalars |
| Float32 loss decomposition | Exact quality loss + gain loss for all 6,250 steps |
| Sources / before-after seal | 37/37 planned source hashes match; all 88 audited input files unchanged |

The eight role/family totals are **12,504 for each role in families 0 and 1**, and **12,496 for each role in families 2 and 3**. The family cycle continues across epoch boundaries. The labels cover 43,986 distinct sequence/frame pairs. All sample identities, frames, crop geometries and sampled ranks were independently reconstructed without importing the model or using the historical analyzer as the arithmetic oracle.

There are **28,075 positive, 29,634 negative and 42,291 zero gains**. Negative benefits were retained. Mean saved primary IoU is 0.277290411655, mean sampled-query IoU is 0.247797271261, and mean signed benefit is −0.029493140411. These describe the sampled training labels; they do not measure learned query/STOP selection. Last-step losses are 0.078193075955 total, 0.056052125990 quality and 0.022140949965 gain.

Evidence: `scripts/train_candidate_evidence_policy.py:249`, `:255`, `:268`, `:274`; `dual_rgbt_global/training.py:26`; `dual_rgbt_evidence/regions.py:43`; archived formal `steps.csv` and the hash-bound private labels.

## GT provenance and evaluation type

**Real dataset GT supervised labels on synthetic tracking states.** The exact manifest hash is `389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc`. Its train/dev sequence order matches the repository split files. The 882 train, 97 dev and 245 test name sets are disjoint; the manifest contains test names rather than test annotations. The trainer selects `manifest['train']`, and all 100,000 audited target lookups address that split.

`scripts/prepare_data.py:11` reads dataset `init.txt`; the inherited loader uses the same annotation file (`lib/train/dataset/lasher.py:68`). The source and sealed manifest establish this provenance; original external annotation files were not reread locally. Main/query IoU uses these GT boxes rather than model outputs, with no prediction-statistic denominator or GT clipping to the crop (`dual_rgbt_evidence/regions.py:43`). Controller entropy normalization is a feature, not a normalized evaluation metric (`dual_rgbt_global/candidate_policy.py:51`).

The current-frame GT also determines the jittered search state, while earlier GT boxes determine templates; history is zero (`dual_rgbt_global/training.py:33`; `scripts/train_candidate_evidence_policy.py:254`). Candidate proposal/head methods do not take a target argument, and the labels compare executed predictions with GT after execution (`scripts/train_candidate_evidence_policy.py:50`). This supports offline fitting with a GT crop prior. It does not establish causal rollout performance.

## Native completion, optimizer and freeze evidence

The local config and COMPLETE objects exactly match the original observer response. Three collected formal artifact hashes, the private label hash, two query transport receipts and four SCP receipts close exactly. The rendered observer program is byte-identical to the sealed helper instantiated with its embedded original launch and plan. The first diagnostic row and its 100-row loss means also match the final CSV. `scripts/run_task.py:28` stores the actual child return code rather than inferring completion from a counter.

The source calls AdamW once before each emitted step (`scripts/train_global_evidence_policy.py:40`; `scripts/train_candidate_evidence_policy.py:266`). At every epoch save it asserts optimizer-state step equality, 257 unchanged state tensors, exactly 8 changed quality/gain tensors and no gradients on frozen parameters (`scripts/train_candidate_evidence_policy.py:157`, `:291`). The continuous CSV and native exit0 support **6,250 real recorded AdamW updates and the native freeze assertions**. Local checkpoint tensors are absent, so no independent tensor comparison or gradient replay is claimed.

Nine of the 15 plan inputs were directly rehashed from local, supplied-private or archived bytes. The remaining six inputs, including three input checkpoints, and the protected CAD checkpoint are covered by the native producer's mandatory start/end hash assertions (`scripts/train_candidate_evidence_policy.py:194`, `:296`); they were not rehashed from local bytes in this review. The final checkpoint hash is `b84d00454717cae56048753e0fe54c1d6b92cfcd2820c01fa65296508488ccb3`, checked against remote bytes by the original completed observer. The private label CSV hash is `9a5680a4c61d3cd78b4ea5307c5a6e9f6dcbda48504dcfb2109f90184737da4d` and was independently rehashed here.

M0 admission closes to the exact native COMPLETE hash, four updates, recorded strict 12-field reload equality, K1 equality and a 10-frame causal execution/replay probe. The formal producer's strict reload branch is **sanity-only** (`scripts/train_candidate_evidence_policy.py:167`), and formal `causal_probe` is null (`:295`). This review does not claim a formal final-checkpoint reload or extend the M0 probe into a causal C result.

## Limits and semantic warnings

- **Full-run clipping is unverified.** There are 15,502 raw-to-executed differences among 200,000 recorded boxes. With no full-B image headers, the audit cannot independently recompute upper image-bound clipping or candidate-center membership. The machine report records clip error as `null`, not zero. The exact IoU checks apply to the saved executed boxes (`dual_rgbt_evidence/regions.py:35`; `lib/utils/box_ops.py:97`).
- **The historical B checker can hide a NaN scalar.** Its box checks are finite-aware, but `max(previous, abs(error))` at `tools/check_candidate_policy_sanity.py:69` can retain zero when the new error is NaN. The fresh audit explicitly checked every actual formal scalar finite before arithmetic; all passed. Later C-checker repairs are not credited retroactively, and no historical source/report was edited.
- **Scope remains training completion.** One seed and five offline epochs establish a completed fit. No learned query/STOP advantage, new causal C training result, new dev/test scores, complete-system latency, 78/72/62 target attainment, speed improvement or three-module causal efficacy follows from these losses or sampled-label statistics.

The private audit ledger preserves exact commands, stdout/stderr, native receipts, before/after seals and checker sources. One reviewer tool error is preserved: a result JSON was read before the CPU checker had finished. The same checker then completed exit0; no rerun or experiment-input change occurred. Only this Markdown report and its JSON companion were written in the repository.
