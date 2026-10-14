# Global preview ranking: independent result audit

**Substantive verdict: WARN. Deterministic checks: PASS. Blocking issues: 0.**

Requested reviewer: `gpt-6-astra` / `max`; actual routing **UNATTESTED**. Fresh same-family review with **provisional** acceptance. The request included expected counts and interpretive cautions, so this was not a blinded path-only prompt. Assertions were checked against primary artifacts; executor explanations and earlier review verdicts were not used as proof.

The original full fit completed with **50,000 samples, 6,250 updates, five epochs and seed 2027**. All 6,250 logged losses are finite and steps are consecutive, with 1,250 updates per epoch. Independent NumPy replay of sample selection reaches all **882 training sequences in every epoch**; the overall exposure range is 36–79 samples per sequence. This replay verifies sample identities and annotation validity, not neural execution.

The native task, original terminal tool result (exit 0, chunk `76e567`), per-query stdout/stderr receipts, copied config/COMPLETE/steps and public completion/last-node summaries agree. The original CPU analyzer has a retained exit-0 receipt and matching stdout/input hashes. Genuine two-update sanity precedes formal execution; both start from the same completed, nonsanity full-B checkpoint. Formal initialization differs from the sanity output. See `scripts/train_global_preview_rank.py:137`, `:147` and the sanity/formal completion artifacts.

## Integrity checks

| Check | Result | Evidence and boundary |
|---|---|---|
| A. GT provenance | PASS | The frozen `init.txt` dataset manifest matches train882/dev97 split lists; training uses only `train`. Targets are dataset boxes. Original annotation files were not recaptured. `scripts/prepare_data.py:11`, `:17`; `lib/train/dataset/lasher.py:70`; `dual_rgbt_global/training.py:18`. |
| B. Score normalization | PASS | GT coverage is intersection area / GT area; means weight states equally. Gaussian target normalization and log-softmax belong to the training objective. Reported coverage is not normalized by prediction statistics. Raw rank/margin remain relative score diagnostics. `scripts/train_global_preview_rank.py:40`, `:91`, `:99`. |
| C. Existence and bindings | PASS | Original exit-0 artifacts, 31 source hashes, plan, fixed diagnostic, manifest, before/after hashes and analyzer inputs agree. Loss CSV and persisted outputs are real files. |
| D. Metric execution | PASS | Training loss and both inspections are called by main and their outputs exist; the analyzer has an actual recorded invocation. `scripts/train_global_preview_rank.py:162`, `:177`, `:205`; `tools/analyze_global_preview_rank.py:43`, `:88`. |
| E. Scope | WARN | One seed and 1,752 fixed old visited states from 20 reused training sequences. Effects are mixed, including worse full-population wide coverage. |
| F. Evaluation type | real_gt | Training diagnostic against dataset GT; separate nearest-cell oracle geometry is explicitly an oracle, not a deployed model result. |

All **1,752 scheduled states occur exactly once before and after**: sequence/frame/state identities, original auxiliary modality, native subgroup labels, local/wide side budgets and current-frame header grid agree. The original before proposal is reproduced with maximum coordinate error **0**. Independent scalar float32 centre membership and GT coverage reproduce both raw result files and original local/wide query coverage with maximum error **0**. All native/analyzer aggregates match. A separate higher-precision interval formula differs by at most **1.185e-6**, consistent with endpoint rounding. See `scripts/train_global_preview_rank.py:68`, `:84`; `tools/analyze_global_preview_rank.py:47`.

The previous nearest-cell oracle ROI aggregates also reproduce exactly; its other geometry aggregates differ by at most **1.694e-7**. Those oracle results establish geometric reachability under a correct-location prior, not achieved recovery.

## Recomputed training diagnostics

| Population | ROI | Centre inside, before → after | Mean GT coverage, before → after |
|---|---|---:|---:|
| All 1,752 states | Local | 256 → 348 | 0.147663306 → 0.187836592 |
| All 1,752 states | Wide | **760 → 720** | **0.424258897 → 0.412815496** |
| Native main-outside, 239 states | Local | 11 → 25 | 0.043167989 → 0.100593881 |
| Native main-outside, 239 states | Wide | 49 → 55 | 0.209653351 → 0.228472993 |

The full-population local gain is 254 newly included centres versus 162 lost centres. Wide loses 319 previously included centres and gains 279: **40 fewer included centres**, with mean GT coverage down **0.011443401** (1.144 percentage points). The 239-state native subgroup improves modestly at both budgets, but 214/239 local and 184/239 wide centres remain outside after fitting. **1,729 of 1,752 proposals change.** These observations do not support uniform global proposal improvement.

| Population | Mean rank, before → after | Median rank, before → after | Mean raw score margin, before → after |
|---|---:|---:|---:|
| All states | 217.803082 → 144.425228 | 150.5 → 90 | -0.656459432 → -0.104007882 |
| Native main-outside | 200.711297 → 100.585774 | 153 → 85 | -0.643936195 → -0.074457753 |

These rank/margin means are independently aggregated from stored **native-derived** values. Scoremaps are absent, so per-state rank/margin cannot be independently recomputed. Range, finiteness, margin sign and rank-zero/tie consistency pass. Rank counts strictly higher scores; tied scores may share rank zero. Margin approaching zero is not a calibrated probability or proof of accurate localization. Across all states, rank improves in 1,066, declines in 680 and ties in six.

Epoch mean loss is **6.316822, 6.057771, 5.996201, 5.953705, 5.925206**. Finite decreasing training loss from this one fit does not establish cross-run stability or held-out performance.

## Evidence limits and claim impact

- The **259 frozen tensors / six changed preview tensors**, finite saved state, optimizer step count and strict seven-field reload check are completed native source assertions. This audit did not load checkpoint tensors or rerun PyTorch/NN operations. The reload probe is one final-batch example, not optimizer-resume equivalence. See `scripts/train_global_preview_rank.py:185`, `:192` and `scripts/train_global_evidence_policy.py:61`.
- Native main-outside subgroup membership cannot be reconstructed without the original main geometry. Current-frame header snapshot identities were checked; images and original annotations were not recaptured.
- The diagnostic fixes the initial frame-0 template, old auxiliary role and old ROI budgets. Changed previews feed frozen controller weights, so unchanged controller weights do not ensure unchanged decisions or calibration. No new trajectory or updated B/C evaluation was performed.
- The supported claim is a completed preview fit with **mixed fixed-state training effects**, including improved local coverage and worse overall wide coverage. Generalization, stable multi-seed behavior, official PR/NPR/SR, predicted-box IoU, recovery, speed/cost, completed C, or success of the full research plan remain unsupported.

## Actual checks and retained evidence

| Audit step | Actual exit |
|---|---:|
| Initial source/input seal | 0 |
| Schema discovery, original attempt | 1 |
| Schema discovery, second attempt | 1 |
| Corrected schema discovery | 0 |
| Initial binding checker | 1 |
| Corrected native/config/log bindings | 0 |
| Dataset/sample replay | 0 |
| Independent paired result reconstruction | 0 |
| Scheduled-state/oracle scope checks | 0 |
| Final input seal | 0 |

The three unsuccessful attempts were checker/discovery errors: an observer-versus-CPU receipt path, `len()` applied to the integer seed, and comparing the sanity hash with the collected summary instead of the original native COMPLETE file. Failed scripts and exact stdout/stderr/exits were preserved. Corrected checks pass without any project-input edit.

All **96 reviewed input files** (70 repository, 26 private) remain byte-identical across their first/final seals, including all **31 sealed sources** and the plan. The companion JSON contains exact hashes, checker hashes, actual process receipts and safe aggregates. Private request, scripts, raw process outputs and failure originals are retained privately. No raw GT, coordinates, individual result rows or private paths are published here. This audit used Python 3.12.12 and NumPy 1.26.4, with no PyTorch/project import, NN/GPU/SSH call or installation. Later HANDOFF/paper/presentation claims are outside this review.
