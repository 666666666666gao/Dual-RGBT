# Completed frozen auxiliary-content diagnostic

2026-10-14. Full B training and the subsequent content diagnostic are complete. The frozen admission protocol remains a historical SOURCE_PENDING document; actual source review, real GPU smoke and terminal evidence establish the current status.

Full B completed 6,250 updates over 50,000 pairs, with all 882 training sequences visited. CPU comparison against its true initial full A checkpoint confirmed 251 unchanged tensors, 14 allowed changed tensors, Adam step6250 and finite saved state. Its checkpoint SHA is `b116d764b11213080a8f8a763883a8cfc96fd03c69847d43193fc94c0383f008`. This is completed B, not completed C or final test evaluation.

The frozen diagnostic used 20 train sequences, 8,854 actual tracked frames, 1,752 labelled visited states and 31,536 comparisons. All 265 state tensors remained unchanged; no backward or optimizer updates occurred. Within each state and region, TRUE/ZERO_AUX/SHIFT_AUX hold query coordinates, token budget, primary features and templates fixed. The intervention changes only current auxiliary high-resolution search pixels. Both cheap previews and auxiliary templates remain available, so this is a conditional content experiment.

| Region | TRUE mean IoU | ZERO_AUX | SHIFT_AUX | TRUE minus STOP |
|---|---:|---:|---:|---:|
| LOCAL_MAIN | 0.527540 | 0.464626 | 0.435487 | +0.009589 |
| LOCAL_ALTERNATE | 0.521659 | 0.452318 | 0.412246 | +0.003708 |
| GLOBAL_LOCAL | 0.106520 | 0.091417 | 0.066516 | -0.411431 |
| GLOBAL_WIDE | 0.192227 | 0.195202 | 0.091890 | -0.325724 |
| GT_CENTER_LOCAL | 0.644672 | 0.494621 | 0.470555 | +0.126721 |
| GT_CENTER_WIDE | 0.497801 | 0.432380 | 0.371911 | -0.020151 |

Native STOP mean IoU is 0.517951. Correctly located local auxiliary content has measurable value: LOCAL_MAIN TRUE−ZERO is +0.062915; GT_CENTER_LOCAL is +0.150052. However, the small one-step LOCAL_MAIN gain over STOP does not establish trajectory improvement. GLOBAL_WIDE TRUE is slightly below ZERO, so a universal benefit claim is contradicted by this population. GT-centred controls supply privileged location information; their scores are diagnostic and cannot be reported as a deployable policy or trajectory upper bound. Cyclic SHIFT is spatial misregistration, not a new distractor identity.

The immediate bottleneck is the external query location. Of 239 states labelled main-centre-outside by the native run, GLOBAL_LOCAL covers the GT centre in 11 and GLOBAL_WIDE in 49. The query membership and area coverage were independently recomputed from private query geometry and dataset GT. Main executed boxes/search geometry were not saved, so STOP IoU and membership in that subgroup remain native-derived.

A separate header-only CPU geometry check chose the valid 32x32 grid cell nearest GT, preserving the original ROI side. It covered the target centre in all 1,752 states, including all 239 main-outside states, at both existing budgets; local mean GT area coverage in the outside subgroup was 0.995888, wide 1.0. Thus these samples do not justify increasing ROI size to repair coarse-grid quantization. This oracle check is geometry only, with no NN, predicted boxes, accuracy metric or deployable selector. At global input256, outside-subgroup target width/height medians are 9.6/15.733 pixels, documenting actual appearance scale without proving a scale encoder is necessary.

The current preview already uses the initial template's central 4x4 descriptor; it does not average the whole template or use a contaminated recent template descriptor. Existing global maps have Gaussian sigmoid-BCE supervision. The next concrete trial fits the existing six-tensor preview CNN using a unit-mass spatial ranking objective, with unchanged inference geometry and no added parameter. Loss imbalance is a hypothesis to test, not an established root cause. Any preview update requires policy label/calibration refresh before full C/dev/test. See [ranking protocol](GLOBAL_PREVIEW_RANK_PROTOCOL.md).

Independent result audit: deterministic PASS, substantive WARN, zero blockers. All 259 recorded checks passed; 31,536 IoUs/query-coverages and native/analyzer aggregates had maximum error0, with 69/69 inputs unchanged. Original native tool result was retained; separate SSH stdout/stderr and the parent's original CPU-analysis receipt were not saved. The reviewer recorded its own independent computation. Review is fresh same-family/provisional, actual routing UNATTESTED, and not blinded. See [audit](../reports/GLOBAL_CONTENT_RESULT_AUDIT.md), [CPU aggregates](../reports/global_content_formal_b_analysis.json), [geometry aggregates](../reports/global_grid_geometry_analysis.json).

No full LasHeR PR/NPR/SR, final-model latency, seed stability, paired A/B improvement or three effective innovations follows from this diagnostic. Full C, dev selection, full245 test, paired real-system cost and module controls are still required. The user baseline72.74/69.16/57.84 remains supplied and is not rerun.
