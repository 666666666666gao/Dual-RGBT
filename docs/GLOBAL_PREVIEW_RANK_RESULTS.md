# Full preview-ranking fit: mixed proposal results

The fit completed at2026-10-14 17:05:27.615879 BJT with actual exit0: 50,000 pairs, 6,250 optimizer updates, all882 train sequences seen. Six preview tensors changed, 259 other tensors remained bit-equal, and seven network outputs passed strict save/reload comparison. Final checkpoint SHA is ad0599434e4b0024f206a108e6813e7c00127ecec829339832d00526890aef85. All6,250 logged losses were finite. These are training facts, not full C or test performance.

The comparison holds the old1,752 visited train states, auxiliary roles, initial templates and ROI sides fixed. Before proposals reproduce the prior full-B diagnostic. Both versions use global256/grid32 and local/wide ROI sizes; the changed network objective is unit-mass spatial cross entropy. Ground-truth cell rank and score margins have no saved full response maps and remain native-derived. Query membership and area coverage were independently recomputed from private proposals and dataset GT.

| Scope | Measure | Before | After |
|---|---|---:|---:|
| All1,752 | Local centre coverage | 256 | 348 |
| All1,752 | Wide centre coverage | 760 | 720 |
| All1,752 | Local mean GT-area coverage | 0.147663 | 0.187837 |
| All1,752 | Wide mean GT-area coverage | 0.424259 | 0.412815 |
| Native main-outside239 | Local centre coverage | 11 | 25 |
| Native main-outside239 | Wide centre coverage | 49 | 55 |
| Native main-outside239 | Local mean GT-area coverage | 0.043168 | 0.100594 |
| Native main-outside239 | Wide mean GT-area coverage | 0.209653 | 0.228473 |

Nearest-GT-cell mean rank improved from217.803 to144.425 overall and200.711 to100.586 in the native outside subgroup. That ordinal change does not ensure that the chosen maximum covers the target. The overall wide result deteriorated. Outside local/wide coverage remains far below the earlier239/239 reachable-centre geometry diagnostic. The result supports continuing to fix proposals; it does not support treating loss balancing as a solved recovery mechanism or proceeding directly to final C/test with unchanged controllers.

CPU analysis actual exit0 reconstructed coverage with max error0 and found1,729 changed proposal states. Independent audit is WARN/0 blockers: all6,250 continuous loss rows and the50,000 sampled training sequence choices were verified, each epoch visiting all882 sequences; paired proposals and aggregates agree.96 inputs and source31 stayed unchanged. Native rank/margin, frozen-state/reload assertions, single seed and training-reference scope are retained as limits. See [independent audit](../reports/GLOBAL_PREVIEW_RANK_RESULT_AUDIT.md) and [CPU analysis](../reports/global_preview_rank_formal_analysis.json).

The subsequent frozen template64 versus template32 comparison completed with mixed results: outside local centre coverage25→34, but overall348→324 and wide720→711. It does not solve proposal selection. Global pixels, grid and query ROI sides were unchanged; central feature crops4x4/2x2 also alter receptive-field context, so scale alone was not isolated. Both still use initial templates. See [scale results](GLOBAL_DESCRIPTOR_SCALE_RESULTS.md). A separate candidate-set diagnostic passed SOURCE review with0 blockers; its real GPU admission is still pending and it and checks whether a few separated global peaks offer useful single-query geometric opportunity.

Changed preview features require refreshed quality/gain supervision and dev calibration before deployment. Full C, complete97-dev/245-test, real-system efficiency and the three core module controls are outstanding. User baseline72.74/69.16/57.84 is retained without another baseline accuracy run; target78/72/62 remains unmet.
