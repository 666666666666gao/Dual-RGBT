# Frozen descriptor-scale comparison

The full comparison completed at2026-10-14 18:46:28.008695 BJT with actual exit0:1752 fixed train states,3504 comparisons,265 unchanged tensors and no gradient/optimizer/backward. Its earlier two-state GPU1 admission passed strict reload with actual exit0. Both runs used the completed full882 preview-fit checkpoint ad0599434e4b0024f206a108e6813e7c00127ecec829339832d00526890aef85.

| Scope | Measure | Template64 | Template32 |
|---|---|---:|---:|
| All1752 | Local centre coverage |348|324|
| All1752 | Wide centre coverage |720|711|
| All1752 | Local mean GT-area coverage |0.187837|0.178434|
| All1752 | Wide mean GT-area coverage |0.412815|0.402505|
| Native main-outside239 | Local centre coverage |25|34|
| Native main-outside239 | Wide centre coverage |55|56|
| Native main-outside239 | Local mean GT-area coverage |0.100594|0.138684|
| Native main-outside239 | Wide mean GT-area coverage |0.228473|0.241615|

Template32 improves some native outside cases but worsens overall local and wide coverage. This is not an overall fix. Global pixels256, grid32, old auxiliary roles, shared CNN weights and ROI sides were held fixed. The central template feature crop changed from4x4 to2x2, altering receptive-field context as well as appearance scale. Both use the initial template; these data do not isolate scale alone, appearance ageing or distractor identity.

CPU analysis actual exit0 reproduced all3504 query membership/area calculations with maximum error0 and matched every Template64 baseline row to the previous fitted ranking. Main-outside labels and cosine ranks/margins remain native-derived, because this comparison did not save full score maps. Fresh independent audit completed with323 checks,73 sealed inputs unchanged, frozen-result integrity PASS and overall WARN/0 blockers. Its claim limits retain the mixed results and the absence of score-map reconstruction, isolated scale causality, tracking metrics or latency evidence. SOURCE admission was PASS/0 blockers; both reviews are same-family/provisional with actual model routing UNATTESTED. See [independent audit](../reports/GLOBAL_DESCRIPTOR_SCALE_RESULT_AUDIT.md), [CPU analysis](../reports/global_descriptor_scale_formal_analysis.json) and [formal completion](../reports/global_descriptor_scale_formal_completion.json).

The scale hypothesis was informed by the actual fixed source of [SiamFC exemplar/search crop geometry](https://github.com/huanglianghua/siamfc-pytorch/blob/39572051d7c3d9850ca6100efa04091e18c2f706/siamfc/siamfc.py). This is design inspiration, not a reproduction or evidence that scale is the sole cause here. Exact retrieved-source provenance is retained in reports/global_descriptor_scale_reference_provenance.json.

Next, a separately frozen candidate-set diagnostic checks at most1/2/4/8 spatially separated maxima without changing the fitted CNN or Template64. It measures the best **single** ROI's GT coverage after collection, not a deployable oracle or simultaneous heavy queries. Full C,97-dev/245-test, real paired system cost and the three core controls remain outstanding;78/72/62 remains unmet.
