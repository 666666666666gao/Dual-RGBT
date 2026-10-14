# Spatial candidate opportunity after full preview fit

The full frozen GPU1 diagnostic completed at2026-10-14 19:18:24.118446 BJT with actual exit0:1752 old train states,265 unchanged state tensors, no optimizer/backward, and all Top1 proposals reproduced. The earlier two-state sanity also completed with actual exit0 and strict reload of scores/indices/points. All use the fitted preview checkpoint ad0599434e4b0024f206a108e6813e7c00127ecec829339832d00526890aef85 and unchanged source32/plan34eac262.

Candidates are collected before reading current GT: greedy cosine maxima on valid global256/grid32 cells, separated by half the original local ROI side. The initial Template64 descriptor, old auxiliary role and local/wide ROI sides remain fixed. Prefixes contain at most1/2/4/8 points;1729 states yielded8, the other23 yielded2–7. All239 native main-search-outside states yielded8. Main-search-outside membership remains an earlier native label; no saved main geometry was used to reconstruct it.

| Maximum candidates | All1752 local centre coverage | All1752 wide centre coverage | Outside239 local | Outside239 wide |
|---:|---:|---:|---:|---:|
|1|348|720|25|55|
|2|571|1055|36|80|
|4|841|1346|63|130|
|8|1171|1526|113|184|

Coverage means that at least one candidate's single ROI covers the GT centre. GT-area coverage is the maximum of individual ROI coverages, not their union or summed gains. No heavy auxiliary query or gain controller executed. These are oracle action-set diagnostics on fixed prior training states, not deployable selection, new trajectories, a tracking upper bound, official PR/NPR/SR or latency evidence.

The original collector failed with actual exit1 because its remote query retained the previous diagnostic's comparisons.json filename. The NN task itself completed successfully. The original script, SSH stderr/transport and native tool-result object are preserved privately. A corrected read-only collector recovered candidates.json and scores.npy from the **same** task; it did not restart or change NN execution. Final collection actual exit0; public completion SHA9e7bd7dccf63cb5df8596bd74f7a609a5f0a42a3fed8047bf934a01650e3c657. CPU analysis actuald17ee1 exit0 reconstructed every greedy choice from the saved full masked CNN maps and all ROI coverages, maximum error0. This improves provenance over earlier ranking-only diagnostics without saved score maps.

An explicitly post-hoc CPU control used10 fixed random seeds2027–2036. Random priorities use the same grid, validity, spatial separation and ROI sides. Each state/seed/K uses an identical actual prefix count for the CNN and random lists: min(K,CNN count,random count). This matches candidate counts, not full system execution cost. Random proposals are chosen before reading GT. The following counts average over the10 seeds:

| K | All CNN local/wide | All random local/wide | Outside CNN local/wide | Outside random local/wide |
|---:|---:|---:|---:|---:|
|1|348 /720|148.2 /482.8|25 /55|7.6 /27.1|
|2|571 /1055|271.1 /738.3|36 /80|12.7 /45.9|
|4|841 /1346|470.3 /1014.1|63 /130|24.0 /72.8|
|8|1171 /1526|740.6 /1295.7|113 /184|45.5 /116.3|

The candidate set has better geometric opportunity than this random control. That supports attempting to learn which single query is worthwhile. It does not establish that a controller can identify the useful candidate before encoding, that the auxiliary pixels improve localization, or that the additional scheduling is fast. The outside subset still misses55/239 centres even with8 wide candidates, and local opportunity is more limited.

Fresh independent audit completed with a scoped PASS/0 blockers:181 evidence-chain checks,124423 NumPy checks and141 claim/seal checks;115 audited inputs unchanged. All1752 greedy lists/13940 candidates/14016 coverage results and all10 random seeds/70080 paired prefixes reproduce exactly. There are329 shorter-than-requested prefixes and34 random lists shorter than CNN; paired actual counts are matched, not executed cost. Native outside labels and frozen/checkpoint assertions retain their stated provenance limits. Both SOURCE and result reviews are same-family/provisional, actual routing UNATTESTED. The audit sealed the pre-status-update RESULTS snapshot e003f75f690439c9f90091be0eca30923bd4d2ffeb5f89496f1dd0706f6a5388; this paragraph records its outcome afterward, without changing tables or claims. See [independent audit](../reports/GLOBAL_CANDIDATES_RESULT_AUDIT.md), [formal completion](../reports/global_candidates_formal_completion.json), [CPU reconstruction](../reports/global_candidates_formal_analysis.json), [post-hoc control plan](../reports/global_candidates_random_control_plan.json) and [per-seed control results](../reports/global_candidates_random_control.json).

Next is a separately reviewed frozen test of **executed** candidate-query localization benefit, followed by single-query gain selection if the action set is useful; see [next experiment plan](NEXT_CANDIDATE_UTILITY.md). The current deployable gain path still has four hard-coded query slots. Current NN code was not changed by this diagnostic. Full C,97-dev/245-test, real paired cost and three effective core controls remain outstanding. User baseline72.74/69.16/57.84 is retained, and target78/72/62 remains unmet.
