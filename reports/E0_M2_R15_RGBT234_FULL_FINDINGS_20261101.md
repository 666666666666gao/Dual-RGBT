# R15 E0+M2 RGBT234 Full TEST Result

Native eval_e0_m2_rgbt234_r15_p26_20261101 completed with exit0 on26 GPU1.
Official scoring succeeded once and complete artifact/trace-weighted compute
verification passed. All234 sequences and116650 frames were included.

| Variant | MPR | MSR | FPS |
| --- | ---: | ---: | ---: |
| Matched frozen E0 | 90.459440 | 68.444253 | 11.810636 |
| E0+M1 guard | 90.762147 | 68.720338 | 11.631646 |
| R15 E0+M2 | 85.176301 | 64.680807 | 8.606659 |
| M2 minus E0 | -5.283140 | -3.763445 | - |

The same frozen parent SHA256
ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c
is used by these variants. R15 lacks M1/M3. It misses targets90.9/65.6
by5.723699/0.919193 points. All results are exploratory TEST only,
test_eligible=false; no TEST training, threshold/checkpoint/seed selection
or independent final-validation claim.

103490697 parameters; mean116.189098ms/P95124.563591ms.
Mean tracked-frame103.440624GMAC and all-frame103.376183GMAC include all
4771 candidate-generation calls, identity head and per-sequence anchor
(anchor only in all-frame costs). Counts cover convolution/dense/attention,
not normalization/nonlinearities/ROI/crops/decoding/memory/bookkeeping.
FPS includes decoding and initialization under concurrent workloads; it is
not an isolated throughput comparison.

M2 selected a replacement4521/4771times; this frequency is not accuracy.
Two completed benchmarks now show regression for the independent25frame
replacement policy. The full LasHeR run remains unchanged. See
E0_M2_R15_RGBT210_FULL_FINDINGS_20261101.md for source mechanisms and
the cadence qualification: M2 calls1,26,51,... cannot execute its own
divisible-by5 template-write branch in full TEST. Do not attribute this
regression to that unexecuted branch.

Existing TRAIN metadata shows no-match holdout recognition62/132,
versus2685/3318 correct candidate picks. State-stratified cache replay
and the missing M1 failure gate need investigation on TRAIN, not TEST
parameter selection. A separate source diagnosis was requested from the
human-authorized existing review chat; no revised policy has been launched.

Score SHA256
c7ec22719407a7a3ad78e754b4a27e09f0cedf9725f98aabc40f22c8d9f563b4.
Compute SHA256
13f8c63b692b360593e4d6324a00bb6e61f78960251f74f9425e82470dc75621.
Plan SHA256
72524ae570aa79b138bd7e138affd1e9bc6e4a1fd3c535871b1ecf4dcba4485c.
Score/curves/compute/log and runtime config/plan/COMPLETE/native status
are archived in reports/e0_m2_online_scores_r15_20261101.
