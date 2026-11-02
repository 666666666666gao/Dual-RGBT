# Fine-Tuned E0 Epoch40 Full RGBT210 And RGBT234

Both native tasks completed with exit0, unchanged evaluation weights and
zero optimizer/backward updates. The official scorer ran once for each
complete dataset; matching-source route-weighted compute summaries passed.
These are the user's requested complete E0 executor fine-tuning results,
not the unrelated stalled CADTrack trainer or an M1/M2 model.

| Benchmark | Complete Coverage | Precision | Success | Target Precision / Success |
| --- | --- | ---: | ---: | --- |
| RGBT210 | 210 sequences / 104706 frames | PR88.368511 | SR65.080225 | 88.7 / 62.9 |
| RGBT234 | 234 sequences / 116650 frames | MPR90.362006 | MSR67.470107 | 90.9 / 65.6 |

RGBT210 PR is0.331489points below target; SR is2.180225points above.
RGBT234 MPR is0.537994points below target; MSR is1.870107points above.
Original protected E0 RGBT210 was88.828508/66.124608; epoch40 differs
by-0.459997/-1.044383points. This completed result does not establish a
fine-tuning gain, convergence or checkpoint preference.

| Benchmark | Parameters | FPS | Mean / P95 ms | Mean Tracked-Frame GMAC |
| --- | ---: | ---: | --- | ---: |
| RGBT210 | 103374215 | 11.686902 | 85.565875 / 115.464386 | 53.697104 |
| RGBT234 | 103374215 | 12.559872 | 79.618647 / 106.762438 | 53.336867 |

FPS includes decoding and sequence initialization under concurrent load,
excluding model loading; it is not isolated throughput. MACs weight the
actual full-run route counts with matching-source CPU arithmetic profiles.
They count convolution/dense/attention and exclude initialization,
nonlinearities, normalization, crop/ROI, decoding, memory and bookkeeping.
Mean GMAC denominators exclude the210/234initialization frames.

## Archived Evidence

All paths below are under
`reports/e0_ep20_p26_evaluation_r1_20261102/`.

- `eval_e0_ep20_e040_rgbt210_r1_p26_20261102.json`:
  SHA256984e4834f75719a39afde204d2dd2e1af4c52a56a4b651fdbe58f1ca46503959.
- `eval_e0_ep20_e040_rgbt234_r1_p26_20261102.json`:
  SHA25669f7affcbfb54060eee43fd03ca7e192e386e37df8e1d0b3e4c3dfbb7af5dbf2.
- Corresponding `.npz` official curves, `_compute.json` summaries and
  `runtime/<task>/config.json`, `COMPLETE.json`, `native_status.json`.

Both scores bind epoch040 checkpoint
eb63d329d382df1e8fc5bc69e73c41a11d1c263bd4ad7f4a8af7db1107b3026f,
exact evaluated sources and toolkit protocol. The computation summaries
verify score/native/source/manifest/artifact identities independently.

## Continued Queue

GPU1 remains at most3concurrent admitted jobs. After each genuine
completion freed a slot, the next fixed plan item was launched:

- Epoch39 LasHeR wrapper1140155, child1140162, started03:28:27UTC.
  Real child and first two completed sequence receipts verified03:30UTC.
- Epoch39 RGBT210 wrapper1147363, started03:32:56UTC.
  Actual child1147373/live nativeRUNNING/new trace and completed sequences
  verified03:36UTC. Launch receipt is archived.

The original epoch40 LasHeR and GPU0 R15 LasHeR remain unchanged.
Next queued item is epoch39 RGBT234, then epoch38LasHeR. No duplicate
scoring, failed task ID reuse, new25GPU work or26GPU2/3 interference.

All full TEST results are exploratory_test=true/test_eligible=false.
Report every fixed epoch's outcome. No TEST checkpoint, seed, threshold,
parameter or training-recipe selection, or independent final validation.
The original three-benchmark goal remains unmet.
