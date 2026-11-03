# P1B B128 Capacity Results

Both exact admitted capacity tasks completed with native exit0 on GPU3.
Each rebuilt 128 real TRAIN states from the first eight published s0 caches,
then applied two policy updates to the same batch using the actual snapshot
FIT loss, frozen executor, AdamW groups and clipping. No checkpoint was
written and no collector state was modified.

| Measurement | P25 DropMoE | P26 SOT |
| --- | ---: | ---: |
| Batch size | 128 | 128 |
| Updates | 2 | 2 |
| Peak allocated | 3.719692 GiB | 3.719692 GiB |
| Peak reserved | 4.654297 GiB | 4.654297 GiB |
| First update | 13.452598 s | 15.532812 s |
| Second update | 9.938433 s | 11.789803 s |
| Loss, first / second update | 0.526785 / 0.524317 | 0.497187 / 0.495000 |
| Native duration, including setup | 87.185755 s | 116.688709 s |
| Free memory before CUDA model | 21,981 MiB | 21,674 MiB |
| Native state | COMPLETE/exit0 | COMPLETE/exit0 |

Both branches passed finite loss/gradient/state checks and the unchanged
non-policy tensor check. Learning rates were the existing preview1e-5 and
quality/gain1e-4, weight decay1e-4 and gradient clip1. These two updates
are not evidence of convergence or completed policy refresh.

The second update excludes initial CUDA warmup but remains a single repeated
batch measurement. The existing collector shared GPU3 throughout; CPU
reconstruction, worker prefetch and full dataset iteration are not timed by
the update measurements. No end-to-end FIT duration or FPS result is claimed.

B128 fits, but its measured memory usage is substantially below the user's
desired training utilization. It is not the final full-FIT batch choice.
A larger concrete capacity candidate, such as B512, should be separately
bound and measured before selecting a larger batch and corresponding learning
rates. Do not modify the admitted R1 tool/spec or reuse either task ID to
make that change. No OOM occurred and no fallback or restart was used.

The required full collection acceptance and separate full-FIT recipe
admission remain outstanding. The capacity result does not replace either
the earlier FIT smoke or full data acceptance.

## Provenance

- Tool SHA256:
  `f557eda41606796a6e22076b3b8cda57de67d2cb5d07863f9b375c0aac006be3`
- Exact admission review SOURCE_PASS0, with54 independent CPU checks:
  `e7ef17a50eb4529dac23d33147fdabc3fef21e25dca0d187db72f97cb2678c67`
- DropMoE measurement SHA256:
  `d7f4b7a214c76dc84fafbedc3cad1372cd71a2a9fc60983863449d77b7d9919d`
- SOT measurement SHA256:
  `917ba6d1714fc7b620abea10b4658ef258b2f95bf7415be673de45adfba78723`
- Raw exact specs, reviewed drafts, measurements and native status/logs:
  `reports/p1b_capacity_r1_20261102/{dropmoe,sot}/`

Local validation matched native task/GPU/command, final spec SHA, tool SHA,
two updates and finite values. Remote rechecks matched all original98 source
files and7 inputs on both snapshots after measurement.

At2026-11-03 02:16UTC, all eight original collector wrappers and children
were still live. P25 had268/882 sequences,125752 frames and24957 records;
P26 had310/882 sequences,149651 frames and29696 records. P25 temperatures
were74/70/65/63C and all four power limits remained250W.
