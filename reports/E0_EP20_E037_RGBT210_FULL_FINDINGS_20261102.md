# E0 Epoch37 Full RGBT210 Result

`eval_e0_ep20_e037_rgbt210_r1_p25_20261101` is nativeCOMPLETE/exit0,
210sequences/104706frames. The existing CPU controller scored once and
completed the route-compute summary; the executor only read/archived results.

| Measurement | Value |
| --- | ---: |
| PR | 89.1079925776 |
| SR | 65.3956417928 |
| FPS | 16.3759061688 |
| Mean frame milliseconds | 61.0653230235 |
| P95 frame milliseconds | 82.4037782500 |
| Parameters | 103374215 |
| Mean tracked-frame GMACs | 53.6322010658 |

Descriptive target differences are+0.4079925776PR and+2.4956417928SR.
This fixed result exceeds88.7/62.9 on exploratory TEST. It does not select a
checkpoint for promotion, independently validate the final method, establish
LasHeR/RGBT234 targets or establish the three-module claim. Report all fixed
results; no threshold, seed or recipe was changed from these TEST outcomes.

For comparison, the other already fixed complete RGBT210results are:

| Epoch | PR | SR |
| --- | ---: | ---: |
| 37 | 89.107993 | 65.395642 |
| 38 | 88.561048 | 65.030110 |
| 39 | 89.018952 | 65.402349 |
| 40 | 88.368511 | 65.080225 |

FPS is actual concurrent-load timing, not isolated relative speed. Standard
conv/matrix MACs weight actual route/token counts and exclude initialization.
The matching49source fixed-shape CPU profile was originally executed with
epoch1TRAIN-smoke weights. Reuse follows identical shapes/routes; this is
not a newly measured epoch37profiling forward. Other operation categories
are outside that profile's standard MAC convention.

Native/checkpoint/full-count bindings and compute's exact score/profile
hashes were checked. Raw score/curves/compute/config/receipt/native are
archived byte-exact under`e0_ep20_p25_full_scores_r1_20261102/`.

- Score SHA256:`5a07152a9ca0055108a62fab7471e4cb7d2c81b60f7fe1d7db3b41a90b99634b`.
- Compute SHA256:`c4148e40df7ebe2bff6912df69fe4828871f5be70ca10d36bef959d37a9ff675`.
- Profile SHA256:`708d3aaafb486a5f21a28bf134207a3ae2bf59e0c61607336e9a6addd4a8989e`.
