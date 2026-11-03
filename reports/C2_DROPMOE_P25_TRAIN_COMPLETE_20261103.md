# C2 DropMoE P25 Full TRAIN Completion

Observed 2026-11-03 11:44 UTC (19:44 UTC+8) from native host 2025.
Both admitted C2 training tasks completed their fixed 20-epoch recipes with
native exit code 0. This is executor TRAIN completion, not convergence,
tracker adaptation, benchmark evidence, or goal completion.

| Variant | Native task / wrapper | GPU | Updates | Epoch 20 seconds | Epoch 20 mean loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Zero | `train_c2_dropmoe_zero_full_r1_p25_20261103` / 4074337 | 1 | 50,000 | 863.350389115003 | 2.5728894776 |
| FiLM | `train_c2_dropmoe_film_full_r1_p25_20261103` / 4074338 | 2 | 50,000 | 865.306132145997 | 2.5707488955 |

The native receipts report `COMPLETE`, exit 0, with the exact admitted
commands. Each run's `COMPLETE.json` reports 20 epochs, 50,000 updates,
5,000 updates for each of ten routes, preserved policy modules,
`policy_refresh_required=true`, and `benchmark_evidence=false`. Independent
read-only parsing confirmed 50,000 CSV rows and finite loss, gradient norm,
and elapsed-time values. Per-epoch mean TRAIN losses are recorded below;
their trajectory is descriptive and is not a convergence or selection test.

P25 zero mean loss by epoch:
`2.723912, 2.709397, 2.645441, 2.680831, 2.644270, 2.626329, 2.588679,
2.567892, 2.551465, 2.598422, 2.628431, 2.599686, 2.609750, 2.589839,
2.609289, 2.573003, 2.569602, 2.564934, 2.576034, 2.572889`.

P25 FiLM mean loss by epoch:
`2.724851, 2.709956, 2.649481, 2.684782, 2.637112, 2.635219, 2.594202,
2.569167, 2.558126, 2.593842, 2.639242, 2.603315, 2.613338, 2.589523,
2.607925, 2.577117, 2.566829, 2.561633, 2.566999, 2.570749`.

Completion and retained checkpoint hashes from the native `COMPLETE.json`:

| Artifact | Zero SHA256 | FiLM SHA256 |
| --- | --- | --- |
| `COMPLETE.json` | `826e5e5edbdcaa969a64b32fa9c886ab6fbcdda4fe9c0ddc29ca0bae9a210319` | `efb8fc3f7df9467641c54fb961e937726b687942a39758d5b58c880fac811345` |
| `config.json` | `d22c34eb844a70d797909848bade5b5c2cf22f5d6e84bb10800a5b347dff1a20` | `d1a855202cd3b2a3d63b0a2122f22358c2a0831033e84ec2686a662e7f9b9dd3` |
| `precision.json` | `8fba9d3d90e81208ddaf86ac94bffd762ee8834b463586421b9c74e7c7a4fecc` | `8fba9d3d90e81208ddaf86ac94bffd762ee8834b463586421b9c74e7c7a4fecc` |
| `steps.csv` | `6c0ec05a773885813b95951221519d647ae6ec4f4a56cfd1f079f6b4dc6d90c8` | `d44196f443a20271d359385dd70dae35ead0d922784efcd81419815edb6336c5` |
| `epoch_016.pth` | `2518f25cce23cceb48fc5f54802cb262f61b45b45d3e2299283f95314434c3bb` | `1140bbe33957c507fe195cb2289f9812457d08d89eb65a5a63ced9e05a95480e` |
| `epoch_017.pth` | `41729fe11737df7e196b3ebfe1124fcd689833fe59a4b421504e2724395c1a11` | `51a4705577756f46fec9f501242b0052c18434c310155af1674e01494197afc9` |
| `epoch_018.pth` | `6f69aeee979502aff7105b532c0b00f60e3adb58a3b5afaf0fdbab02cba22820` | `0f65a33c6a8f632e74d0fef5afcf7e04886972a626f68acdb92423c1be28977e` |
| `epoch_019.pth` | `d7a6be2bcd097abd84582e6f4321191a1486358340b7c85f0017ab9d2d4c881c` | `f8cd1ad775212baedf41af38320991cbf6a7c1f96361b35ffdae03ac4944b7d4` |
| `epoch_020.pth` | `5835a4162e436f2f42aabcd0e94c6bfb9b123457f017a3f9090e2ad778dbbf97` | `41fd6f9d889037d283661f4cfed6de9314ddc9a429c16c3aeb7a78a1bdcf8d09` |

At the observation, both C2 wrappers had exited and GPUs 1/2 were idle.
The independent P25 P1B FIT remained active on GPU3 at 100% utilization,
68 C and 249.06 W (within the 250 W / 85 C limits). Do not terminate or
repurpose it based on this C2 completion.

No C2 checkpoint was passed to the old P1B loader. The reviewed local
C2-aware loader is not deployed or admitted: actual completed-checkpoint
reconstruction, new source/input bindings, newly collected TRAIN records,
and a separately reviewed policy-refresh plan remain required before any
follow-on GPU work or full-tracker claim. TEST remains exploratory and
cannot select a variant or checkpoint.
