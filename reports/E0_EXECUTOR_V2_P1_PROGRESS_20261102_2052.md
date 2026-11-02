# P1A Executor TRAIN Progress

Snapshot time: 2026-11-02 20:52 UTC

## Native State

Both P25 R4 wrappers remain alive and their steps.csv files are being
updated. Six of 20 epochs are complete.

| Run | GPU | Global step | Current epoch progress |
|---|---:|---:|---:|
| DropMoE ep20 | 0 | 15,400 / 50,000 | 400 / 2,500 |
| SOT control ep14 | 1 | 16,750 / 50,000 | 1,750 / 2,500 |

Mean TRAIN loss by completed epoch:

| Epoch | DropMoE | SOT control |
|---:|---:|---:|
| 1 | 3.071410 | 3.052869 |
| 2 | 2.870117 | 2.865242 |
| 3 | 2.771239 | 2.770273 |
| 4 | 2.753803 | 2.749885 |
| 5 | 2.690905 | 2.676070 |
| 6 | 2.682850 | 2.672850 |

TRAIN loss decreased throughout these six epochs, with smaller changes in
epochs 5-6. This is not evidence of tracking convergence or benchmark gains.
No dev/TEST inference was run and no benchmark metric is available.

## GPU State

At 20:52 UTC, P25 GPU0/1 each used 9,063 MiB, at 80/70 C and 247.85/247.35 W
under the 250 W limit. GPU2/3 were idle at 12 MiB each.

P26 GPUs0-3 were idle at 23/15/15/181 MiB with 0% utilization and no compute
processes. P1A admission remains bound to P25; P1B/architecture work still
requires its own admission.

No task was restarted and no weights were removed.
