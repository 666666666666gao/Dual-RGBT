# P1A Executor TRAIN Progress

Snapshot time: 2026-11-02 20:18-20:19 UTC

## Native State

Both P25 R4 tasks remain RUNNING with live wrappers and trainer children.

| Run | GPU | Global step | Completed epochs | Epoch 5 progress |
|---|---:|---:|---:|---:|
| DropMoE ep20 | 0 | 10,050 / 50,000 | 4 | 50 / 2,500 |
| SOT control ep14 | 1 | 10,450 / 50,000 | 4 | 450 / 2,500 |

Both use 20 epochs, 20,000 samples per epoch, batch 8, seed 2027. The
reviewed trainer retains epochs 16-20; no checkpoint is expected yet.

## TRAIN Loss

Means are computed from the `loss` column in `steps.csv` over all 2,500
updates in each completed epoch.

| Run | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
|---|---:|---:|---:|---:|
| DropMoE | 3.07141 | 2.87012 | 2.77124 | 2.75380 |
| SOT control | 3.05287 | 2.86524 | 2.77027 | 2.74988 |

TRAIN loss continues to decrease, with smaller changes in epochs 3-4. Four
epochs are insufficient to establish convergence or benchmark improvement.
No dev/TEST evaluation was run and no benchmark metrics are available.

## GPU State

At 20:18 UTC, P25 GPU0/1 used 9,063 MiB each, at 80/71 C and 248.69/237.26 W
under the 250 W power limit. GPU2/3 were idle at 12 MiB each. At 20:19 UTC,
P26 GPUs0-3 were idle at 23/15/15/181 MiB, with no compute processes.

No task was restarted and no weights were removed.
