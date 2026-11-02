# P1A Executor TRAIN Progress

Snapshot time: 2026-11-02 20:05 UTC

## Native State

Both P25 R4 tasks remain RUNNING with live wrappers and trainer children.

| Run | GPU | Global step | Completed epochs | Epoch 4 progress |
|---|---:|---:|---:|---:|
| DropMoE ep20 | 0 | 7,850 / 50,000 | 3 | 350 / 2,500 |
| SOT control ep14 | 1 | 7,950 / 50,000 | 3 | 450 / 2,500 |

Both use 20 epochs, 20,000 samples per epoch, batch 8, seed 2027. The
trainer retains epochs 16-20; no checkpoint is expected before epoch 16.

## TRAIN Loss

Means below are computed from the `loss` column in `steps.csv` over all
2,500 updates in each completed epoch.

| Run | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| DropMoE | 3.07141 | 2.87012 | 2.77124 |
| SOT control | 3.05287 | 2.86524 | 2.77027 |

The observed TRAIN loss is decreasing through epoch 3. This is not evidence
of benchmark improvement or convergence; only three epochs have completed.
No dev or TEST evaluation was run, and no benchmark metrics are available.

## GPU State

At 20:05 UTC, P25 GPU0/1 used 9,063 MiB each, at 79/70 C and 247.89/229.07 W
under the 250 W limit. GPU2/3 were idle at 12 MiB each. P26 GPUs0-3 were idle
at 23/15/15/181 MiB, with no compute processes.

No task was restarted and no weights were removed.
