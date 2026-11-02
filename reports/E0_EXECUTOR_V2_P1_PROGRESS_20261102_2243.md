# P1A Executor TRAIN Progress

Snapshot time: 2026-11-02 22:43:49 UTC

## Native State

Both reviewed P25 R4 wrappers and trainer children were alive, and both
`steps.csv` files had fresh writes:

| Run | GPU | Global step | Update budget | CSV epoch |
|---|---:|---:|---:|---:|
| DropMoE ep20 | 0 | 32,900 | 50,000 | 14 |
| SOT control ep14 | 1 | 37,550 | 50,000 | 16 |

SOT has entered epoch 16, the first epoch in the reviewed retention window
16-20. DropMoE is still in epoch 14. The run directories contain only
`config.json` and `steps.csv`; no checkpoint files are present yet. Recheck
after the next epoch completion boundary before evaluating retained weights.

No dev/TEST inference or benchmark score exists. The latest individual losses
were finite but are not epoch means or tracking metrics. Do not early-stop
based on the previously observed TRAIN-loss plateau; continue the reviewed
20-epoch recipe.

## GPU State

At 22:43 UTC, P25 GPU0/1 were at 92%/90% utilization, 80/71 C and
247.80/245.34 W under the 250 W limit. GPU2/3 were idle at 12 MiB.

P26 GPUs0-3 were idle at 23/15/15/181 MiB with 0% utilization and no compute
processes. P26 still lacks the DropMoE ep20 starting checkpoint, and the
active P1A admission is bound to P25. No task was restarted, duplicated, or
cleaned up.
