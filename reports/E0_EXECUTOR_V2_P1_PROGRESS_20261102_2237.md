# P1A Executor TRAIN Progress and Loss Trend

Snapshot time: 2026-11-02 22:37:03 UTC

## Native State

Both reviewed P25 R4 wrapper/trainer pairs remained alive, with fresh CSV
writes:

| Run | GPU | Global step | Update budget |
|---|---:|---:|---:|
| DropMoE ep20 | 0 | 31,850 | 50,000 |
| SOT control ep14 | 1 | 36,300 | 50,000 |

Every logged loss in both CSVs was finite. Epoch-wise mean TRAIN losses:

| Epoch | DropMoE | SOT control |
|---:|---:|---:|
| 8 | 2.586955 | 2.587280 |
| 9 | 2.565857 | 2.564624 |
| 10 | 2.569954 | 2.573797 |
| 11 | 2.492776 | 2.504730 |
| 12 | 2.476060 | 2.476517 |
| 13 | 2.461757 (2,001 rows) | 2.460984 |
| 14 | not started | 2.474019 |
| 15 | not started | 2.477621 (1,451 rows) |

TRAIN loss flattened around epochs 9-10 and then declined gradually toward
about 2.46 by epochs 12-13. The SOT loss remains around 2.47 in epoch 14 and
the partial epoch 15. This suggests the optimization loss is nearing a
plateau around epochs 12-14, but does not establish tracking-metric
convergence, PR/NPR/SR gains, or a safe early-stop point. Continue the
reviewed 20-epoch recipe; do not use TEST for selection.

There are no benchmark scores or dev/TEST runs. Both run directories still
contain only `config.json` and `steps.csv`; no checkpoint has been written.
The reviewed trainer retains epochs 16-20.

## GPU State

P25 GPU0/1 were at 91%/99% utilization, 81/71 C and 247.87/249.89 W under
the 250 W limit. GPU2/3 were idle at 12 MiB.

P26 GPUs0-3 were idle at 23/15/15/181 MiB with 0% utilization and no compute
processes. The DropMoE ep20 initialization checkpoint remains absent on P26,
and P1A remains host-bound to P25. No duplicate run, restart, or weight
cleanup occurred.
