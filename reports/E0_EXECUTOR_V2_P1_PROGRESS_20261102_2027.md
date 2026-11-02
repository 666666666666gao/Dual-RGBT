# P1A Executor TRAIN Progress

Snapshot time: 2026-11-02 20:27 UTC

## Native State

Both P25 R4 tasks remain alive under their original wrappers and trainer
children. Five of 20 epochs are complete; both are in epoch 6.

| Run | GPU | Global step | Current epoch progress |
|---|---:|---:|---:|
| DropMoE ep20 | 0 | 11,300 / 50,000 | 1,300 / 2,500 |
| SOT control ep14 | 1 | 11,950 / 50,000 | 1,950 / 2,500 |

Both continue to write finite TRAIN losses. No dev or TEST evaluation was
run and no benchmark metrics are available.

## GPU State

At 20:27 UTC, P25 GPU0/1 used 9,063 MiB each at 81/71 C and 249.13/249.45 W,
below the configured 250 W limit. GPU2/3 were idle at 12 MiB each.

P26 GPUs0-3 were idle at 23/15/15/181 MiB, 0% utilization, with no compute
processes. The user authorized use of all four GPUs. The P1A R4 admission is
bound to P25, and P26 lacks the DropMoE ep20 initialization checkpoint.
P1B and subsequent architecture stages have not passed their own required
admission; no duplicate or unreviewed GPU job was launched.

No task was restarted and no weights were removed.
