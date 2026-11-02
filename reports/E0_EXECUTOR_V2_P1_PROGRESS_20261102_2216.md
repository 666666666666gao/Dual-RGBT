# P1A Executor TRAIN Progress

Snapshot time: 2026-11-02 22:16:43 UTC

## Native State

Both reviewed P25 R4 wrappers and trainer children were alive. Their
`steps.csv` files had fresh writes at the snapshot:

| Run | GPU | Global step | Progress |
|---|---:|---:|---:|
| DropMoE ep20 | 0 | 28,650 / 50,000 | 57.3% |
| SOT control ep14 | 1 | 32,450 / 50,000 | 64.9% |

The most recent logged loss values were finite. They are individual sampled
steps, not epoch means or tracking metrics. No dev/TEST inference or benchmark
score exists. The run directories contain only `config.json` and `steps.csv`;
no checkpoint has been written yet. The reviewed configuration retains epochs
16-20.

## GPU State

At 22:16 UTC, P25 GPU0/1 used 9,063 MiB each at 94%/99% utilization,
80/71 C, and 248.44/246.26 W under the 250 W limit. GPU2/3 were idle at
12 MiB each.

At 22:11 UTC, P26 GPUs0-3 were idle at 23/15/15/181 MiB with 0% utilization
and no compute processes. The user authorized all four P26 GPUs. P26 has the
LasHeR TRAIN tree and the ep14 checkpoint, but not the DropMoE ep20
initialization checkpoint. The active P1A admission is bound to P25; P1B and
P2 still require P1A completion and their own admission. No duplicate or
unreviewed job was started.

No task was restarted, no weights were removed, and no benchmark result is
claimed.
