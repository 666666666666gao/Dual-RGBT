# P1A Executor TRAIN Progress

Snapshot time: 2026-11-02 23:13:19 UTC

## Native State

Both reviewed P25 R4 wrapper/trainer pairs remained alive with fresh CSV
writes:

| Run | GPU | Global step | Update budget | CSV epoch |
|---|---:|---:|---:|---:|
| DropMoE ep20 | 0 | 37,600 | 50,000 | 16 |
| SOT control ep14 | 1 | 43,100 | 50,000 | 18 |

The SOT run has written two retained checkpoints:

| Epoch | File size | SHA256 |
|---:|---:|---|
| 16 | 1,237,700,105 bytes | `c60508e7a2f04f0fc44713608801fbd5e3fa4a8bac172f7fd0f0ed04cefbe02e` |
| 17 | 1,237,700,105 bytes | `4f129a66da57f39350ac8dd3b1c563086870e94556d72902df90ab3b68096161` |

Both files were hashed but are not yet reload-verified or benchmarked. The
DropMoE run has entered epoch 16 but has not written a checkpoint yet. Both
formal runs remain active; no checkpoint selection or TEST inference has
occurred.

## GPU State

At 23:13 UTC, P25 GPU0/1 were at 99%/94% utilization, 82/71 C and
249.27/247.45 W under the 250 W limit. GPU2/3 were idle at 12 MiB.

P26 GPUs0-3 were idle at 23/15/15/181 MiB, 0% utilization and no compute
processes. P26 lacks the DropMoE ep20 initialization checkpoint, and the
active P1A admission remains bound to P25. No run was restarted or duplicated,
and no weights were deleted.

There are no benchmark scores. No dev/TEST inference was run.
