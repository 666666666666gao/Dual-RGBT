# P1A Executor TRAIN Progress

Snapshot time: 2026-11-02 22:58:49 UTC

## Native State

Both reviewed P25 R4 wrapper/trainer pairs remained alive and their CSVs
continued updating:

| Run | GPU | Global step | Update budget | CSV epoch |
|---|---:|---:|---:|---:|
| DropMoE ep20 | 0 | 35,300 | 50,000 | 15 |
| SOT control ep14 | 1 | 40,350 | 50,000 | 17 |

The SOT run produced `epoch_016.pth` after completing epoch 16. The file is
1,237,700,105 bytes with SHA256
`c60508e7a2f04f0fc44713608801fbd5e3fa4a8bac172f7fd0f0ed04cefbe02e`.
This confirms the checkpoint file was written and hashed; full-training
completion, checkpoint reload, and benchmark scoring remain unverified.
DropMoE has not yet entered epoch 16 and has no checkpoint file. Do not score
or select the epoch-16 checkpoint while the P1A runs are incomplete.

There are no benchmark scores. No dev/TEST inference was run. Preserve
`epoch_016.pth` and continue both reviewed 20-epoch runs.

## GPU State

At 22:58 UTC, P25 GPU0/1 were at 94%/98% utilization, 81/71 C and
249.62/245.38 W under the 250 W limit. GPU2/3 were idle at 12 MiB.

P26 GPUs0-3 were idle at 23/15/15/181 MiB with 0% utilization and no compute
processes. P26 still lacks the DropMoE ep20 initialization checkpoint, and
the active P1A admission remains bound to P25. No task was restarted,
duplicated, or cleaned up.
