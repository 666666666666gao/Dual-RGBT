# P1B B640 Capacity R2

Both exact reviewed B640/two-update tasks were launched on physical GPU3,
sharing the card with the unchanged corresponding TRAIN collector. Review:
`E0_EXECUTOR_V2_P1B_CAPACITY_ADMISSION_R2_20261102.json`,
SHA `2a327d062f25bb0b54b833cf3f3e7c08bb688e1a5821a670b938f903d332e9ad`.
The new supplemental tool SHA is
`9d4b555a429b8b45a43e6bb89933534bad91f067190e67c981a316358c022212`.
Original deployed R1 tools, specs and outputs remain unchanged.

Terminal states verified at 2026-11-03 02:42 UTC:

| Branch | Native State | Observation |
| --- | --- | --- |
| P25 DropMoE | FAILED/exit1 | CUDA out of memory in first loss forward |
| P26 SOT | FAILED/exit1 | CUDA out of memory in first loss forward |

P25 task `measure_p1b_dropmoe_b640_r2_p25_20261102` ran for
232.830049 seconds, including actual CPU state reconstruction. The failure
occurred in the attention matrix allocation for counterfactual correction,
before completing the first measured update. The error reports a requested
4.24 GiB allocation, 12.08 GiB already allocated, 16.60 GiB reserved and
3.24 GiB free. These are error-time values, not successful measured peaks.
The message's logical GPU0 corresponds to bound physical GPU3.

There is no successful B640 measurement report or checkpoint on P25.
No automatic retry, smaller-batch fallback, allocator configuration change,
collector restart or cache deletion was performed. Native status and exact
log bytes are archived in
`reports/p1b_capacity_r2_20261102/dropmoe/runtime/`.

P26 task `measure_p1b_sot_b640_r2_p26_20261102` also failed in the first
counterfactual correction attention forward, after343.149362 native seconds.
It requested4.24GiB with12.08GiB allocated,16.60GiB reserved and2.93GiB
free. No successful update, measurement report or checkpoint was produced.
Its exact terminal native status and log bytes are archived in
`reports/p1b_capacity_r2_20261102/sot/runtime/`. Do not rerun either task.

All eight original collector wrappers and actual children were live at
02:42 UTC. P25 had382/882 completed sequences and P26 had397/882.
Every P25 card retained its250W limit. These are collection progress counts,
not completed policy training or full benchmark metrics.

The earlier successful B128 measurement remains valid evidence for that
batch only. The unsuccessful B640 run does not authorize a new batch or
formal batch/LR recipe. Full collection acceptance and exact five-epoch FIT
admission are still outstanding.
