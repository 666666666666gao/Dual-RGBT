# C2 DropMoE Zero Policy FIT Epoch 4

At 2026-11-04 16:19 UTC, P25 task
`fit_c2_policy_dropmoe_zero_full_b512_r1_p25_20261104` remained native
`RUNNING` (wrapper/child `2096742/2096743`) on GPU1. The child had about
96.5% CPU, 19,459 MiB GPU memory, 100% GPU utilization, 71 C, and 248.94 W
against the configured 250 W limit.

Checkpoint `epoch_004.pth` was written at 16:16:21 UTC. Its size is
415,525,979 bytes and SHA256 is
`ce4d7fcf3885c923ca8aff144656e2dd15994cc4a9a2c5d461107ab1acadcbcf`.
The enabled plan SHA256 is
`8d33d391b0bbb0203a98ebac53f249c8440f2ff6bd5e92d8662cbb7f9d450c74`.
The configured trainer saves a checkpoint only after that epoch's finite
state checks, frozen-executor equality checks, changed-policy checks and
strict reload/prediction-equivalence check. This is an epoch-4 checkpoint
witness, not final FIT completion or benchmark evidence.

The latest `steps.csv` flush was epoch 4 / step 700. At 16:19 UTC the native
wrapper and child were still alive and GPU1 utilization was 100%; no
`COMPLETE.json` existed. Do not infer that epoch 5 is complete from the
checkpoint filename or the CSV flush. The runbook requires five epochs,
890 updates, native `COMPLETE`/exit0 and the bound epoch005 checkpoint
before C2 tracker evaluation.

At the same observation, P25 FiLM replay R3 (`2228313/2228314`) and P26 SOT
zero R6 (`2322989/2322990`) and FiLM R1 (`2535697/2535698`) replay
verifiers were still running with active children. No final acceptance
receipts were present. P25 GPUs0/3 and P26 GPUs0/1 were idle, but no
follow-on task was admitted solely to occupy them. Preserve the active tasks
and use only their separately reviewed plans and fresh runtime gates after
real terminal receipts.

This FIT remains TRAIN-only. It does not establish convergence, tracking
accuracy, a selected checkpoint or goal completion. TEST/dev remain
disabled for selection. Existing targets remain unmet.
