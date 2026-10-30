# Shared GPU evaluation

The user requested concurrent evaluation on one card to keep whole cards
available for actual large-batch training.

## Measured Capacity

Server25 has four 24 GiB GPUs and 64 logical CPUs. Existing evaluation workers
use about 2258 MiB device memory and 5.3 GiB resident host memory. At the
10:25 UTC snapshot, available host memory was about 225 GiB.

At 10:54 UTC, two existing size checkpoints were evaluated concurrently on GPU3
alongside its running TRAIN-state collector. Both 11-frame native tasks exited
0, kept all 265 model state tensors frozen, and reproduced their original saved
predictions and integer routes. The measured card peak was 6785 MiB. These
short probes establish concurrent execution and consistency, not long-running
throughput improvement or a new full-dev result.

Evidence: `reports/SHARED_EVAL_PROBE_LAUNCH_20261030.json`,
`reports/SHARED_EVAL_PROBE_MEMORY_20261030.json`, and
`reports/SHARED_EVAL_PROBE_VERIFIED_20261030.json`.

## Schedule

The two registered size chains keep their original TRAIN GPUs0/1. Subsequent
complete97dev evaluations use GPU3, sharing with its existing collector until
that collector completes. GPU2 retains its registered collection queue.

With two scheduling chains, admission reserves space for both simultaneous
launches: at least 6144 MiB free device memory, fewer than three current compute
processes, and at least 16 GiB available host memory. Two simultaneous admissions
can bring the observed process count to at most four. This scope is specifically
the two named size chains, not a general-purpose four-worker cluster scheduler.

`tools/rehome_size_tail_eval.py` checks the exact old CPU controller command,
current state identity and native child. The new controller validates its review,
original plan and live child before acknowledging readiness. Only the old CPU
controller PID receives SIGTERM. The GPU wrapper and its model process continue.
The replacement acquires the same controller lock and adopts the existing child;
it neither restarts nor duplicates that epoch.
For an adopted wrapper, it waits for process exit before reading the native
terminal JSON, because the historical wrapper writes that file non-atomically.
Newly launched wrappers are likewise read only after their Popen has exited.

The original training plan bytes, resume hash, checkpoint format, TRAIN recipe,
full-dev scorer, stopping criteria and last-five retention stay bound. A separate
handoff records the scheduling change, with the old and new controller identities
and the subsequent evaluation GPU. Polling remains 180 seconds.

## Interpretation

Report throughput from actual complete evaluations after shared scheduling.
Do not infer throughput from VRAM usage or short smoke FPS. Dev remains exclusively
for tuning and selection; final test eligibility and protected E0 are unchanged.
