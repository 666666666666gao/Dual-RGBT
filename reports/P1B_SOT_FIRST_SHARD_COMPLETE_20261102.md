# First SOT TRAIN Shard Complete

The original P26 task `collect_p1b_sot_full_s0_r1_p26_20261102`
finished natively with COMPLETE/exit0. The archived audit was observed at
2026-11-03 04:43:30 UTC and records 221 sequences, 98,613 tracked frames
and 19,574 stride-five TRAIN records.

The ordered sequence names and counts match the exact TRAIN manifest shard.
Native command, GPU, hostname, plan digest, saved configuration and TF32-off
precision bindings passed. All 221 cache SHA256 values matched the terminal
receipt. Raw plan, native status, log, receipt, config, progress and precision
bytes are archived in `reports/p1b_completed_shards_r1_20261102/sot/s0/`;
`audit.json` binds the archived files and cache sizes/hashes.
The read-only helper was `work/archive_p1b_completed_shard.py`.
This audit did not deserialize tensors, launch inference or use a GPU.

At 04:47 UTC, original P26 s1/s2/s3 children were live at 203/176/172 completed
sequences, making 772/882 sequences across all shards. GPU0 was running the
fixed LasHeR control, GPU1 and GPU2 collection plus RGBT210/RGBT234 controls,
and GPU3 collection. All four GPUs had actual compute processes.

P25 s0/s1 were already COMPLETE/exit0; s2/s3 remained live at 206/218,
making 866/882 sequences. The s3 remaining manifest includes a 12,782-frame
sequence, followed by a 90-frame sequence. Progress is saved after whole
sequences, so an unchanged count alone does not demonstrate failure.
No restart or GPU reset was performed.

The shard audit is not canonical four-shard collection acceptance.
After all four shards finish for a branch, run the existing full verifier,
then bind the accepted collection into a separately reviewed B512,
four-worker, five-epoch scaled FIT plan before actual training.
P1B refreshes preview/quality/gain with a frozen executor; it is not
M1/M2 training. Full TEST controls remain exploratory and test-ineligible.
