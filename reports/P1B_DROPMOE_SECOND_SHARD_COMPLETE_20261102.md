# DropMoE Second TRAIN Shard Complete

The original P25 task `collect_p1b_dropmoe_full_s1_r1_p25_20261102`
finished natively with COMPLETE/exit0. The terminal receipt contains
221 sequences, 103,761 tracked frames and 20,578 stride-five TRAIN records.
These totals and the exact ordered sequence names match its TRAIN manifest
shard. The original native command, GPU, hostname, plan digest, saved config
and TF32-off precision record match.

All 221 cache files were read and their SHA256 values matched the terminal
receipt. Exact raw plan, native status, log, receipt, config, progress and
precision bytes are archived under
`reports/p1b_completed_shards_r1_20261102/dropmoe/s1/`.
`audit.json` binds the archived files by SHA256 and records cache sizes/hashes.
The read-only audit source is `work/archive_p1b_completed_shard.py`.
No cache tensor deserialization, GPU use, inference or training occurred.

Together with the already audited s0, completed DropMoE shards account for
442 sequences, 202,374 tracked frames and 40,152 records. This is not the
canonical four-shard full collection acceptance. Original s2/s3 remain live;
only after all four finish can the full verifier inspect all 882 sequences
and their tensor records, followed by a separately bound scaled FIT plan.
P1B trains the existing preview/quality/gain modules with a frozen executor;
this completion is not an M1/M2 training result.

Six fixed epoch20 controls remain live. At the separate evaluator inspection,
DropMoE completed 58/110/99 sequences for LasHeR/RGBT210/RGBT234; SOT completed
34/74/92. All actual evaluator children were live and trace ages below nine
seconds. These are progress counts, not partial accuracy metrics. No full
official score or weighted full-run compute result was available.
