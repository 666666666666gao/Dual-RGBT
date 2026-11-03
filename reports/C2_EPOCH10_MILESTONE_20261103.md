# C2 Epoch 10 Milestone

At 2026-11-03 09:51:45 UTC, both P25 C2 training variants completed epoch 10
and remained live in epoch 13. Each completed 2,500 updates with 250 updates
on each of the ten modality/action routes. This is the configured scheduler
milestone epoch `[10]`; it is training progress, not benchmark evidence or a
convergence claim.

| Variant | Epoch seconds | Mean TRAIN loss | Live flushed updates |
| --- | ---: | ---: | ---: |
| Zero | 858.3402851100 | 2.5984220953 | 30,925 |
| FiLM | 857.4380025600 | 2.5938416050 | 30,975 |

Raw native witnesses, logs, CSVs, plans and hashes are preserved at
`reports/c2_training_full_r1_20261103/dropmoe/{zero,film}/epoch_010_witness/`.
The read-only archiver verified exact contiguous updates and all ten route
counts. Both native tasks and trainer children were live when the witnesses
were captured.

P26 zero and FiLM were still training epoch 10 at 24,250 flushed updates each
at 09:49:51 UTC. The P25 GPU0 control slot remains free; no new task was
launched because no additional reviewed job was ready.
