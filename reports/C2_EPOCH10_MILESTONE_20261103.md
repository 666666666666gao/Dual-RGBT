# C2 Epoch 10 Milestone

Both hosts' C2 zero and FiLM variants completed epoch 10 with 2,500 updates
and 250 updates on each of the ten modality/action routes. P25 witnesses were
captured at 09:51:45 UTC; P26 witnesses at 09:58:27 UTC. This is the
configured scheduler milestone epoch `[10]`; it is training progress, not
benchmark evidence or a convergence claim.

| Variant | Epoch seconds | Mean TRAIN loss | Live flushed updates |
| --- | ---: | ---: | ---: |
| Zero | 858.3402851100 | 2.5984220953 | 30,925 |
| FiLM | 857.4380025600 | 2.5938416050 | 30,975 |

| SOT / P26 zero | 1,074.8594641270 | 2.5874126115 | 25,275 |
| SOT / P26 FiLM | 1,075.0552204780 | 2.5872194023 | 25,275 |

Raw native witnesses, logs, CSVs, plans and hashes are preserved at
`reports/c2_training_full_r1_20261103/{dropmoe,sot}/{zero,film}/epoch_010_witness/`.
The read-only archiver verified exact contiguous updates and all ten route
counts. All four native tasks and trainer children remained live; they had
advanced to epochs 13 (P25) and 11 (P26) by 09:58 UTC.

At the 09:57 UTC P26 check, P1B FIT remained native RUNNING with a live
trainer and four active data-loader workers. Its last flushed CSV row was
epoch 2, step 225, over 14 minutes old. GPU3 had returned to 100% utilization,
but no new update was yet flushed. This is an unresolved long-progress gap,
not a confirmed failure or completion; preserve the task and recheck later.
P25 GPU0 remains free; no additional reviewed job was ready to launch.
