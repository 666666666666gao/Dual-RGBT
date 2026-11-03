# Actual P1A Training Time and First P1B Shard

Read-only audit of original P25 native logs and steps.csv, plus the completed
DropMoE TRAIN shard. No models were loaded, no GPU work was launched, and no
TEST annotations or scores were used.

| Branch | Actual Training Host | Native Total | Mean Epoch | Epoch 1 TRAIN Loss | Epoch 20 TRAIN Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| DropMoE | P25 GPU0 | 5.3948 hours | 16.1723 minutes | 3.071410 | 2.424090 |
| SOT | P25 GPU1 | 4.7154 hours | 14.1331 minutes | 3.052869 | 2.428864 |

Both original native tasks are COMPLETE/exit0. All 20 epochs have 2,500
updates and ten modality/action routes with 250 updates each. All logged
losses are finite. Epoch time is the original EPOCH_COMPLETE wall timer,
including loading, training and checkpoint writes. Native total also includes
startup. SOT artifacts are now on P26, but its training ran on P25.
These are P1A measurements, not a P1B throughput estimate. Falling TRAIN
loss does not prove tracking convergence or benchmark improvement.

P25 DropMoE collector s0 is COMPLETE/exit0. Its plan hash matches the receipt,
config equals plan, and native command equals plan command. Receipt records
221 sequences, 98,613 tracked frames and 19,574 records. All 221 cache files
were read and their SHA256 values match the receipt. This is a completed
shard byte audit, not the four-shard canonical full collection acceptance.
Do not start FIT until all four shards pass that full verifier.

Exact native records, log/CSV hashes, all per-epoch means/times, receipt and
cache hash map are in
`reports/p1a_train_timing_first_shard_r1_20261102/audit.json`.
Audit source: `work/audit_p1a_timing_and_first_shard.py`.
No benchmark target, full weighted MACs, FPS or convergence claim is made.
