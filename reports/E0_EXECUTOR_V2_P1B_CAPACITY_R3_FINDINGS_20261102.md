# P1B B512 With Explicit Scaled Learning Rates

Both newly admitted R3 tasks completed with native exit0 on physical GPU3.
They reconstructed512 real TRAIN states, then applied two updates to the
same batch with preview LR4e-5 and quality/gain LR4e-4. Frozen executor
state and finite gradient/state checks passed. No weights were written.

| Measurement | P25 DropMoE | P26 SOT |
| --- | ---: | ---: |
| Batch / updates | 512 / 2 | 512 / 2 |
| Peak allocated | 13.705040 GiB | 13.705040 GiB |
| Peak reserved | 17.257813 GiB | 17.257813 GiB |
| First update | 47.163464 s | 52.768236 s |
| Second update | 45.116627 s | 52.848254 s |
| First / second loss | 1.118773 / 1.099765 | 1.091316 / 1.068094 |
| Native duration including reconstruction | 262.031749 s | 360.698759 s |
| Free GPU memory before CUDA | 21,981 MiB | 21,674 MiB |
| Native state | COMPLETE / exit0 | COMPLETE / exit0 |

Both measurements used AdamW weight decay1e-4 and gradient clipping1.
The explicit learning-rate scale is512/128=4, matching the separate
scaled FIT entry. This verifies capacity and two finite actual updates,
not a five-epoch training run or convergence.

The P25 card was observed at21,698MiB total while the short test and its
existing collector coexisted. Per-process torch peak reserved memory is
not whole-card usage; the collector adds its own allocation.

## Timing Limitation

The timed updates exclude state reconstruction, full-loader iteration,
prefetch and checkpoint I/O, and include concurrent collector contention.
For91,091 records, B512 entails178 updates per epoch and890 for five
epochs. Multiplying178 by the observed second-update time gives roughly
2.23h/epoch on P25 and2.61h/epoch on P26 under this short test's conditions.
These are extrapolations, not measured epoch durations or promises.
Actual loader, artifact verification and changing system load must be
timed during full FIT. Two falling losses cannot establish convergence.

## Provenance

- Exact admission report SOURCE_PASS0:
  `c37e1e1bf1b236ab59aa04ecbe231c5bb87de8dd062a8d4463884bd73bacdbbb`
- Tool deployed under the new filename
  `tools/measure_e0_policy_refresh_capacity_b512_r3.py`:
  `5abc3fba957b7e8e837179c104e1f94ec837001abd114bc4d80e2cf15e346e7e`
- P25 measurement:
  `9eb2b2a20165dea302d23a3bac8871f9e7ef575f90d36de57549116f9190ea4e`
- P26 measurement:
  `fc7e69ca5f15bd7c5d04201aec03b050bdd8746d9c0117fd5d124ec6e70509e9`
- Exact finalized specs, disabled reviewed drafts, measurements and raw
  native status/logs:
  `reports/p1b_capacity_r3_20261102/{dropmoe,sot}/`

`work/verify_capacity_r3.py` checked archived native task/GPU/argv,
review/spec/tool bindings, learning rates, finite updates, telemetry and
the absence of checkpoint or benchmark claims. Both original98-source
and7-input maps were revalidated remotely after measurement.

All eight original collector wrappers and children were live at
02:58UTC: P25 completed471/882 sequences and P26 completed459/882.
Neither branch yet has full collection acceptance or formal FIT admission.
Do not rerun R3. Preserve R1/R2 tools, failed B640 evidence and partial
collector caches. Next actual full collection acceptance precedes the
separate exact B512/scaled-LR five-epoch FIT plan review and launch.

The old E0 MAC profile has a different model-source SHA from P1 and
cannot be copied into a new benchmark report. Rebind and run the exact
current-source CPU profile before aggregating new trace-weighted MACs.
