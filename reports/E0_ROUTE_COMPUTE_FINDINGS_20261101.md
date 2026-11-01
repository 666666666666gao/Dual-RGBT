# E0 Route Compute Findings

## Executed Measurement

The standalone CPU profiler executed all ten modality/action combinations of
the current `DualRGBT.forward_action` architecture. The checkpoint was the
actual ep20 fine-tuning TRAIN smoke weight, not a completed formal-training
weight. Every convolution, dense projection and attention product MAC count
matched a second calculation using executed module shapes plus the functional
auxiliary patch projection.

Raw evidence:
`reports/E0_ROUTE_MAC_CPU_CROSSCHECK_P25_20261101.json`,
SHA256 `58d4d837e5f2925c33ed469ff7d6523922cfce253f91ed968a9a73b96f44f1e0`.
Runtime was host25 `/data2/gb/Dual-RGBT`, Python3.7/PyTorch1.12.1+cu113,
one CPU thread, CUDA hidden and never initialized. No training process was
signalled, paused or changed. Model tensors remained bit-equal after execution.

| Executed Path | Primary Tokens | Auxiliary Tokens | GMACs Per Frame |
| --- | ---: | ---: | ---: |
| STOP | 385 | 0 | 40.042316672 |
| PRIMARY | 385 | 193 | 61.608700800 |
| SECONDARY | 385 | 193 | 61.608700800 |
| PREVIEW | 385 | 193 | 61.608700800 |
| EXPAND | 385 | 385 | 80.074910592 |

RGB-primary and TIR-primary counts agree for every action. Batch size was one;
templates were `[1,2,6,128,128]`, search `[1,6,256,256]`. A MAC is a
multiply-accumulate pair, counted as two supported profiler FLOPs. Bias adds,
normalization, nonlinearities, interpolation, ROI sampling, data movement,
image decoding and tracker bookkeeping are excluded. These are architecture
arithmetic counts, not total hardware instructions or measured latency.

## Components

| Component | Parameters | Measured Matrix/Convolution MACs |
| --- | ---: | ---: |
| E0 | 103374215 | Depends on executed action above |
| M1 reliability head | 117507 | 117376 per prediction |
| M2 identity head, one candidate | 116482 | 426112 per set |
| M2 identity head, 24 candidates | 116482 | 5704704 per set |
| M2 initial dual-modal deep anchor | Uses E0 encoder | 71316836352 per initialization |

The real M2 head forward agrees with the analytic expression
`196608 + 229504 * candidate_count`. This confirms only head arithmetic.
Candidate creation, extra search encoding and the observation path are separate
costs; no cheap complete M2 recovery or completed M2/M3 integration is implied.
The M1 head excludes ROI observation, CPU transfer, template copies and guard
bookkeeping. E0's parameter count is the whole instantiated module count,
including registered parameters not necessarily exercised by every route.

## Implications And Pending Evidence

- A local auxiliary action adds 21.566384128 GMAC over STOP; EXPAND adds
  40.032593920 GMAC. Future workload cost must use the observed action histogram
  from each completed full run, not the nominal sparse-token percentage.
- Two template sets plus the class token remain in each auxiliary encoder
  call: 64 search patches still mean 193 total tokens. This explains why
  auxiliary token reduction alone is insufficient to describe end-to-end cost.
- The existing policy's nominal local/full auxiliary token cost is `193/385`,
  about50.13%. Measured extra local/full route MACs are about53.87%.
  The token budget is a proxy rather than exact measured compute. This is a
  future TRAIN-only cost-model hypothesis, not evidence that changing the cost
  will improve accuracy. No cost weight, policy, threshold or running task was
  changed; do not choose a replacement from TEST outcomes.
- E0, M1 and M2 parameters and architecture route MACs are now measured.
  Full-dataset action-weighted MACs and official-score FPS remain pending each
  full run's completion. Existing scorers already record FPS, mean/p95 timing
  and parameter counts. Concurrent evaluation timing is not isolated throughput.
- Formal ep20 training remains live. The measurement is not evidence of
  convergence, tracking accuracy or attainment of any benchmark target.

## Live LasHeR Diagnosis

The two host26 LasHeR jobs remained at 58 completed sequences while processing
`blkboy`shead`, which has 12862 frames. Direct inspection found live children
1629804/1629805 and recent trace writes: control frame10512, guard frame10051,
trace ages about8.1s/0.5s. This is a long sequence, not evidence of an exited
task. Preserve the jobs; inspect again near the predicted sequence boundary.
No partial score or model adjustment was made from these TEST observations.

## Instrumentation Changes

Only `tools/profile_e0_route_macs.py` was added as executable instrumentation.
It loads frozen weights into its own CPU process, installs temporary hooks on
that process's model, removes them after each measurement and writes a new
report. No existing model, tracker, training or evaluation source was edited.
