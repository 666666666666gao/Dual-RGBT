# E0 Route Compute Findings

## First Complete Full-Run Aggregation

The two RGBT210 runs completed and were officially scored once on2026-11-01.
The summarizer then passed both complete runs,104706 frames each,
including104496 tracked frames. Control/guard mean GMAC is
64.023387996/64.034767531; total parameters103374215/103491722;
concurrent-load FPS11.134969/11.137673. Source bindings, raw trace coverage,
timing and score hashes passed. This is the first actual full-run positive
witness, superseding the earlier pending-execution note below.

Raw reports are in `reports/e0_m1_full_scores_r1_20261101/`; interpretation,
accuracy, denominators and exact hashes are recorded in
`reports/E0_M1_RGBT210_FULL_FINDINGS_20261101.md`.
This does not complete the remaining benchmarks or trained E0 evaluations.

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

## Training-Root Provenance Check

Comparing the first profile's 127-source collector map to the ep20 evaluation
map found different hashes for `lib/config/cadtrack/config.py`,
`lib/models/cadtrack/cadtrack.py` and `scripts/joint_evaluation_contract.py`.
The first profile matches all30 shared paths of the live M1 evaluation map,
but must not be reused blindly for ep20 evaluation.

The profiler now supports an explicit `--e0-only` measurement for the live
training root, whose package lacks the observer required by the optional
identity-head import. This selects the requested measurement scope; it does
not substitute another implementation on failure. The original profiler was
retained byte-for-byte under
`reports/source_snapshots/e0_cpu_profile_r1_20261101/profile_e0_route_macs.py`,
SHA256 `7301ef5efc32a65d94ad3d72950ac8d538a518a6509141c03c990bcc2015df8d`.
Its original remote mirror copy and raw report remain unchanged.

A second actual CPU execution in `/home/gaob/Dual-RGBT` used the ep20 final
review's49 source bindings. It independently measured all ten E0 routes, with
the same MAC and parameter results as the table above. Frozen state remained
equal and CUDA was never initialized. Raw evidence:
`reports/E0_FINETUNED_ROUTE_MAC_CPU_CROSSCHECK_P25_20261101.json`,
SHA256 `708d3aaafb486a5f21a28bf134207a3ae2bf59e0c61607336e9a6addd4a8989e`.
Actual source binding passed against the completed real ep20 TRAIN-smoke config
on25 (49 common paths), and against both actual M1 full-run configs on26
using the first report (30 common paths each).

## Full-Run Aggregation

New `tools/summarize_e0_full_compute.py` consumes an existing official full
score, its exact hash, a matching profile and all raw run artifacts. It requires
native COMPLETE/exit0, full sequence/frame coverage, no optimizer updates,
all consecutive trace frames, route/token consistency and matching source
hashes. It writes a separate result and never invokes the official scorer.

The mean MAC field covers tracked frames only, weighted by their actual
modality/action counts. It explicitly excludes initialization arithmetic.
M1 guard adds117376 head MACs per tracked frame. Reported FPS is copied and
recomputed from the full-run timing array, which includes initialization.
These different denominators are explicit; no isolated-throughput claim is
made from concurrent evaluation.

Use the first profile for M1 runs on26 and the second for fine-tuned E0 runs
on25. After official scoring:

```text
CPU_SCORING_PYTHON tools/summarize_e0_full_compute.py
  --root ABS_PROJECT_ROOT --run ABS_RUN
  --score ABS_OFFICIAL_SCORE --score-sha256 EXACT_SCORE_SHA256
  --profile ABS_MATCHING_PROFILE --profile-sha256 EXACT_PROFILE_SHA256
  --output NEW_COMPUTE_REPORT
```

Three focused tests passed locally and in both remote CPU-scoring environments:
complete trace arithmetic, rejection of missing frames/wrong token routes,
and rejection of the observed CADTrack source drift. Tests are synthetic
engineering checks, not benchmark evidence. A completed full-run aggregation
has not yet executed because all six M1 evaluations are still running.
No existing scorer, model or running evaluation source changed.
