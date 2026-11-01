# R15 M2 compute reporting readiness

Observed on host 2026 at approximately 2026-11-01 21:20 UTC.
This is a source/configuration audit, not a complete benchmark result.

## Verified runtime configuration

All three R15 full TEST configurations record 103,490,697 total parameters:
103,374,215 E0 parameters plus 116,482 M2 parameters. M1 is not included in
this E0+M2 variant. The configuration's timing scope includes image decoding,
E0, M2 candidate generation and initialization, and excludes model loading.
The full scorer computes FPS, mean latency and P95 from frame_seconds.npy
after native COMPLETE/exit0 and full artifact validation.

## Missing full M2 MAC accounting

tools/score_e0_m2.py currently does not copy total_parameters into its score
JSON and does not compute MACs. The parameter count remains available in the
hash-bound config.json. tools/summarize_e0_full_compute.py accepts only E0
fine-tuning or M1 guard stages and their different receipt/score schemas;
it cannot be applied unchanged to R15.

dual_rgbt/e0_m2_online.py calls execute_candidates on scheduled frames.
scripts/collect_e0_identity_train.py executes 12 candidate crop queries, each
calling model.primary for both modalities: 24 primary forwards per invocation,
followed by the identity head. These extra forwards must be counted even when
the identity head selects no match. Initialization also computes a two-modality
identity anchor once per sequence. Counting only the M2 head understates cost.

Existing CPU profile
reports/E0_ROUTE_MAC_CPU_CROSSCHECK_P25_20261101.json has SHA256
58d4d837e5f2925c33ed469ff7d6523922cfce253f91ed968a9a73b96f44f1e0.
Its measured anchor cost is 71,316,836,352 MACs per initialization and its
24-candidate identity head cost is 5,704,704 MACs per invocation. These are
component measurements, not full R15 per-frame MACs.

The profile and each remote R15 config share 127 source paths; the only
different hash is scripts/run_task_four_gpu.py. Thus the full source maps
are not identical. Any future reuse must explicitly bind the relevant
model/operator dependencies and document the wrapper difference. Do not
relax the existing summarizer's assertions or present the old profile as
an exact full-map match.

## Next required evidence

After full native completion, run the official scorer once per dataset.
Preserve its report and reference config.json for total parameters. A separate
M2 compute summary must validate the actual plan/native/receipt/artifact schema,
bind measured primary-forward costs to the executed sources, count complete
trace routes and M2 invocations, and include candidate-generation and anchor
costs with explicit initialization and operator scope. No extra TEST inference
is required for trace aggregation. FPS under concurrent load must be labeled
as such. No inference source, reviewed plan or active run was modified by
this audit. Complete accuracy, FPS and full M2 MAC results remain pending.

## Candidate primary measurement completed

tools/profile_e0_m2_candidate_macs.py was added separately from the R15 source
map. Its SHA256 is
55e4c3220fa8faf12250819b035a4369e41de27d17b264dc2334b05c4e846b5e.
In-memory syntax compilation passed. The human-authorized existing review chat
returned SOURCE_PASS/0 for this exact script in
reports/E0_M2_CANDIDATE_MAC_SOURCE_REVIEW_R1_20261101.json, SHA256
2d5ed1c8b2092bfb29f77e14ea7ae459992c707a9fcd55b1f74d581919864677.
This is same-family provisional engineering review, not scientific acceptance.

The remote finalized LasHeR plan SHA256 is
004ed6f3b623b95e95ab075602e80bef1a087c2c9e6c5d9cccfc271984d1ad6f.
At preparation time both the new script and intended output were absent on
host26. After review, only the added script and its report were deployed and
their remote hashes verified. The following CPU-only command then completed
with exit0:

```bash
CUDA_VISIBLE_DEVICES='' /data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python \
  tools/profile_e0_m2_candidate_macs.py \
  --root /data2/gb/Dual-RGBT \
  --plan reports/e0_m2_online_r15_20261101/lasher_plan.json \
  --plan-sha256 004ed6f3b623b95e95ab075602e80bef1a087c2c9e6c5d9cccfc271984d1ad6f \
  --output reports/E0_M2_CANDIDATE_PRIMARY_MAC_CPU_R1_20261101.json
```

The output was fetched byte-exactly. Raw report
reports/E0_M2_CANDIDATE_PRIMARY_MAC_CPU_R1_20261101.json has SHA256
6bd236f196ce4dc96e064278101460ae8d6088ab53dcefcf8e1bd578ef1bc8d7.
It records CPU / PyTorch1.12.1+cu113, model_unchanged=true and
cuda_initialized=false. Both modalities individually match independent
Conv2d/Linear/attention formulas:

| Component | MACs |
| --- | ---: |
| One RGB primary forward | 40,032,593,920 |
| One TIR primary forward | 40,032,593,920 |
| Twelve queries, both modalities (24 forwards) | 960,782,254,080 |

Only two primary forwards were measured on synthetic fixed-shape inputs;
12 times their sum is the source-verified fixed-query arithmetic count.
This 960.782254080 GMAC per candidate-generation invocation excludes the
identity head, anchor, E0 routes, crop/ROI work and other operations listed
in the raw report. It must not be reported as the full per-frame or full-run
MAC cost. Complete trace-weighted M2 accounting and benchmark FPS/accuracy
remain pending. Do not rerun into the existing output or change active R15
source hashes to add measurements.
