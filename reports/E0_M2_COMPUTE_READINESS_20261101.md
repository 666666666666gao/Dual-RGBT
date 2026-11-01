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
