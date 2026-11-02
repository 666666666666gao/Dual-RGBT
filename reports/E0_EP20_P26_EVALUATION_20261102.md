# Completed E0 Ep20 Fine-Tuning And Host26 Evaluation

The user's requested E0 fine-tune completed on25 GPU1, independently of
the stalled GPU0 CADTrack training. Native task
train_e0_executor_ep20_r1_20261101 is COMPLETE/exit0 at
2026-11-01 16:38:45 UTC. Forty epochs contain100000optimizer updates,
2500updates and20000sampled pairs each. The ten modality/action routes
each receive250updates per epoch. The original E0 policy stays frozen.
The executor LR starts1e-5 and backbone1e-6, with0.1decay after10/30.

Actual summed epoch time is12.628316hours, mean18.942475minutes/epoch.
Finalfive mean TRAIN losses are2.141702,2.108977,2.115855,2.109977,
2.111386. These different seeded-pair TRAIN losses do not establish
tracking accuracy or convergence. Raw full per-epoch/per-route results:
E0_EP20_TRAIN_COMPLETE_P25_20261102.json.

## Preserved Weights

Originals remain at25:
/data2/gaob/Dual-RGBT/e0_executor_ep20_lasher_r1_20261101.
Destination on26:
/data2/gb/Dual-RGBT/checkpoints/e0_ep20_p25_completed_20261102.
All retained epochs036-040 are evaluation/audit dependencies.

Epoch040 actual transferred SHA256 is
eb63d329d382df1e8fc5bc69e73c41a11d1c263bd4ad7f4a8af7db1107b3026f.
Actual26 CPU strict load passed:265state tensors,103374215parameters,
all finite; all15frozen policy tensors match the original parent exactly.
No CUDA initialized, training or benchmark inference was performed by
that check. Receipt:E0_EP20_E040_P26_CPU_STRICT_LOAD_20261102.json.

The ep0014 factory on26 constructs the architecture; the complete
ep20-finetuned E0 checkpoint overwrites its full net with strict=True.
The factory is not the evaluated learned state. No raw CADTrack checkpoint
is substituted for the requested E0 model.

## Evaluation Schedule

Draft:e0_ep20_p26_evaluation_r1_20261102/draft.json, SHA256
bb65d9900591f86b7474cb0318ca9d37ce68e7794601178d5234f88ee0b79316.
Actual host/input audit is stored beside it. The draft binds49sources,
6inputs and15full evaluations,5epochs times3benchmarks.
Fixed order is40,39,38,37,36; each epoch lists LasHeR first, then
RGBT210 andRGBT234. Epoch40 is first by schedule, not TEST selection.
Keep original E0 tracker and cost_weight0.05. No M1/M2/M3 is attached.
Only26GPU1 is available for this queue, at most3concurrent tasks.
GPU0's running R15 LasHeR inference is preserved.

The original E0 evaluator/scorer are reused without changes. Two entries
are absent on26 and require minimal deployment after actual-host
supplemental source review. Current26 dependencies stay unchanged.
Five dependencies differ from the old25 review and are explicitly listed
in host_audit.json, so the old25 report is not treated as26 admission.

Before full inference, require exact review/draft/source/input/weight
binding, actual TRAIN11frame smoke COMPLETE/exit0 and zero updates with
unchanged tensors. Full runs must contain245/210/234sequences and
220703/104706/116650frames respectively. Run the official scorer only
once after each complete native receipt. Report every result descriptively;
TEST is exploratory/test-ineligible and does not select a checkpoint,
seed, threshold or training recipe.

The evaluator records full-run decoded timing and parameter count.
Actual matching-source CPU route MAC measurements plus full-run trace
route counts supply MACs; timing under concurrent load is not isolated
throughput. No numerical benchmark, FPS or dataset-weighted MAC claim
is made before the corresponding evidence is complete.

## Runtime Admission And Launch

Actual26 source review:E0_EP20_P26_EVALUATION_SOURCE_REVIEW_R1_20261102.json,
SOURCE_PASS/0, SHA256
da13efb9bb4d7db1052d3c2a25d893316294b2ae4fc68c385d0a932592d1a458.
Attribution remains UNATTESTED/same-family/provisional, not independent
scientific acceptance. Both unchanged entry files were deployed; all49
actual source and6input hashes passed remote launch checks.
All5weights plus config/COMPLETE were transferred and individually
rehashed against25. Receipt:E0_EP20_FINAL_FIVE_TRANSFER_P26_20261102.json.
All original25weights remain preserved.

The draft remains launch_allowed=false. Final plan differs only by
launch_allowed=true, SHA256
2ca9951e0a1e8512eccd2dfd920621f8d90e1de34a962bb19112a78b4fa93f4f.
The actual startup integration checks review host/root/stage/draft,
source/input hashes, GPU1 capacity and successful smoke before full runs.
These controls are not assumed to be enforced by the generic evaluator.

TRAIN smoke task:smoke_e0_ep20_e040_r1_p26_20261102, wrapper718638.
COMPLETE/exit0 after30.803seconds including loading;11frames/10tracked,
zero optimizer/backward calls, model unchanged. Predictions/trace/native/
config/SMOKE receipt are archived in
e0_ep20_p26_evaluation_r1_20261102/smoke and smoke_native.json.
Receipt SHA256:
ae96b63c89e9c8d3831ecb6e895e7b2bd753e05480f8e791f8454d191312dccc.

After verifying the actual smoke, full epoch40 tasks started
2026-11-02 00:57:06-07UTC onGPU1:

| Benchmark | Native Task | Wrapper |
| --- | --- | ---: |
| LasHeR | eval_e0_ep20_e040_lasher_r1_p26_20261102 | 726621 |
| RGBT210 | eval_e0_ep20_e040_rgbt210_r1_p26_20261102 | 726635 |
| RGBT234 | eval_e0_ep20_e040_rgbt234_r1_p26_20261102 | 726639 |

Full launch records and exact commands:
e0_ep20_p26_evaluation_r1_20261102/epoch040_launch.json.
The remaining12jobs are scheduled, not claimed to be running or complete.
No new25task or26GPU2/3workload was launched.

At01:07UTC a direct process audit verified all three wrappers and actual
children726624/726636/726640 live. The subsequent progress read showed
6/28/18completedsequences for LasHeR/RGBT210/RGBT234, with fresh traces.
GPU1 uses6814MiB, utilization51%. Existing GPU0 R15 LasHeR child3795734
is live and has completed145/245sequences. No full benchmark score exists
yet for the newly fine-tuned E0. The existing five-minute heartbeat now
contains the admitted queue, exact bindings and sole-scoring instructions.

## CPU Arithmetic Profile

Actual final epoch40 CPU profile passed all10route cross-checks, with
CUDA uninitialized and model unchanged. Profile:
E0_EP20_E040_P26_ROUTE_MACS_20261102.json, SHA256
a9823f52ae922e207deac081654c662cfb18aaeb546750217d11bea3896daf5d.
The existing profiler was deployed as a separate CPU-only tool;
it does not alter any evaluated model or tracking source.

| Action In Either Modality | GMAC Per Tracked Frame |
| --- | ---: |
| STOP | 40.042317 |
| PRIMARY / SECONDARY / PREVIEW | 61.608701 |
| EXPAND | 80.074911 |

These are fixed-shape convolution/dense/attention arithmetic counts,
independently matched by module formulas. They exclude initialization,
nonlinear/normalization/ROI/decode/memory/bookkeeping operations.
They are not full-dataset weighted MACs or isolated FPS; full-run route
counts and decoded timings are still required for the final report.
