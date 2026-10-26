# RankSTOP inference scaling result audit — 2026-10-26

**Verdict: WARN — qualified result PASS; 0 blocking findings.** All978 deterministic checks pass. The observed best configuration is8 processes/OpenCV1 thread at49.330892 aggregateFPS for this fixed TRAIN workload.

Fresh-context Codex review; same-family/provisional. Requested model gpt-6-astra,effort max; actual model/effort/routing are UNATTESTED. Deterministic arithmetic/hash checks are distinguished from semantic review and from neural replay.

## Verified measurements

Each unprofiled arm performs the same8 TRAIN streams ×128 frame events:8 initializations +1016 tracking calls. The frame numerator includes initialization; no frames are dropped or repeated within an arm. Model total parameter count103,218,887 is corroborated in all34 worker receipts.

| Arm | Processes | OpenCV threads | Measured seconds | Aggregate FPS | Spawn-to-join FPS | GPU0 sampled peak MiB | GPU0 timed mean utilization % |
|---|---:|---:|---:|---:|---:|---:|---:|
| arm0_n1_cv80 | 1 | 80 | 116.056609 | 8.823280 | 7.552486 | 2291 | 10.066667 |
| arm1_n2_cv80 | 2 | 80 | 67.780973 | 15.107485 | 11.760727 | 4557 | 19.491803 |
| arm2_n4_cv80 | 4 | 80 | 39.161284 | 26.148274 | 17.216195 | 9087 | 30.742857 |
| arm3_n8_cv80 | 8 | 80 | 26.631272 | 38.451036 | 19.777179 | 18149 | 49.625000 |
| arm4_n1_cv80 | 1 | 80 | 121.302951 | 8.441674 | 6.915617 | 2291 | 9.254545 |
| arm5_n1_cv1 | 1 | 1 | 101.694309 | 10.069393 | 8.625233 | 2291 | 10.860215 |
| arm6_n2_cv1 | 2 | 1 | 55.228222 | 18.541245 | 14.099442 | 4557 | 23.460000 |
| arm7_n4_cv1 | 4 | 1 | 31.228290 | 32.790780 | 20.227414 | 9087 | 43.928571 |
| arm8_n8_cv1 | 8 | 1 | 20.757784 | 49.330892 | 23.136925 | 18149 | 73.500000 |
| arm9_n1_cv1 | 1 | 1 | 96.466884 | 10.615042 | 9.046853 | 2291 | 10.795455 |

Raw best-arm evidence: `profile_output/rankstop_inference_scaling_server26_20261026/arm8_n8_cv1/COMPLETE.json:56`; independent per-worker/timing/GPU recomputation: `.aris/traces/experiment-audit/2026-10-26_rankstop_inference_scaling_result_review/DETERMINISTIC_RECOMPUTATION.json`. All12 arm summaries match the native log; profile arms are excluded from this table/ranking.

BestN8/CV1 worker block rates are6.1664–7.4589FPS. SerialN1/CV1 blocks are10.0694 and10.6150FPS. Therefore49.3309 is total throughput across8 workers; it is not single-stream49.3309FPS. No per-frame service mean/P95 is available. Spawn-to-join best-arm rate is23.1369FPS; this is not whole-command latency.

## Integrity checks A–F

### A ground truth provenance — PASS

Initialization uses the installed LasHeR TRAIN init.txt first box through the hash-bound TRAIN manifest. All8 installed initial boxes equal the manifest and stored prediction[0]; 882 TRAIN and97 dev names are disjoint for the8 selected streams. Only boxes[0] is used on the profiling path. Model-output equality is an explicitly scoped cross-setting consistency check, not a substitute for dataset GT accuracy. No PR/NPR/SR or accuracy claim is generated.

Evidence: `scripts/profile_rankstop_scaling.py:34`, `scripts/profile_rankstop_scaling.py:190`, `dual_rgbt/data.py:15`, `.aris/traces/experiment-audit/2026-10-26_rankstop_inference_scaling_result_review/REMOTE_READONLY_PROVENANCE.json:195`.

### B score normalization — PASS

All12 total-frame counts, earliest-start/latest-finish timestamps, durations, aggregate FPS, sampled GPU0 peak memory and timed-sample mean utilization recompute exactly. Rate numerator is actual stored frame count; denominator is measured elapsed time, never model-output statistics. End-to-end FPS arithmetic recomputes from the stored spawn-to-join duration, but the raw spawn/join endpoints were not saved separately. Any speedup is a ratio of runtime rates and must retain its baseline.

Evidence: `scripts/profile_rankstop_scaling.py:146`, `scripts/profile_rankstop_scaling.py:153`, `profile_output/rankstop_inference_scaling_server26_20261026/arm8_n8_cv1/COMPLETE.json:56`, `.aris/traces/experiment-audit/2026-10-26_rankstop_inference_scaling_result_review/DETERMINISTIC_RECOMPUTATION.json:4430`.

### C result file existence — PASS

Top-level COMPLETE,10 unprofiled arm COMPLETE files,2 cProfile COMPLETE files,34 worker JSON files and2 .prof files exist. Parent command/task/GPU0/PID identity binds startup,config,node and native COMPLETE exit0; native wrapper702029 owns profiler parent702037 in the archived startup witness. Every worker PID matches admissions and report ownership and has native join exit0. All12 log summaries and the final summary agree. All54 archived remote artifact hashes match local copies. This profile exit0 does not repair the original training run missing native-exit receipt.

Evidence: `runs/pipeline_tasks/rankstop_inference_scaling_server26_20261026/status.json:16`, `scripts/run_task.py:23`, `scripts/profile_rankstop_scaling.py:131`, `scripts/profile_rankstop_scaling.py:213`, `profile_output/rankstop_inference_scaling_server26_20261026/COMPLETE.json:15761`, `.aris/traces/experiment-audit/2026-10-26_rankstop_inference_scaling_result_review/DETERMINISTIC_RECOMPUTATION.json:6208`.

### D dead code and execution — PASS

The executed worker path calls trajectory and final-state construction, and the parent calculates/checks the stored summaries. Both saved cProfile files independently reproduce their top40 entries; each records1 initialize,127 track,128 read_pair,256 imread and257 CUDA synchronize calls. Separate profile arms are excluded from best-arm argmax. All12 saved trajectory sets equal their serial reference, including the profiled stream which the runner itself did not compare. No unused scoring helper is presented as an executed accuracy result.

Evidence: `scripts/profile_rankstop_scaling.py:87`, `scripts/profile_rankstop_scaling.py:93`, `scripts/profile_rankstop_scaling.py:210`, `scripts/profile_rankstop_scaling.py:217`, `scripts/profile_rankstop_scaling.py:228`, `runs/pipeline_tasks/rankstop_inference_scaling_server26_20261026/run.log:365`, `.aris/traces/experiment-audit/2026-10-26_rankstop_inference_scaling_result_review/DETERMINISTIC_RECOMPUTATION.json:5873`.

### E scope assessment — WARN

The evidence covers one final checkpoint,seed2027,cost0.05,N1/2/4/8,OpenCV80/1,on GPU0 RTX3090,8 selected TRAIN prefixes of128 frames. Each unprofiled arm has8 initializations plus1016 track calls, not1024 track calls. Only19/1016 calls request auxiliary encoding (17 local-main,2 local-alternate); global-local/global-wide branches are not exercised. N2/4/8 have one run per setting; N1 repeats drift -4.324991% atCV80 and+5.418880% atCV1. Fixed ordering, warm filesystem pages and a continuing GPU1 workload limit causal attribution/generalization. Supports a qualified observed-throughput selection, not full97dev/test FPS, single-stream49.33FPS, universal optimality, all-branch equivalence or accuracy improvement.

Evidence: `reports/RANKSTOP_INFERENCE_SCALING_PLAN_20261026.json:90`, `reports/RANKSTOP_INFERENCE_SCALING_PLAN_20261026.json:103`, `scripts/profile_rankstop_scaling.py:197`, `scripts/profile_rankstop_scaling.py:233`, `.aris/traces/experiment-audit/2026-10-26_rankstop_inference_scaling_result_review/DETERMINISTIC_RECOMPUTATION.json:6222`.

### F evaluation type — PASS

The checklist nearest class for the no-accuracy runtime measurement is self_supervised_proxy: this is an empirical system profile with no task score. The serial-output parity subtest is synthetic_proxy consistency; it compares model outputs to model outputs and makes no accuracy inference. Initialization provenance alone is real_gt from installed TRAIN annotations. These labels do not classify the throughput values as synthetic timing.

Evidence: `scripts/profile_rankstop_scaling.py:34`, `scripts/profile_rankstop_scaling.py:213`, `scripts/profile_rankstop_scaling.py:233`.

## Provenance, exit status and equality

Native wrapper702029 recorded COMPLETE/exit0 for the exact profiler command onGPU0, from2026-10-26T13:54:18.107318Z to14:10:25.379676Z (967.272358s). Archived startup identifies profiler parent702037; all34 spawned worker IDs align with their readiness and report records and all34 joins exit0. No current process query was needed. Source ownership logic is at `scripts/run_task.py:23` and `scripts/profile_rankstop_scaling.py:184`.

All49 listed source hashes match, including the unchanged48 files inherited from the dev plan. All26 bound remote input hashes match; all54 remote archived artifact hashes match local copies. The8 installed TRAIN initial boxes match init.txt,manifest and stored prediction[0], and selected names do not overlap97dev. Checkpoint SHA256: `527a3bc0f56f6a26e31adcc5e656f971b6bd34bbcd220251151d7db1cc228753`.

Every one of10,240 stored main-arm prediction rows and10,160 route rows is exactly equal to the corresponding serial reference, including all8 route fields. All80 main-arm final box/template/history digest records agree per sequence. Both separate128-frame cProfile outputs also agree, adding256 prediction rows,254 route rows and2 digests. This independently checks saved records; it does not regenerate network outputs or recover raw template/history tensors.

The source loads the same final model strictly,sets eval/requires_grad(false),uses no_grad,seed2027,torchthreads1 andTF32false,and asserts the full state_dict unchanged from before warmup through measured execution. No optimizer/backward or precision-reduction path is introduced by this runner. The audit imported no Torch and loaded no checkpoint. These source/runtime witnesses are not independent tensor replay.

## Limitations and reporting requirements

### L1 — Aggregate throughput is not single-stream service FPS

Best observed N8/CV1 is49.33089223000486 aggregate FPS over1024 frame events/20.757783889770508s. Its8 individual128-frame worker blocks are6.166403811712504–7.4588949689563115FPS; these are block averages including initialization/state summaries, not per-frame service latency. N1/CV1 serial block results are10.069393362398067 and10.615041723066662FPS. No per-frame mean/P95 service-time trace was saved by this runner.

### L2 — Timing scopes and unrecorded raw endpoints

aggregate_fps excludes imports,model construction/checkpoint load,5-frame worker warmup and post-timing model-state comparison/JSON output, but includes image decoding,8 initializations,1016 track steps,list construction and per-stream final-state hashing. Spawn-to-join end_to_end_fps includes worker load/warmup/verification/output and is23.136924537815403FPS for bestN8, but excludes parent hash checks,filesystem prewarm,monitor lifecycle and final report writes. Parent began/released/joined timestamps are not persisted separately, so only this field arithmetic/source definition is independently verified. Native whole-command elapsed time is967.272358417511s across all12 arms and setup.

### L3 — Fixed short scope and low auxiliary-query activity

8 TRAIN sequence prefixes,128 frames each,seed2027,cost0.05,one final checkpoint. The1016 track calls consist of997STOP,17local-main,2local-alternate and0global-local/global-wide; every route has8 candidate proposals. This sparse-query workload does not establish scaling for other sequences,costs,full trajectories or auxiliary-heavy workloads. No accuracy metrics were computed.

### L4 — Ordered arms, repeats and shared host background

N1 repeats change -4.324991006796742% forCV80 and+5.418880175107654% forCV1. N2/N4/N8 have one replicate each; arm order is fixed and pages were prewarmed. GPU1 remains at2328MiB in every saved sample and has varying sampled utilization. BestN8 timed GPU1 mean utilization7.888888888888889%,range1–17%; per-arm background summaries are preserved. GPU1 host-resource interference is uncontrolled; no confidence interval or universal optimum is supported.

### L5 — Saved-state evidence is not independent neural replay

Audit compares all saved prediction numbers,8 non-timing route fields and final-state SHA256 strings exactly. The digest covers concatenated bytes of box state,templates,history; raw templates/history were not saved and cannot be rehashed by this audit. Frame=127 and since_query can be reconstructed from the fixed workload/routes; deterministic window and other attributes are source-inspected,not independently tensor-replayed. model_state_bit_equal is a witnessed runtime assertion based on state_dict snapshots before warmup and after measurement. This audit performs0 NN calls,0 checkpoint deserializations and0 profiling reruns.

### L6 — Sampling and CPU-accounting definitions

854 GPU samples have mean intervals about1.10–1.13s rather than exactly1.0s because each query takes time. Device peak memory is a sampled peak over the whole arm,utilization an unweighted arithmetic mean of samples inside the worker measurement interval. BestN8 uses18 timed samples,meanGPU0util73.5%,sampled peak18149MiB. Each worker reports peak allocated546334208B and reserved614465536B,distinct from whole-device memory. cpu_children_seconds includes full reaped worker lifetimes and nvidia-smi subprocess accounting,not steady-inference-only CPU time.

### L7 — Binding and inherited provenance boundary

49 listed source files hash-match locally; original48 bindings equal the frozen dev plan. All26 plan input files hash-match in the archived remote project,and all54 copied run artifacts match remote bytes. Original model/tracker builders,strict final state load,eval/requires_grad(false),no_grad,seed2027,torchthreads1 andTF32false are source verified; no optimizer/backward/AMP/half path is introduced. Runtime precision/tensor dtypes are not separately dumped. The49-file plan does not independently seal every imported lib file,experiment YAML,installed package or every image byte; selected image paths exist and first-frame image hashes were captured. Original RankSTOP training native exit remains unknown; this inference run has its own genuine exit0.

### L8 — Profile overhead and hotspot interpretation

Both cProfile arms use only the first stream2up and are excluded from the10-arm speed ranking. Each has383302 total/352879 primitive calls. read_pair cumulative time is4.157345s(CV80) versus2.461215s(CV1),while track is9.601712s versus10.085093s. These overlapping Python cumulative timings can contain CUDA waiting and do not constitute kernel attribution or proof of a single causal bottleneck.

### L9 — One saved memory-sum discrepancy retained

arm7_n4_cv1 sample index6 at1793023553.9618304 reports GPU0 used59MiB/free24069MiB,sum24128MiB versus24127MiB in the other samples. It is before that arm timed start1793023563.5502985,far below the peak,and does not alter any recomputed FPS,peak or timed utilization. Cause is undetermined. The initial audit harness treated exact used+free as invariant; that unsupported check was removed while preserving initial script/output and this observation. No experimental data changed.

## Claim impact

| Claim | Assessment |
|---|---|
| N8/CV1 had the largest aggregate FPS among these10 measured arms. | supported |
| BestN8 is4.65–4.90x the measuredCV1 N1 baselines. | needs_qualifier — 4.6472631494992624x versus repeat and4.899092770992562x versus first; aggregate short-workload rates only. |
| All stored predictions/routes/final-state digests are exact across settings. | supported — Scope is saved records for these8 prefixes; no independent NN replay or raw-state rehash. |
| This tracker runs49.33FPS on one live stream or full97dev/test. | unsupported |
| Changing OpenCV/concurrency improves accuracy or universally preserves all trajectories. | unsupported |
| No training or precision-reduction change was introduced by this standalone profile. | supported_with_evidence_boundary — Source path and runtime unchanged-state assertions; precision not independently replayed and full environment not cryptographically sealed. |
| Reported total model parameter count is103218887. | runtime_witness_supported — Identical in all34 worker receipts and all arm summaries; count expression inspected,checkpoint/model not loaded again by auditor. |
| The original training process exited0 because this profile exited0. | unsupported |

## Reproduction and trace

This audit used only stdlib parsing,hashing,arithmetic and saved pstats files. The single read-only SSH operation was restricted to archived profile/input hashes and installed TRAIN initialization provenance. No active dev status,resource,temperature,power or PID query was made. No source,input,result,plan,HANDOFF or Git state was changed.

- Report JSON: `reports/RANKSTOP_INFERENCE_SCALING_RESULT_AUDIT_20261026.json`
- Deterministic verifier/output: `.aris/traces/experiment-audit/2026-10-26_rankstop_inference_scaling_result_review/verify_saved_artifacts.py` and `.aris/traces/experiment-audit/2026-10-26_rankstop_inference_scaling_result_review/DETERMINISTIC_RECOMPUTATION.json`
- Read-only remote script/output: `.aris/traces/experiment-audit/2026-10-26_rankstop_inference_scaling_result_review/remote_readonly_provenance.py` and `.aris/traces/experiment-audit/2026-10-26_rankstop_inference_scaling_result_review/REMOTE_READONLY_PROVENANCE.json`
- Full request,response,metadata and hash manifest: `.aris/traces/experiment-audit/2026-10-26_rankstop_inference_scaling_result_review/`

No additional run is required to accept the qualified measurements in this report. Stronger claims remain unsupported by this artifact set.
