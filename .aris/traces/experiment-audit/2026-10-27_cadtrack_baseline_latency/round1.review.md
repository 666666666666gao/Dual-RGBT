# CADTrack baseline latency source review — 2026-10-27

Status: **SOURCE_FAIL**, one blocking issue. Fresh same-family review; acceptance provisional. Requested gpt-6-astra/max; actual model and effort UNATTESTED.

B1: The runtime source map omits the loaded CADTrack YAML/default configuration and original model factory, backbone and head. Consequently its hash guard does not bind the configuration/forward being timed. Add the actual baseline dependencies to the existing source map and refresh/re-review the plan. No tracker or trainer behavior change is needed. Local key baseline sources match the historical latency review; no alteration is alleged.

The remaining bounded checks pass: registered six-channel input and TrackerParams interfaces match the original tracker; online paired templates and the original update rule are retained. Three TRAIN sequences provide 20 warm frames and200 measured frames each. CUDA-bracketed service timing includes crop/preprocessing/forward/head/template update; end-to-end also includes image decode. FPS is600 divided by summed seconds, with mean/P95 latency, and initialization is separate.

The two selected GPU0 wrappers are223759 and223760, from the exact hashed execution receipt. Before signals, native PID/command, actual cmdlines, UID, PGID, child GPU and exclusive GPU0 compute-client membership are checked. Only those two groups are signaled; finally resumes both on ordinary Python exceptions and verifies non-stopped states before completion. Launch the profiler detached through the native task wrapper; this review does not promise recovery from uncatchable process termination.

All59 registered source hashes and the execution receipt hash matched. Expected checkpoint/manifest hashes match existing copy receipts; reviewer did not open weights or remotely inspect data. Offline AST parsing passed the profiler, launcher, wrapper and tracker. The local manifest is absent, so actual sequence availability remains an explicit pre-pause/runtime requirement.

This review does not establish measured FPS, actual total parameters, baseline accuracy or a matched Ours speedup. No remote, GPU, model or checkpoint-payload operation was performed. The initial failed plan/review are preserved in `.aris/traces/experiment-audit/2026-10-27_cadtrack_baseline_latency/`.
