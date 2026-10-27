# CADTrack baseline latency source review — R2, 2026-10-27

Status: **SOURCE_PASS**, zero blocking issues. Fresh same-family review; acceptance provisional. Requested gpt-6-astra/max; actual model and effort UNATTESTED.

R1 B1 is resolved. The plan adds18 hashes for the original configuration, model factory/forward, backbone, head and supporting dependencies. All77 hashes match local files. The only plan change is these additions; the existing59 hashes and all timing/ownership/checkpoint settings are unchanged. Profiler, model and trainer code were not modified. Plan SHA256: `04699d8d564f8e6c0e52c2a8f7b55beccbce76c6c21f86ad560a2ca93392eeaf`.

The retained bounded checks pass: registered six-channel input and TrackerParams interfaces match the original tracker; paired templates and the original update rule are retained. Three TRAIN sequences provide20 warm frames and200 measured frames each. CUDA-bracketed service timing includes crop/preprocessing/forward/head/template update; end-to-end also includes image decode. FPS is600 divided by summed seconds, with mean/P95 latency. Initialization is separate.

Only the two GPU0 groups selected from the hashed execution receipt are signaled. Actual native PID/command, cmdlines, UID, PGID, child GPU and exclusive GPU0 compute-client membership are checked before pause. Finally resumes those groups on ordinary Python exceptions and verifies non-stopped states before completion. The executor confirms detached native-wrapper launch with start_new_session=True; uncatchable process termination recovery is outside this source verdict.

Expected checkpoint/manifest hashes match existing copy receipts; weights were not opened and remote data were not inspected. The local manifest is absent, so named-sequence availability remains a runtime check. The existing offline AST check passed; no code changed in R2, so it was not needlessly repeated.

**SOURCE_PASS means source readiness only.** It is not measured FPS, an observed parameter count, baseline accuracy or a matched Ours speedup. No remote, GPU, model or checkpoint-payload operation was performed. R1 and R2 plans/reviews are retained in `.aris/traces/experiment-audit/2026-10-27_cadtrack_baseline_latency/`.
