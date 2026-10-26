# Training concurrency source review

**SOURCE_PASS; 0 blocking issues.** No concrete worker-import or IPC deadlock blocker found for the intended direct-script+spawn invocation. Requested reviewer:gpt-6-astra/max; actual route:UNATTESTED. Same-family/provisional.

Plan SHA:`0f9a31ac23c4a3c1450ef683e3c386e2f138756985ae4a71864a116543c03ce5`. New source SHA:`c76b74b3efd4828f909a63842ec94c33e9da6eb52c65a9ce674fdc80baaf34fa`. All51 source hashes/ASTs pass; the active50 are unchanged. All28 input bindings pass using16 current local hashes and12 unchanged remote-only bindings from sealed prior verification. No active remote query or neural execution was performed.

The script directory remains on sys.path when the profiler prepends ROOT; multiprocessing spawn inherits it. Direct-script --help succeeds, and real spawned CPU workers resolve the three native trainer helpers correctly. The target environment's archived actual sanity/formal-startup receipts establish those unchanged helper imports and native4-update admission. First additional1 remains the profiler's actual NN/process admission before larger arms.

Each of8 fixedTRAIN streams reloads originalB6250 and creates a fresh optimizer/tracker/NumPy sampler. The native causal track,teacher,replay,loss and step_optimizer paths are reused. Current GT follows tracking, teacher state-preservation assertions remain, and32 finite actual optimizer updates are required per stream. Stream assignment and final unique coverage preserve256 updates per arm. Freeze/changed/optimizer checks and child exit0 are required. GPU1 totals are2/3/5/9/2 NN processes including its existing formal task; GPU0 is observed only, andGPU2/3 are not used.

Five normal CPU spawn arms at1/2/4/8/1 workers passed, each covering8 streams/256 mock updates. Three injected failures—badREADY,worker exit7 during work and a deliberately held worker—each propagated failure, released both barriers and left no owned worker alive. The held worker exercised terminate/join. CPU fixture waits were capped at5s/.25s; production uses180/900s message waits and30s joins. This tests control flow, not GPU execution. Source cleanup is limited to children created by that arm and addresses the recorded priorR1 stranded-worker failure; no retries or fallback appear.

Timing is separated correctly. Model construction/initial load are beforeREADY. Training wall spans release through the latest synchronized worker completion, including stream resets,tracking,teacher/backward/AdamW and step JSON writing. CPU state validation occurs after those completion timestamps and is excluded from training wall; validation-inclusive and spawn-to-join timings are separate. Full step hashes include timing; the prediction/route hash removes route.seconds. Torch seeding is per worker, sampler reset per stream, and the plan makes no training bitwise-equality guarantee.

Existing-run observations occur only at arm boundaries and record logged CSV updates. Buffering/flushes limit their time resolution; they are not exact per-step timers. Repeated additional1 checks cache/load drift. Prefix throughput is not full882 training throughput, inferenceFPS or a proven causal acceleration; altered learned routes can change conditional work.

The profile validates its current native wrapper and the expected existingGPU1 wrapper/child/argv before work. Recorded local M0/native0/reload/freeze/replay and150-finite-update startup evidence was inspected without querying active jobs. Live identity checks and actual first-arm NN success remain required at execution; larger-arm GPU memory capacity is not established by CPU fixtures.

| File | Change | Instrumentation |
|---|---|---|
| `scripts/profile_candidate_training_scaling.py` | Standalone file created by executor; reviewed only | Resource/progress observations,training workers,timers,JSON outputs,owned-child cleanup; lines21-231 |
| `reports/TRAIN_CONCURRENCY_PLAN_20261026.json` | Configuration created by executor; reviewed only | Fixed streams,arms,hashes,job identities and scope |
| Active50 sources | Unchanged | No inline instrumentation |
| `reports/TRAIN_CONCURRENCY_SOURCE_REVIEW_20261026.json/.md` and `.aris/traces/system-profile/2026-10-26_training_scaling_source_review/` | Created by reviewer | Source findings and CPU-only fixtures |

Detailed tests:`.aris/traces/system-profile/2026-10-26_training_scaling_source_review/cpu_verification.json`. Source review is complete; the planned first actual arm is ready to attempt under its runtime identity checks.
