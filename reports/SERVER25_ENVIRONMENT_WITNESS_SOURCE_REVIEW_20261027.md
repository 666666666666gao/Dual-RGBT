# Server25 environment witness source review

- Status: **SOURCE_PASS**
- Blocking issues: **0**
- Review: fresh same-family context, provisional; actual reviewer model **UNATTESTED**.
- Scope: specified witness, benchmark reload helpers, tracker interfaces and two deployment receipts, with only directly relevant source/data dependencies. No remote, CUDA, model or checkpoint execution.
- Witness SHA256: `58221a977c339d78239fc33297566fd151aa4d9229640687cb25c5a2967f5cfc`.

No concrete interface or source blocker was found. The source performs one seeded tiny-network CUDA forward/backward/AdamW update and strict weight reload, then uses the existing fixed B checkpoint for initialization plus ten online tracks. The local manifest contains sequence `2up` with 176 registered TRAIN image pairs; frames 0 through 10 are selected. The checkpoint digest matches the runtime-copy receipt, and the existing candidate-policy plan specifies 6250 formal updates.

The model constructor and strict net reload match benchmark variant `e1`. The asserted `103218887` is the sum of all parameters after freezing, so it is a total-model count. This review did not instantiate the model or independently reopen the remote checkpoint payload.

The saved tracker fields match the actual interface: state, templates, history, window, frame and since_query. The existing reload helper rebuilds/reloads the model and replays frame 10, checking exact box/route (excluding time), templates and history equality. It restores an in-memory snapshot; it does not claim disk serialization of tracker state.

The deployment plan fixes `/home/gaob/Dual-RGBT/conda-envs/dual_rgbt`. The witness verifies the prefix relative to its deployed root. Imported server26 benchmark command constants are not called by the selected helper path, and the witness has no remote/job/queue operations. The report does not independently attest which host invokes the script.

`smoke_fps` times the ten online frames including image reads, excluding initialization, construction and reload. The output explicitly separates it from formal FPS and accuracy; tiny-kernel and tracker update counts are separate. The final receipt is emitted only after reload verification.

## Checks and evidence

- **seeded_cuda_kernel: PASS** — Seed 2027, a CUDA Linear/GELU/Linear network, one finite loss/backward/AdamW step, finite parameter gradients and changed parameters, then a separate strict checkpoint reload with equal outputs. This is a tiny synthetic kernel witness, not tracker training. Evidence: `scripts/server25_environment_witness.py:31-50`.

- **model_and_checkpoint: PASS** — Candidate policy factory and full net strict load match the benchmark e1 construction. The pinned b84d0045... checkpoint digest matches the runtime-copy receipt; the existing candidate-policy plan identifies its formal run as 6250 updates. Sum over every model parameter, after freezing, is asserted equal to 103218887, so this is total rather than trainable parameter count. Runtime count and checkpoint payload were not executed/reopened in this local source audit. Evidence: `scripts/server25_environment_witness.py:53-60`; `dual_rgbt_global/candidate_policy.py:87-90`; `dual_rgbt/model.py:34-40`; `scripts/run_benchmark_tracking.py:21-29`; `reports/SERVER25_REJOIN_RUNTIME_COPY_20261027.json:16-19`.

- **true_train_frames: PASS** — The inspected manifest is LasHeR train_smoke, one sequence 2up, 176 RGB/thermal pairs under the server25 trainingset path. Its SHA256 equals the copied-input receipt. Initialization reads frame index 0 and tracking reads indices 1 through 10: 11 distinct real image pairs, 10 online tracks. Only the dataset initial box enters the tracker; later labels are not used to select actions or produce an accuracy score. Evidence: `scripts/server25_environment_witness.py:61-76`; `scripts/benchmark_inputs.py:109-124`; `dual_rgbt/data.py:15-19`; `data/benchmarks/lasher_train_smoke.json:1`.

- **tracker_interface_and_reload: PASS** — Constructor, initialize and track interfaces align. The snapshot before frame index 10 contains state/templates/history/window/frame/since_query and that image. reload_probe independently rebuilds and strictly reloads the model, restores the complete currently used tracker state, replays that image, and asserts exact box/route (excluding time), templates and history equality plus absent model gradients. The tracker snapshot is restored in memory; this is not a claim of persisted tracker-state serialization. Evidence: `scripts/server25_environment_witness.py:64-79`; `dual_rgbt_global/tracker.py:16-32`; `dual_rgbt_global/candidate_tracker.py:15-20`; `dual_rgbt_global/candidate_tracker.py:50-69`; `scripts/run_benchmark_tracking.py:105-125`.

- **server25_paths_and_isolation: PASS** — The reviewed deployment plan fixes /home/gaob/Dual-RGBT and its original conda prefix. Witness paths derive from the deployed file root, and sys.prefix must resolve to that root/conda-envs/dual_rgbt. The called freeze/reload/build_model/build_tracker path does not invoke benchmark main, evaluation_command or its server26 constants. There is no SSH, process control, queue operation, training launch or write to server26. This is source verification of the planned server25 invocation, not host identity attestation. Evidence: `scripts/server25_environment_witness.py:13-30`; `scripts/server25_environment_witness.py:53-61`; `reports/SERVER25_REJOIN_ENVIRONMENT_PLAN_20261027.json:3-11`; `reports/SERVER25_REJOIN_RUNTIME_COPY_20261027.json:9-10`; `scripts/run_benchmark_tracking.py:21-42`; `scripts/run_benchmark_tracking.py:105-125`.

- **scope_and_fps: PASS** — Timing covers 10 tracked frames including image reads, with CUDA synchronization, and excludes initialization/model loading/reload. It is named smoke_fps and explicitly excluded from formal benchmark FPS or accuracy. Kernel optimizer_updates=1 and tracker optimizer_updates=0 are separate. COMPLETE.json is written only after the checks and reload return. Evidence: `scripts/server25_environment_witness.py:66-88`.

- **syntax: PASS** — Parsed all three primary Python sources with stdlib ast.parse(feature_version=(3,7)); no source module was imported or executed. The first generic python launcher attempt failed with No pyvenv.cfg file; the installed uv-managed CPython 3.13 interpreter completed this syntax-only check offline. Evidence: `scripts/server25_environment_witness.py`; `scripts/run_benchmark_tracking.py`; `dual_rgbt_global/candidate_tracker.py`.

## Claim boundary

This is source approval for the bounded environment witness. Actual environment acceptance still requires the native runtime receipt. No formal FPS, accuracy, full tracker-training or four-GPU admission claim is supported by this review. No fixes are required.
