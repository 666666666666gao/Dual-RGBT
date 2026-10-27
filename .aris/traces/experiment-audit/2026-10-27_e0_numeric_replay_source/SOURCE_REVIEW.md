# E0 numeric replay source review

Status: **SOURCE_PASS**. Blocking issues: **0**. Issues: none.

Date: 2026-10-27. Audit: experiment-audit, fresh source review. Requested route: gpt-6-astra / max; actual model and effort: **UNATTESTED**. Review independence: **same-family**. Acceptance: **provisional**.

Exact plan: `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json`  
SHA-256: `20b65baa4e19feecedcdaa3b8de2c58ccfaaa2bcd9f5bbf28155b11b83bbc223`  
Replay source SHA-256: `f4d13c941a29e9e763a04ccd07aafef56dc388dfbbf4b654c8bbe3b98f6e1aff`.

The exact 120-entry source map is preserved in `SOURCE_REVIEW.json`.

## Checks

### gt provenance: PASS

Archived model trajectories are explicitly labeled reproduction references, not GT or tuning labels. Real dataset box[0] initializes each trajectory; subsequent boxes are never supplied to track(). Six archive hashes and every first-11-frame prefix match the local primary files.

Evidence: `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:149`; `tools/replay_e0_numeric_settings.py:61`; `tools/replay_e0_numeric_settings.py:65`; `reports/E0_NUMERIC_REPLAY_PREFIXES_20261027.json:418`.

### score normalization: PASS

Comparison reports an integer exact-frame count and raw maximum absolute box difference after identical six-decimal formatting. No accuracy percentage or denominator from prediction statistics is reported. Model entropy normalization is an internal policy feature, not an evaluation metric.

Evidence: `tools/replay_e0_numeric_settings.py:68`; `tools/replay_e0_numeric_settings.py:74`; `scripts/run_benchmark_tracking.py:246`; `scripts/run_tracking.py:112`; `dual_rgbt/model.py:150`.

### result existence: PASS

This is a READY source plan, not a claim of completed replay. All six reference files exist and match their hashes and copied prefixes. The runtime creates a new run directory exclusively and emits COMPLETE.json only after all four phases and the post-source check.

Evidence: `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:2`; `tools/replay_e0_numeric_settings.py:39`; `tools/replay_e0_numeric_settings.py:79`; `tools/replay_e0_numeric_settings.py:86`.

### dead code and causal execution: PASS

The entrypoint calls main; each phase creates a fresh tracker for each sequence, reads frame 0 and frames 1..10, and calls the actual E0 predict path. Templates, motion history, and crops depend only on initial state and preceding predictions. Initialization is included in each 11-box comparison: the plan yields 12 initializations and 120 actual track/predict calls, 132 stored boxes in total, if runtime completes.

Evidence: `tools/replay_e0_numeric_settings.py:55`; `tools/replay_e0_numeric_settings.py:60`; `tools/replay_e0_numeric_settings.py:64`; `tools/replay_e0_numeric_settings.py:91`; `dual_rgbt/data.py:15`; `dual_rgbt/tracker.py:34`; `dual_rgbt/tracker.py:47`; `dual_rgbt/model.py:181`.

### numeric control and repeats: PASS

The plan fixes cuDNN TF32 to False/True/False/True; every phase explicitly fixes CUDA matmul TF32 False. Same loaded model, sequence order, cost 0, inputs, and fresh tracker state are used throughout. cuDNN benchmark is observed, not changed. Raw predictions/routes are retained for repeated-phase checks; repeated-phase equality is not preclaimed.

Evidence: `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:142`; `tools/replay_e0_numeric_settings.py:45`; `tools/replay_e0_numeric_settings.py:48`; `tools/replay_e0_numeric_settings.py:51`; `tools/replay_e0_numeric_settings.py:55`; `tools/replay_e0_numeric_settings.py:60`; `tools/replay_e0_numeric_settings.py:76`.

### fixed weights and training protection: PASS

The only loaded checkpoint identities are the inherited CADTrack constructor checkpoint and historical E0 stage_c_all882_20261013/latest.pth. Both are hash pinned before model construction; E0 loads strictly and is frozen/eval/no_grad. No optimizer, backward call, training entrypoint, signal, pause, active-checkpoint path, or source write is reached. Imported training module main is guarded. No target source or plan was edited by this audit.

Evidence: `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:5`; `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:130`; `tools/replay_e0_numeric_settings.py:21`; `tools/replay_e0_numeric_settings.py:31`; `tools/replay_e0_numeric_settings.py:51`; `scripts/run_benchmark_tracking.py:21`; `scripts/run_benchmark_tracking.py:40`; `dual_rgbt/model.py:34`; `scripts/train_global_evidence_policy.py:309`.

### native binding and metadata: PASS

Startup requires SOURCE_PASS with zero blockers, exact plan SHA and identical source map, then verifies all source/input hashes. Native status must be RUNNING with parent PID, exact task/command, and physical GPU 0; CUDA_VISIBLE_DEVICES must be 0. The sealed wrapper sets PCI_BUS_ID/GPU visibility and persists actual child exit. COMPLETE retains checkpoint SHA, initial backend flags, PyTorch/GPU metadata, each phase/sequence trajectory and comparison, parameter count and scope. COMPLETE must be retained with the reviewed plan and native status; it is not a standalone attestation of plan/source identity.

Evidence: `tools/replay_e0_numeric_settings.py:26`; `tools/replay_e0_numeric_settings.py:33`; `tools/replay_e0_numeric_settings.py:79`; `tools/replay_e0_numeric_settings.py:81`; `scripts/run_task.py:21`; `scripts/run_task.py:24`; `scripts/run_task.py:28`.

### source and input integrity: PASS

All 120 local source bytes match the exact plan source map. The inherited 119 source entries match the parent plan without additions or changes within that inherited map; replay helper is the sole extra source. Prefix JSON hash matches and all six embedded archive digests and first-11 arrays were independently verified. The two checkpoints and raw data/manifest.json are absent locally and were not read remotely; their runtime SHA checks are verified in source, not executed here.

Evidence: `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:8`; `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:128`; `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:130`; `tools/replay_e0_numeric_settings.py:31`; `tools/replay_e0_numeric_settings.py:79`.

### scope: PASS

The plan is explicitly limited to three existing LasHeR DEV sequence prefixes, eleven boxes each, four numerical phases. Both local registered LasHeR DEV manifests identify these as the first three sequences. The plan forbids full accuracy/FPS claims and TF32 score selection. The replay does not call benchmark main or scoring and drops route seconds.

Evidence: `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:136`; `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:141`; `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:149`; `tools/replay_e0_numeric_settings.py:67`; `scripts/run_benchmark_tracking.py:290`.

## Deterministic verification

- 120/120 local source hashes matched the plan.
- All 119 inherited source entries are unchanged; the only extra entry is the replay helper.
- Six/six archived prediction file hashes and all six first-11 arrays matched exactly.
- Eight relevant Python files parsed successfully with `ast.parse`; no target imports or neural calls were executed.
- Both local registered LasHeR DEV manifests list the requested three sequences first and carry initialization boxes matching both archives.
- The default local `python` shim failed with `No pyvenv.cfg file`; checks were then run with the existing `E:\python.exe` interpreter. This was a reviewer-tool environment issue, not a target execution result.

## Limits and claim impact

- SOURCE_PASS applies only to the audited source and exact plan hash. No neural model was executed, remote host queried, training status observed, or GPU task started by this reviewer.
- The CADTrack checkpoint, historical E0 checkpoint, and original data/manifest.json are not present locally. Their hashes inherit the prior plan and are mandatory preflight assertions in the replay. Actual runtime bytes and the referenced image contents remain unverified here.
- Exact-frame counts include the supplied initialization box. Runtime scope is 120 online predictions plus 12 initialization boxes, not 132 neural predictions.
- Four phases are two in-process repeats per numerical setting. They do not establish cross-process, cross-hardware, full-DEV, full-test, accuracy, FPS, or training-provenance claims.
- Historical effective TF32 flags were not recorded; any future prefix agreement can establish only bounded trajectory reproduction, not prove the original full-run numerical configuration.
- COMPLETE.json must be packaged with the exact reviewed plan/source review and native task status/exit; its own metadata omits the plan hash and native task identity.

Only the bounded source implementation is supported. Runtime completion, agreement, full accuracy, FPS, and current training quality are not established. No target code or plan was modified; no remote or training observation occurred.
