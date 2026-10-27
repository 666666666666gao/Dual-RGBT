# Server25 training source review

**Verdict: SOURCE_PASS. Blocking issues: 0.**

Fresh source reviewer: /root/audit_server25_training_launch_20261027. Requested policy: gpt-6-astra/max. Actual model and reasoning: UNATTESTED. Same-family, provisional.

The eight new plans are exactly the original unlaunched phase8 configurations, two per GPU. Original recipes and shared source bytes match. No NN, remote, checkpoint or server26 operation was performed.

Index SHA256: a106802a1407b5be185b70c7e9fd789535024bab1c71735aef407909a92fca46

## configuration_identity: PASS

All eight new plan hashes match the index. Their original paths are exactly the eight phase8 entries and the eight absent task/run receipts. Two plans per physical GPU. Seed2027, both original checkpoint paths/hashes, source hashes shared with old plans, sequence order, 882 names, two rounds, stride5, 917106 tracked frames, 182182 updates, 1764 initializations, LR/weight decay and each K/cost/objective are unchanged.

Evidence: tools/register_server25_training.py:24, tools/register_server25_training.py:32, reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json:52, reports/SERVER26_UNLAUNCHED_GRID_REASSIGNMENT_20261027.json:61.

## source_and_inputs: PASS

All54 unique registered source hashes match local files. Every new input map exactly equals the server25 copy receipt plus the acceptance report hash. Both server25 manifest entries bind389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc. Native source/input hashing remains mandatory at start and completion; protected SOTep14 hash cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731 is retained.

Evidence: tools/launch_server25_training.py:25, tools/launch_server25_training.py:56, scripts/train_candidate_joint_quality_trajectory.py:255, scripts/train_candidate_joint_gainpair_trajectory.py:252, reports/SERVER25_REJOIN_RUNTIME_COPY_20261027.json:14.

## environment_evidence: PASS

Acceptance, native and document-verification hashes match, including the witness source and cited document. Parent native exit0 and fresh document-native exit0 support environment acceptance only; total parameters103218887 is recorded. These receipts do not substitute for the eight real M0 runs.

Evidence: reports/SERVER25_ENVIRONMENT_ACCEPTANCE_20261027.json:6, reports/SERVER25_ENVIRONMENT_PRIMARY_NATIVE_20261027.json:12, reports/SERVER25_ENVIRONMENT_DOC_VERIFICATION_20261027.json:40, docs/SERVER25_REJOIN_20261027.md:1.

## native_runtime_and_allocator: PASS

Every plan calls its original native trainer through run_task_four_gpu. The only execution_env addition is OPENCV_FOR_THREADS_NUM=1. No host-trim trainer, allocator tunable, altered model, changed loss, batch or data-order code is introduced.

Evidence: tools/register_server25_training.py:38, tools/launch_server25_training.py:103, scripts/run_task_four_gpu.py:20.

## admission: PASS

Before each batch, requires no compute processes across the node, exactly4 GPU readings each below500MiB, MemAvailable above64GiB, disk free above10GiB, and every plan input hash. Source-only review does not assert that these future live conditions already pass.

Evidence: tools/launch_server25_training.py:45, tools/launch_server25_training.py:98.

## partial_launch_identity: PASS

Each newly created process PID and complete command are saved before another starts. Existing task/run paths are rejected. Wrapper status records its PID, exact child command and real subprocess exit code.

Evidence: tools/launch_server25_training.py:93, tools/launch_server25_training.py:102, tools/launch_server25_training.py:109, scripts/run_task_four_gpu.py:24.

## native_parent_command_contract: PASS

Trainer own_native.pid equals getppid and own_native.command equals [sys.executable]+sys.argv. Launch uses the absolute native trainer path. Formal startup validates the same wrapper PID, command[7:] slice, wrapper /proc cmdline, single child exact cmdline and physical GPU environment.

Evidence: scripts/train_candidate_joint_quality_trajectory.py:263, scripts/train_candidate_joint_gainpair_trajectory.py:260, tools/launch_server25_training.py:64.

## sanity_gate: PASS

Eight concurrent four-update M0 runs are observed after300s. All must have native COMPLETE/exit0 with launched parent PID, exact sanity counts20frames/4updates/1initialization, selected execution equality, teacher state preservation, strict reload equality,251 frozen and14 changed tensors, and matching plan/source/input identity before any formal starts.

Evidence: tools/launch_server25_training.py:111, tools/launch_server25_training.py:120, scripts/train_candidate_joint_quality_trajectory.py:118, scripts/train_candidate_joint_gainpair_trajectory.py:114.

## original_B_formal_reset: PASS

Formal admission accepts only the completed sanity receipt hash, then both trainers build from the factory checkpoint, strictly load original candidate_b step6250, and create a fresh AdamW optimizer. M0 weights never initialize formal training.

Evidence: scripts/train_candidate_joint_quality_trajectory.py:279, scripts/train_candidate_joint_quality_trajectory.py:307, scripts/train_candidate_joint_gainpair_trajectory.py:276, scripts/train_candidate_joint_gainpair_trajectory.py:304.

## full_formal_startup: PASS

After all8 M0s pass, all8 formal jobs start;180s startup checks require each native RUNNING identity, non-sanity matching config, nonempty sequential logged updates and finite loss fields. Native trainers enforce full882x2 and182182updates. Next observation is anchored to the last formal launch plus1800s.

Evidence: tools/launch_server25_training.py:132, tools/launch_server25_training.py:75, scripts/train_candidate_joint_quality_trajectory.py:299, scripts/train_candidate_joint_gainpair_trajectory.py:296.

## controller_isolation_and_exit: PASS

Local entry point targets only SSH2025 and /home/gaob/Dual-RGBT. CPU controller imports no torch and runs through the existing native wrapper for its true exit code. Payload refuses changed existing files, uses new plan/run identities, and never targets server26 or retired queues.

Evidence: tools/server25_runtime.py:8, tools/launch_server25_training.py:153, tools/launch_server25_training.py:165, scripts/run_task_four_gpu.py:28.

## syntax: PASS

In-memory compile and ast.parse(feature_version=(3,7)) passed for the three new tools, wrapper, and both native trainers. No target module was imported or executed.

## integrity_scope: PASS

Trajectory supervision reads dataset TRAIN boxes after deployment-equivalent tracking, while teacher-derived utility is actual executed IoU against those boxes. No metric normalization or new accuracy/FPS/completion claim is introduced. Environment tiny-kernel update is an explicitly synthetic environment probe; ten tracked TRAIN frames are only an environment witness.

Evidence: scripts/train_candidate_joint_quality_trajectory.py:186, scripts/train_candidate_joint_gainpair_trajectory.py:182, reports/SERVER25_ENVIRONMENT_DOC_VERIFICATION_20261027.json:116.

## Limits

- Not a completed M0/formal runtime acceptance or accuracy/FPS claim.
- Runtime manifest/checkpoint contents were not read; local copies of the server25 manifests are absent. Their receipt bindings were checked, and remote admission/native source code enforce all input hashes before NN execution.
- Model/backend and reasoning effort are UNATTESTED; review is same-family/provisional.

The eight per-plan guards contain the exact native-required plan_sha256 and source_sha256 fields. All runtime gates remain required.
