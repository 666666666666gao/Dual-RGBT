# VRAM18 evaluation source review — R1

Status: **SOURCE_PASS — 0 evaluation-source blockers**. Fresh same-family review; acceptance provisional. Requested Astra/max; actual model and reasoning effort **UNATTESTED**.

The extension retains the original protocol's full evaluation contract. Only the source map, registered plan map, review path and scope text differ. All57 source hashes and all32 plan hashes match; the original eight plan bindings are unchanged.

Dataset GT and the supplied toolkit remain authoritative, with no prediction-statistic normalization. The source requires completed full882×2 training/native0 and an exact265-tensor witness; actual11-frame smoke and strict reload; full97dev/54646 frames at costs0/.05/.15; a fresh result audit binding the plan/checkpoint/exact three reports; and strict dev SR>56.43314371195134. Eligible models freeze one checkpoint/config across all three full tests: LasHeR245/220703, RGBT210210/104706, RGBT234234/116650.

The result schema preserves actual total parameters103218887, GPU/input dimensions, decode+initialization-inclusive frame-service FPS and mean/P95 latency. Parallel aggregate throughput and unpaired baseline speedup are excluded. Relevant source: `scripts/joint_evaluation_contract.py:21,32,68,124,147,240,266`; `scripts/run_joint_candidate_evaluation.py:36,77,103,115,143`; `tools/joint_candidate_results.py:16,45,79,124`; `scripts/benchmark_inputs.py:102`; `dual_rgbt_global/candidate_tracker.py:21,65`.

This verdict does not grant training admission: the training source review is SOURCE_FAIL/T1–T3. No runtime/model/accuracy/FPS result is certified, and no per-case training PASS is issued. The actual allocator proof and all runtime ownership/capacity, completion, smoke, dev-audit and test-freeze gates remain mandatory.

Protocol SHA256: `e47ebe248fc8893ebb5a89322428effe7498f7757000f231fac164c9c1e4cee2`. Original protocol SHA256: `9ad6d14fd611dbd8226f187d77ed523c646f344388dfbb1138ac5750df3d83f0`. Exact source and input maps are in the JSON report. Trace: `.aris/traces/experiment-audit/2026-10-27_vram18_source_review`.
