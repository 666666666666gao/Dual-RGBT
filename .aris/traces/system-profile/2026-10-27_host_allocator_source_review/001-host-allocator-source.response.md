# Host allocator profile source review

Status: **SOURCE_FAIL**. One blocking issue. No per-case SOURCE_PASS reports are issued.

This is a fresh source-only experiment-audit review. Requested route: gpt-6-astra / max; actual model and effort: UNATTESTED. Review independence: same-family; acceptance: provisional. No active task, log, PID, weight, GPU or SSH query and no neural execution occurred.

## Blocking issue B1: remote profile lifecycle is not bounded

`tools/profile_host_allocator.py:46` creates the new wrapper with `start_new_session=True`. Lines49-60 loop indefinitely on its status file, with no remote deadline or direct child-liveness test. Line61 only waits after the file reports a terminal state. The600-second timeout at line67 bounds the local SSH call; it does not terminate/reap the detached remote profile group.

Therefore the source does not implement the requested bounded short profile. A stale RUNNING record cannot end the loop; an observer read/assertion failure or local transport timeout can leave the new profile child unmanaged. This is a source finding, not a claim that any live job has failed. The evidence is the actual generated control flow, not a speculative request for a fallback.

Required change: give the generated sampler an explicit deadline and direct child-liveness handling, and terminate/reap only its own newly created profile process group on deadline or sampling failure. Record an honest failed outcome. Add no restart/fallback, and do not signal or edit the original eight tasks. Rehash and re-review the revised source/plans before launch.

## Passing source checks

- All51 current source hashes match; all51 source files parse. Both case plan hashes match. The original50-source review and aggregate-review hash remain bound.
- Both cases preserve all24 original input bindings and the exact882-sequence order. Twelve local non-weight inputs freshly hash-match. The other12 bindings match a sealed historical CPU receipt; no remote input or weight was freshly read.
- Default and compact differ only in the allocator environment and unique run/task/review paths. Seed2027, GPU2, SOTep14, originalB6250, K4, cost0, losses, learning rates and architecture remain unchanged.
- The archived eight owner records match the pre-expansion baseline. The generated program requires exact GPU2 owner child argv/native PID and compute-PID set, then launches only a unique sanity task. No original task is modified or preempted.
- Exact generated default/compact/equality programs compile without execution. Their argv correctly reaches the wrapper and trainer with `--sanity`; native parent/task/GPU/argv binding is checked. Successful stdout contains a single JSON result while training output goes to its own log.
- The real trainer performs4 optimizer updates/20 causal track frames, checks251 frozen/14 changed tensors and step counts, and strictly reloads12 prediction fields. The sampler requires native COMPLETE/exit0 and actual child exit0. These gates are source checks, not new runtime results.
- Planned CPU comparison requires exact265 model tensors, optimizer equality and both CSV files. It runs only after both successful admissions. No formal run is launched; future compact formal use remains dependent on actual paired evidence and separate admission.
- Two-second sampling is purposeful instrumented measurement, not LLM polling. PSS is correctly described as sampled process memory including reload, without continuous-peak or steady-state claims.

## Integrity checklist A-F

A. PASS: planned training uses dataset TRAIN boxes and IoU-based labels after causal tracking (`train_candidate_joint_quality_trajectory.py:186,297`; `train_candidate_evidence_policy.py:57,72`).

B. PASS: no performance-score normalization or new accuracy metric; memory is rawKiB and equivalence is exact equality (`profile_host_allocator.py:59,77`).

C. PASS: the plans make prospective claims; no new profile result is asserted or opened. Existing archived capacity reports are recorded as historical.

D. PASS: prescribed sampling, training, optimizer, reload and equality paths are called, rather than unused functions reported as results.

E. PASS: scope is two sequential short admissions, one seed and one TRAIN sequence,4 updates/20 frames each. No fulltraining/accuracy/FPS/throughput improvement is claimed. B1 separately blocks the bounded-lifecycle requirement.

F. PASS: classification is planned `real_gt` training plus process-memory instrumentation, not benchmark accuracy evaluation.

## Evidence and limits

The JSON report contains the complete source hash map, exact plan hashes, line references and operations declaration. Trace files retain frozen source/plan inputs, generated programs, deterministic verification and this response. OriginalB and SOT hashes were checked as unchanged bindings against sealed receipts, not reread from weights. Current ownership/capacity, both actual admissions, CPU numerical equality and formal compact use remain unverified/not admitted.
