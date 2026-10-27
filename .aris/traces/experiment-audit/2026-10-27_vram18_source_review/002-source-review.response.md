# VRAM18 training source review — R2

**SOURCE_PASS — 0 blockers.** Same-family/provisional. Requested Astra/max; actual model and reasoning effort UNATTESTED. This is source admission only, with no remote/runtime/NN/weight inspection or implementation edit.

R1 is preserved byte-for-byte under `.aris/traces/experiment-audit/2026-10-27_vram18_source_review/round1/`. Its three blockers are resolved:

- **T1**: Startup now captures all same-wave task identities/progress/memory. Health scoring requires >=600 seconds, exact task set, running native wrapper/child identities and unchanged command, positive progress and finite losses. It compares actual aggregate update rate against the sealed original-eight baseline for phase4 and preceding admitted wave thereafter, requires each original task retain >=0.4 of its baseline rate, summed task Swap growth <=65536KiB and current MemAvailable >=8GiB. Phases6/8 require a bound ADMIT_NEXT_WAVE health receipt and the numeric conditions. Scheduling metadata retains the nominal +1200s observation node. Operational state-dependent rates are not claimed as paired speedup or online FPS. Evidence: `tools/launch_vram18_training_wave.py:15`, `tools/launch_vram18_training_wave.py:35`, `tools/launch_vram18_training_wave.py:98`, `tools/launch_vram18_training_wave.py:194`, `tools/observe_vram18_training.py:27`.
- **T2**: Original/prior-wave warmed tasks are cuda_ready=True; current-wave registrations are False. The exact guard requires warmed PIDs <= actual CUDA PIDs <= registered child PIDs, and caps registered live same-card children. CPU initialization no longer requires premature GPU visibility, while unknown processes, missing warm owners and occupied registered slots fail. No preemption/retry/fallback is introduced. Evidence: `tools/launch_vram18_training_wave.py:92`, `tools/launch_vram18_training_wave.py:101`, `tools/launch_vram18_training_wave.py:132`, `tools/launch_vram18_training_wave.py:145`, `tools/launch_vram18_training_wave.py:156`, `tools/launch_vram18_training_wave.py:174`.
- **T3**: Each sanity and formal launch receipt is written to the wave cases before the next task or delay. The cases retain partial-launch identities even while the stage label is at the preceding coarse boundary. Sanity admissions remain mandatory before formal; all startup evidence and measurement_start precede FORMAL_STARTUP_PASS. No resume/retry or restart of the original eight is added. Evidence: `tools/launch_vram18_training_wave.py:155`, `tools/launch_vram18_training_wave.py:157`, `tools/launch_vram18_training_wave.py:173`, `tools/launch_vram18_training_wave.py:175`, `tools/launch_vram18_training_wave.py:194`.

All57 training/evaluation source hashes,24 new plan hashes and the index-bound observer are coherent. The original55 source files, eight original plans, exact32-configuration grid, seed, training recipe and input/order bindings remain unchanged. All98 generated programs parse;48 argv vectors match. Five ownership and nine health fixtures pass; their data are explicitly synthetic and do not constitute measured results.

The actual R4 paired allocator result remains unread and uncertified. Native0 for both cases, exact265-state/optimizer/CSV equality, lower compact sampled peak PSS and each candidate's real sanity remain mandatory. Stage6/8 additionally require actual bound health admission. No original job is stopped, altered or restarted.

Issued24 exact per-plan source guards, each binding this aggregate report SHA. Evaluation retains full97dev at three costs, fresh result audit and uniform frozen three-test configuration, total parameters and frame-service FPS.

Index SHA256: `7ee21e8e15a407c8dd20e9792cb84e44a07ef303a538f94174afc4fd0084892b`.

Launcher SHA256: `4ce30f1fdec91bc199a410861bedfd5d4c058753114de01b4c7703a69c680d42`.

Extended protocol SHA256: `6f0a4c021cdab555f686bd7a3b00262df35487653cd3fd2ed1235adaffad618f`.


# VRAM18 evaluation source review — R2

**SOURCE_PASS — 0 blockers.** Same-family/provisional; requested Astra/max, actual model/effort UNATTESTED.

The resealed57-source/32-plan protocol retains all original evaluation rules and eight original plan hashes. It still requires native0/full882×2 completion and265-tensor witness; actual11-frame smoke+reload; full97dev/54646frames at costs0/.05/.15; fresh audit binding the exact plan, checkpoint and three reports; strict dev SR>56.43314371195134; and one frozen checkpoint/config across LasHeR245/220703, RGBT210210/104706 and RGBT234234/116650. Dataset GT, supplied toolkit metrics, actual total parameters, decode+initialize-inclusive service FPS and mean/P95 timing remain unchanged.

Training R2 now passes source review, but actual R4 allocator equivalence/reducedPSS, candidate sanity, wave health and full-training/evaluation results remain runtime-gated. No live result, model quality, accuracy or FPS is certified. R1 report bytes are preserved in the review trace.

Protocol SHA256: `6f0a4c021cdab555f686bd7a3b00262df35487653cd3fd2ed1235adaffad618f`.
Training source review SHA256: `e5f2a2d1b94e5516b7947eb0e259efe08c269d72e1e791373057d33657d6e374`.
