# RankSTOP Dev Source Review — 2026-10-26

**Verdict: SOURCE_PASS — 0 blocking issues.**

Independent context review under `experiment-audit`; same-family, provisional. Requested routing: `gpt-6-astra`, reasoning `max`. Actual routing: **UNATTESTED**. No GPU or neural execution was performed.

This verdict covers the source path for the registered training smoke and three complete LasHeR dev evaluations. It does not certify training process exit, GPU reload, new scores, test eligibility, target attainment or speedup.

Plan SHA256: `e44dcdbbe0d2192cea2bb698fc9dd50594aef8756fa800d688414e369946e245`. All 48 source hashes match; the JSON report contains the entire plan source map. Six local input hashes match; ten planned training/data files are absent locally and were not rechecked.

## Blocking findings

None.

## Findings

### ground_truth_and_causality: PASS

Source chain loads original LasHeR init.txt into the train/dev manifest, then copies its boxes unchanged into benchmark manifests. Dev inference initializes with frame0 GT only. Subsequent tracker calls receive decoded images and causal tracker state; sequence boxes are written for scoring after inference. Training source reads current GT after tracker.track advances. Original dataset pixels and init.txt files were not available locally for independent content revalidation.

Evidence: `scripts/prepare_data.py:11-25`, `scripts/prepare_data.py:27-30`, `scripts/prepare_benchmark_inputs.py:24-39`, `scripts/run_rankstop_dev.py:97-125`, `dual_rgbt/data.py:15-19`, `dual_rgbt_global/candidate_tracker.py:20-64`, `scripts/train_candidate_rankstop_trajectory.py:167-191`.

### scope_and_completeness: PASS

Local dev manifest has exactly 97 unique sequences and 54,646 image/GT pairs; its names exactly match lasher_val.txt in order and have zero overlap with the 882 training-plan names. Smoke is the registered one training sequence (176 frames), with only 11 frames executed. Dev iterates every sequence and frame, requires 54,646 timing values, and the scoring chain checks every prediction shape, unique sequence coverage, and total frames.

Evidence: `data/benchmarks/lasher_dev.json:1`, `data/benchmarks/lasher_train_smoke.json:1`, `lib/train/data_specs/lasher_val.txt:1`, `reports/global_candidate_rankstop_trajectory_plan_20261018.json:67-950`, `scripts/benchmark_inputs.py:109-124`, `scripts/run_rankstop_dev.py:76-80`, `scripts/run_rankstop_dev.py:97-126`, `scripts/run_rankstop_dev.py:140-144`, `tools/score_rankstop_dev.py:43-51`, `scripts/score_tracking.py:26-56`, `scripts/score_tracking.py:66`.

### strict_reload_smoke: PASS

The smoke uses the final fixed C882 checkpoint. It saves pre-track state, templates, history, window, frame and query age before the last smoke frame, then reloads checkpoint net strictly into a fresh model. Reload requires bit-equal box, all route fields except time, templates and history, and absent gradients. Full dev requires the registered smoke native success plus matching plan/checkpoint and reload PASS. This procedure has not been executed by this review.

Evidence: `scripts/run_rankstop_dev.py:49-55`, `scripts/run_rankstop_dev.py:61-75`, `scripts/run_rankstop_dev.py:113-116`, `scripts/run_rankstop_dev.py:134-137`, `scripts/run_benchmark_tracking.py:105-125`.

### fixed_weight_and_native_identity: PASS

Plan binds the final C882 checkpoint, factory checkpoint, training plan/audit and source hashes; hashes are checked before and after inference. The model is loaded strictly, frozen/eval and enclosed in no_grad, with exact before/after model-state comparison. The current native RUNNING record must belong to the wrapper parent PID, exact command, task and allowed physical GPU, and run/task/cost must match registration. The scoring adapter requires COMPLETE with native exit_code 0 and corresponding config/COMPLETE identity. Existing run_task persists the actual subprocess exit code.

Evidence: `reports/rankstop_dev_plan_20261026.json:55-71`, `reports/rankstop_dev_plan_20261026.json:83-115`, `scripts/run_rankstop_dev.py:32-47`, `scripts/run_rankstop_dev.py:61-75`, `scripts/run_rankstop_dev.py:136-139`, `scripts/run_benchmark_tracking.py:40-42`, `scripts/run_task.py:18-32`, `tools/score_rankstop_dev.py:28-42`.

### plan_and_hash_binding: PASS

All 48 planned source files match their SHA256 values. All 6 available planned inputs match. The other 10 planned training/data inputs are absent in this local checkout and were not reverified here; the runtime requires all of them to match before model execution. Source review admission binds both exact plan hash and the complete source hash map.

Evidence: `reports/rankstop_dev_plan_20261026.json:5-71`, `scripts/run_rankstop_dev.py:32-38`, `scripts/run_rankstop_dev.py:138-139`, `tools/score_rankstop_dev.py:23-25`, `tools/score_rankstop_dev.py:36-44`.

### missing_training_native_status: PASS_WITH_WARNING

The CPU audit explicitly preserves actual_process_exit_code=null and current_native_status_available=false, and labels its status ARTIFACT_CHECKS_PASS_WITH_PROVENANCE_WARNING. It does not synthesize a native terminal record. The dev runner demands the null exit code and forwards warnings to config and completion; the scorer forwards warnings to metrics. This source review did not rerun the training artifact audit or independently verify the remote artifact counts.

Evidence: `tools/audit_rankstop_completion.py:20-71`, `tools/audit_rankstop_completion.py:72-86`, `reports/RANKSTOP_C882_COMPLETION_AUDIT_20261026.json:2-42`, `scripts/run_rankstop_dev.py:56-60`, `scripts/run_rankstop_dev.py:85-92`, `scripts/run_rankstop_dev.py:129-133`, `tools/score_rankstop_dev.py:57-60`.

### timing_semantics: PASS

frame_seconds includes image decode plus synchronized initialization or tracking for each of all 54,646 dev frames. Tracker synchronization bounds GPU work. Service-time samples exclude model loading, file writes and final verification; wall_seconds includes loop/file overhead but excludes the post-loop smoke reload and hash checks. Existing scorer computes mean/p95/fps from service-time samples. Smoke timing includes diagnostic snapshot overhead and is not a performance result. No efficiency comparison has been executed or established.

Evidence: `scripts/run_rankstop_dev.py:94-119`, `scripts/run_rankstop_dev.py:123-144`, `dual_rgbt_global/tracker.py:21-32`, `dual_rgbt_global/candidate_tracker.py:20-22`, `dual_rgbt_global/candidate_tracker.py:65-69`, `scripts/score_tracking.py:65-75`.

### existing_scorer_and_metric_normalization: PASS

score_rankstop_dev calls the unchanged score_tracking.main after native, full-dev, checkpoint, plan, manifest and exact float32 GT validation. AST interface checking found every required config field. All local dev/smoke GT values preserve their float32 bit patterns through .9g text. Existing scorer validates predictions/timing lengths, invokes PR_LasHeR/NPR/SR_LasHeR and averages toolkit curves with only standard percent scaling; there is no normalization by model output statistics. Independent CPU verification with the existing Python 3.12.12/numpy 1.26.4 environment matched all seven planned toolkit hashes, imported the actual dataset classes and three metrics, and executed load_manifest on both complete benchmark manifests. Torch was not imported. Actual scoring of new predictions remains pending. The existing toolkit rounds GT and predictions; NPR normalizes center error by GT width/height, not prediction statistics.

Evidence: `tools/score_rankstop_dev.py:24-56`, `scripts/run_rankstop_dev.py:85-92`, `scripts/run_rankstop_dev.py:124-125`, `scripts/score_tracking.py:17-18`, `scripts/score_tracking.py:21-34`, `scripts/score_tracking.py:39-78`.

### result_existence_and_claim_scope: PASS_WITH_WARNING

No new RankSTOP dev/test score is claimed. Plan reference PR/NPR/SR exactly matches the existing full-dev cost0.15 E1 report (97 sequences, 54,646 frames); it is explicitly a dev reference. Selection/test eligibility remains a separate future gate and this runner has only train_smoke/dev modes. The source PASS does not validate new metric results, target achievement, efficiency, or authorize test through the legacy E0/E1 formal identity.

Evidence: `reports/rankstop_dev_plan_20261026.json:117-126`, `reports/candidate_complete_adopted_20261016/e1_dev_cost0p15_freeze_recomputed.json:2-9`, `scripts/run_rankstop_dev.py:45-48`, `tools/score_rankstop_dev.py:60`, `reports/RANKSTOP_C882_COMPLETION_AUDIT_20261026.json:36-43`.

### legacy_e0_e1_protocol: PASS

The seven checked legacy entry/helper/scoring/selection files are byte-identical to HEAD 0ccee21460366d7f76fc3f9dabc4c0f75192e19c. The new runner imports model/tracker/reload helpers but does not relax old trained_checkpoint checks or rewrite old native task identities. New RankSTOP dev artifacts use their own registered run/task names and a dedicated scoring adapter.

Evidence: `scripts/run_benchmark_tracking.py:45-102`, `scripts/run_benchmark_tracking.py:145-185`, `scripts/benchmark_inputs.py:29-95`, `reports/rankstop_dev_plan_20261026.json:92-115`, `scripts/run_rankstop_dev.py:17`, `tools/score_rankstop_dev.py:15-60`.

## CPU verification

- All 48 plan source SHA256 values matched; no source mismatch or missing source.
- Built-in compile() accepted all 48 Python sources without imports, bytecode writes or neural execution.
- Six available input hashes matched; ten remote training/data inputs absent locally.
- Dev has exactly 97 unique sequences/54646 paired frames; names equal CADTrack lasher_val.txt in order; no overlap with 882 training names.
- Training-plan names equal the 882-name lasher_train.txt set. Smoke name is a member of that training set.
- All manifest GT is finite with positive initial size. Every dev/smoke GT scalar roundtrips float32 -> .9g -> float32 exactly.
- AST comparison found all nine config keys consumed by score_tracking.py supplied by the new runner.
- Seven legacy files are byte-identical to git HEAD 0ccee21460366d7f76fc3f9dabc4c0f75192e19c.
- Plan dev reference numbers exactly match existing E1 full-dev cost0.15 report.

## Warnings and limits

- **W1**: Final training native terminal status is absent. Actual process exit code remains unknown; artifact completion evidence cannot replace it.
- **W2**: SOURCE-only review: no GPU model construction, forward pass, strict reload, full dev inference, performance measurement or actual toolkit scoring was executed. All remain PENDING.
- **W3**: Ten planned training/data inputs and original dataset image/init.txt files are absent locally. Their remote contents and the saved training audit were not independently rechecked. Runtime source requires hash validation; local absence is a scope limitation, not a source-code blocker.
- **W4**: Actual new-prediction scoring remains pending. The existing local Python 3.12.12/numpy 1.26.4 environment and unchanged toolkit were independently CPU-import checked successfully. Toolkit utils.py uses list[str] annotations, so the designated server training Python 3.7 is not the scoring runtime; use the verified existing local scoring environment.
- **W5**: Service-time FPS excludes model loading, artifact writes and hash verification; it is not complete process wall-clock throughput. No speedup claim is supported by this audit.

Unavailable planned inputs:

- `data/manifest.json`
- `runs/stage_b_refreshed_20261013/latest.pth`
- `runs/global_candidate_policy_formal_20261014/latest.pth`
- `runs/global_candidate_policy_formal_20261014/COMPLETE.json`
- `runs/pipeline_tasks/train_global_candidate_policy_formal_20261014/status.json`
- `runs/global_candidate_policy_formal_20261014/config.json`
- `runs/global_candidate_rankstop_trajectory_formal_20261018/COMPLETE.json`
- `runs/global_candidate_rankstop_trajectory_formal_20261018/config.json`
- `runs/global_candidate_rankstop_trajectory_formal_20261018/trajectory_steps.csv`
- `runs/global_candidate_rankstop_trajectory_formal_20261018/latest.pth`

## Still pending

- On the designated server, revalidate every planned input and protected checkpoint hash.
- Registered train-smoke GPU execution and strict reload bit-equality, with actual native COMPLETE exit_code 0.
- Each of the three registered costs: full 97 sequences/54646 frames with frozen checkpoint and actual native completion.
- Run hash-bound actual toolkit scoring through score_rankstop_dev.py and inspect predictions, service timing and metrics.
- Apply the separate full-dev selection and improvement gate before proposing any new test registration.

Only the requested JSON and Markdown review reports were written. No training/inference code or plan was changed.

## Supplemental independent CPU check

The existing local Python 3.12.12 / numpy 1.26.4 imported `BaseRGBTDataet`, `TrackerResult`, `PR_LasHeR`, `NPR`, `SR_LasHeR` and `LasHeR` from the original toolkit. All seven toolkit hashes match the plan. The real `load_manifest` validated smoke 1/176 and dev 97/54,646. Torch was not imported; no GPU was used. New-prediction scoring remains pending.

The toolkit keeps its existing rounding and metric conventions (`rgbt/dataset/basedataset.py:19-24`, `rgbt/metrics/metrics.py:236-361`); NPR uses GT width/height (`rgbt/utils.py:118-119`). Scoring uses the existing local Python environment because `rgbt/utils.py:35` contains `list[str]`, which is unsuitable for the server training Python 3.7 runtime.

The executor separately reported server CPU verification of all planned input hashes. That remote check was not rerun by this reviewer and is not represented as independent reviewer execution.
