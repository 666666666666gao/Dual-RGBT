# E1 full-C882 experiment audit

**Overall: WARN. Deterministic checks: PASS. Blocking issues: 0.**

Completed E1 causal C training, exact final checkpoint integrity and all train query-label arithmetic pass independent deterministic checks; held-out accuracy, paired speed and contribution claims remain unsupported.

Fresh reviewer context; requested `gpt-6-astra` / `max`, actual model and effort **UNATTESTED**. `review_independence: same-family`; `acceptance_status: provisional`. Advisory to the already-running evaluation; no NN, optimizer, GPU/resource query or live-job management was performed.

## Verified completion

| Item | Evidence |
|---|---|
| Native training | Original wrapper PID 827849, physical GPU1, COMPLETE, exit 0; command/config/plan unchanged |
| Work | 882 train sequences × 2 causal rounds; 917,106 tracking calls; 1,764 initializations |
| Supervision | 182,182 exact ordered updates, 91,091 per round, all losses finite |
| Labels | 364,364 sampled + 34,422 selected-extra = 398,786; all IoUs/gains independently recomputed |
| State | 265 initial/final tensor keys/shapes/dtypes finite; 257 exactly frozen, only 8 declared tensors changed |
| AdamW artifact | Eight saved parameter states, all step 182,182 and finite; no optimizer execution |

The exact supervision positions for all 1,764 sequence-round groups were checked. Full tracking-call/initialization totals are corroborated by the successful immutable loop and its terminal assertions; the CSV does not independently log every non-supervised frame route. No loss reduction is treated as held-out accuracy.

Final checkpoint: `d4454775f085e1af65c61f53d4d21f284706289b6ff3aeb1167f73d79d31a362`. Initial B: `b84d00454717cae56048753e0fe54c1d6b92cfcd2820c01fa65296508488ccb3`.

## A ground truth provenance — PASS

All 882 train init.txt files were independently loaded and matched the hash-bound manifest boxes exactly as float64; valid-frame membership was recomputed. Train/dev/test names are disjoint 882/97/245.

Every one of 398786 saved query labels was independently recomputed from saved executed xywh predictions against original-image dataset GT. Model predictions determine boxes and the query state; they do not replace the reference GT.

C initializes from frame0 GT, tracks each subsequent image from its predicted state, and reads current train GT only after the tracker decision for valid stride5 supervision. Teacher queries use cloned pre-track templates/history; source asserts exact tracker state/template/history preservation before every update.

Online evaluation passes only image and initial_box to tracking. Later GT is serialized after predictions and used by the separate scorer. Dev GT is legitimately used for registered configuration selection; test GT is not a training/routing input in this audited code.

The protected CADTrack checkpoint hash is verified, but its upstream training-dataset history and the user-described stripped SOT provenance are not independently established by this audit. Train-only assurance applies to the audited C actions/labels.

Evidence: `scripts/prepare_data.py:11`, `scripts/train_candidate_trajectory_policy.py:158`, `scripts/train_candidate_trajectory_policy.py:163`, `scripts/train_candidate_trajectory_policy.py:172`, `scripts/train_candidate_trajectory_policy.py:179`, `scripts/train_candidate_evidence_policy.py:50`, `scripts/run_benchmark_tracking.py:216`, `scripts/run_benchmark_tracking.py:237`, `scripts/run_benchmark_tracking.py:247`, `C:/Users/gb/.codex_tmp/dual_rgbt_e1_full_c882_result_audit_20261017/remote_cpu_probe.py:57`.

## B score normalization — PASS

Training targets are ordinary original-image float32 intersection/union IoUs and signed query-minus-primary deltas, without clipping GT to the query/main ROI or normalizing by prediction maxima. All labels pass absolute tolerance 1e-6; observed errors are reported separately.

Prediction-normalized entropy in candidate_gains is an internal policy feature, not a reported metric or label denominator. Smooth-L1 quality loss plus mean of two or three gain losses is the recorded objective.

Registered scoring uses supplied toolkit precision thresholds and success curves with frame counts within each sequence and equal sequence averaging. NPR divides center coordinates by GT width/height. Toolkit rounding, initial-frame and legacy LasHeR invalid-GT behavior are retained; this is not independent certification of an external official protocol.

Evidence: `dual_rgbt_evidence/regions.py:43`, `scripts/train_candidate_evidence_policy.py:55`, `scripts/train_candidate_trajectory_policy.py:86`, `dual_rgbt_global/candidate_policy.py:50`, `scripts/score_tracking.py:58`, `tools/score_benchmark_tracking.py:114`, `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/utils.py:118`, `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:266`.

## C result existence and identity — PASS

Native task train_global_candidate_trajectory_formal_20261015, wrapper PID827849, physical GPU1, immutable command, COMPLETE/exit0, adopted config hash, original source/input plan maps and final checkpoint are joined across the intake, controller receipt and terminal observation.

Full CSV recount matches every ordered (round,sequence,frame,step), not merely sequence exposure. Both rounds contain 91091 updates across all882 sequences; all182182 loss triples are finite/nonnegative and total matches quality+gain within float32 rounding.

CPU checkpoint read compares all265 tensor names/shapes/dtypes, verifies initial/final finiteness, exact equality of all257 frozen entries and changes only in the8 declared quality/gain tensors. All8 saved AdamW state steps are182182 with finite state; no optimizer was instantiated or executed.

All398786 label rows and label SHA match COMPLETE. Full-frame route totals917106 and223077 auxiliary queries are checked for arithmetic consistency; per-frame non-supervised routes are available only as the successful executed loop plus terminal aggregates.

No E1 dev/test performance or paired latency is fabricated. HANDOFF historical E0 scores are outside current result recertification.

Evidence: `C:/Users/gb/.codex_tmp/dual_rgbt_e1_full_c882_intake_20261017/training/native_task.json:15`, `C:/Users/gb/.codex_tmp/dual_rgbt_e1_full_c882_intake_20261017/training/COMPLETE.json:2`, `C:/Users/gb/.codex_tmp/dual_rgbt_e1_full_c882_intake_20261017/training/COMPLETE.json:893`, `C:/Users/gb/.codex_tmp/dual_rgbt_e1_full_c882_intake_20261017/training/COMPLETE.json:915`, `scripts/run_task.py:27`, `scripts/train_candidate_trajectory_policy.py:103`, `scripts/train_candidate_trajectory_policy.py:218`, `C:/Users/gb/.codex_tmp/dual_rgbt_e1_full_c882_result_audit_20261017/local_checks.json:2`, `C:/Users/gb/.codex_tmp/dual_rgbt_e1_full_c882_result_audit_20261017/remote_cpu_probe.py:80`.

## D called computation and coverage — PASS

Exact native argv calls train_candidate_trajectory_policy.main; trajectory_stage calls tracker.track, sampled_targets, selected_target, controller_loss and step_optimizer. Actual supervised rows contain both modalities and allfive actions, with43433 selected auxiliary decisions,9011 matching sampled reuses and34422 extra-selected queries.

The full run therefore has real non-STOP and extra-query coverage absent from the historical4-update M0. Selected-execution box equality and teacher nonmutation are executed assertions at182182 supervised updates; this audit did not replay NN outputs or claim separate all-frame equality testing.

LocalEvidenceReader directly shares executor backbone/decode/head. The257 unchanged entries independently corroborate no C changes to encoder, readout, preview or other frozen state. Saved gradients and per-step head predictions are absent, so individual gradient magnitudes and per-step Smooth-L1 reconstruction are outside independent artifact checks.

The controller calls the separate registered scorers and freeze logic only after terminal dev tasks; source reachability is verified, while current E1 scorer completion is not claimed.

Evidence: `scripts/train_candidate_trajectory_policy.py:164`, `scripts/train_candidate_trajectory_policy.py:175`, `scripts/train_candidate_trajectory_policy.py:181`, `scripts/train_global_evidence_policy.py:40`, `dual_rgbt_evidence/policy.py:90`, `dual_rgbt_evidence/readout.py:75`, `dual_rgbt_evidence/readout.py:111`, `scripts/resume_candidate_benchmarks.py:106`.

## E scope and claim ceiling — WARN

One seeded run2027,882 train sequences,2 full causal rounds,917106 tracking calls,1764 initializations and182182 supervised updates are supported. This is completed training and artifact integrity, not held-out accuracy, robustness, convergence or mechanism novelty.

Saved launch evidence covers full LasHeR dev97/54646-frame configurations at costs0 and0.05, each using exactly the final checkpoint, K8, one shard and smoke_frames0. Both were RUNNING at the recorded observation. Cost0.15 is registered for later; no completed three-cost selection, frozen test configuration or test result is in this intake.

After allthree full dev results, source requires recomputation and freeze before full LasHeR245/220703, RGBT210210/104706 and RGBT234234/116650 tests. Paired baseline/E1 timing and scale-candidate experiments have separate protocols and receive no result credit here.

Observed actual Python owner activity is distinct from outer scheduler Ready/3221225786. The saved evidence supports historical owner activity and initial dev launch, not ongoing TaskScheduler hosting, present liveness or a claimed cause of interruption.

Evidence: `reports/global_candidate_trajectory_plan.json:951`, `reports/benchmark_evaluation_plan.json:90`, `reports/benchmark_evaluation_plan.json:123`, `C:/Users/gb/.codex_tmp/dual_rgbt_e1_full_c882_intake_20261017/dev_launch_receipts.json:1`, `C:/Users/gb/.codex_tmp/dual_rgbt_e1_full_c882_intake_20261017/observed_terminal_and_dev_state.json:1`, `scripts/resume_candidate_benchmarks.py:120`, `tools/freeze_benchmark_selection.py:26`, `scripts/run_benchmark_tracking.py:176`, `reports/e1_paired_latency_plan.json:123`, `reports/scale_candidate_training_plan.json:128`.

## F evaluation type — PASS

Saved executed predictions are compared to dataset GT; these are training counterfactual utility labels, not synthetic-reference accuracy. The downstream registered held-out OPE is a separate real_gt evaluation whose completion is not certified here. The preceding B initializer uses declared offline GT-jitter states; it is not the causal C trajectory result.

Evidence: `dual_rgbt_global/training.py:18`, `scripts/train_candidate_evidence_policy.py:50`, `scripts/train_candidate_trajectory_policy.py:172`, `dual_rgbt_evidence/regions.py:43`.

## Label recomputation and provenance

The single CPU-only SSH probe read the existing conda environment with `CUDA_VISIBLE_DEVICES=` and both checkpoints using `map_location="cpu"`. It matched 882 raw training annotation files (459435 unique frames) and 918870 listed RGB/TIR image paths. It rehashed all remote inputs unchanged before exit. Image pixels were not decoded or rehashed. Raw GT and label rows remain private.

All saved label IoUs and signed deltas were recomputed using float32 original-image area/union arithmetic with absolute tolerance 1e-6. Maximum errors: `{"main_iou": 0.0, "query_iou": 0.0, "delta_from_stop": 0.0}`. Negative gains: 214915. Labels whose GT center lies outside the main ROI: 46096; GT was not clipped into that ROI.

Query labels SHA256: `e61640e29a76a57e99260952cadbf57ff88b09dcfd57c1cee0a1e8aaef535f35`. Full trajectory CSV SHA256: `521d4ab9fd4876205f9e086d44275104678d1e42aacb2531cade89c92d2e4aa7`. Raw manifest SHA256: `389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc`.

## Claim impact

| Claim | Disposition | Qualifier |
|---|---|---|
| E1 completed the registered882x2 causal C training budget | supported | Full ordered supervision plus terminal loop counters/assertions, native exit0 and exact final checkpoint; not a per-frame independent NN replay. |
| Only8 quality/gain state tensors changed;257 executor/CNN state entries stayed equal to initial B | supported | All checkpoint tensors independently compared on CPU, including finiteness and exact frozen equality. |
| 398786 train utility labels use original dataset GT and signed executed-box IoU gains | supported | All labels recomputed; all882 raw train annotation files matched. Does not verify upstream pretrained-data history. |
| Teacher does not contaminate tracker templates/history and selected replay equals execution | needs_qualifier | Supported by frozen source and successful assertions at182182 supervised updates; no independent neural replay or per-frame history dump. |
| Initial full-dev evaluation launched with the final E1 checkpoint | supported | Only the two recorded costs0/0.05 at the intake timestamp; not current liveness or completion. |
| E1 final PR/NPR/SR, three-benchmark target attainment or robustness | unsupported | No completed E1 dev freeze/full-test metrics in audited intake; training loss is not held-out accuracy. |
| E1 is faster than baseline or proves2-3 novel effective mechanisms | unsupported | No paired system latency, matching-budget ablation or novelty evidence is generated by this audit. |
| Scale-candidate variant shares this completed C evidence | unsupported | Separate planned variant and training/evaluation lineage; these E1 receipts cannot be reused as its results. |

## Deterministic receipts and seal

Local checker: 1295 grouped checks; native exit 0. One SSH CPU probe: transport and joined native exit 0, empty stderr; original native session 54426, terminal chunk `efcc6d`. Exact raw receipts, every script and the failed default-Python startup attempt are retained in `C:/Users/gb/.codex_tmp/dual_rgbt_e1_full_c882_result_audit_20261017`.

Final unchanged input seal: 140 local files and 962 remote files; SHA256 `64a9564c90e87fef2f8c13d1fd9b22fba115d7fdaf0927a6603d997e4e2cd791`. Every exact input hash is included in EXPERIMENT_AUDIT.json and the private seal. Model/effort routing is not host-attested.

No new training M0 was requested: the executed source is unchanged, historical M0 remains admission-only evidence, and this audit newly checks the completed full-run artifacts. No commits, pushes, HANDOFF edits or experiment-input changes were made.

Continue the admitted evaluation. Final PR/NPR/SR, paired speed, ablation effectiveness, novelty and target attainment require their own completed evidence.
