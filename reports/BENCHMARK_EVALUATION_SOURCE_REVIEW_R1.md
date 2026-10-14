# Three-benchmark evaluation SOURCE review — R1

**SOURCE_FAIL — 4 blocking issues.** The exact plan is `f4313e78413811251f09b6d824313c838d2cebd3defb5615aaf3e1480385773c`. This verdict does not admit neural smoke or formal evaluation. Requested reviewer: `gpt-6-astra` / `max`; actual routing **UNATTESTED**. Fresh context, same-family, provisional.

## Blocking findings

1. **Full-C training identity/configuration is incomplete.** `scripts/run_benchmark_tracking.py:45` checks counts, selected config fields and native COMPLETE/exit0, but the unchanged gate accepts an unrelated native task/command and GPU7 for either model. E1 additionally accepts empty source/input maps, another factory checkpoint, K1 configuration, 882 duplicate names, and `COMPLETE.status=FAILED` when its checked numbers remain unchanged. The runner never compares the on-disk training config with the checkpoint config. These are original-function fixtures with a JSON-only load stub, not evidence of corruption in the real training. Bind expected training task/GPU/argv/run, actual E0/E1 config schemas, exact E1 source/input/factory/config and unique full882 membership. Reuse E0's existing `scripts/audit_all_trajectories.py` artifact for its source/input/order/checkpoint evidence.

2. **The test freeze is only self-declared.** At `scripts/run_benchmark_tracking.py:143`, a file saying LasHeR_dev97/54646 with a matching checkpoint/config is accepted without any dev result receipt. The unchanged guard also accepts a rewritten cost on the next invocation, an unplanned cost0.77, and different LasHeR/RGBT234 configurations. The start/end SHA only protects one process interval (`:243`). Create the freeze from bound complete dev results for the planned candidates and final checkpoint, record the deterministic choice, and bind the one-time selection SHA into test launch/admission and scoring. Enforce the current plan's common configuration across all three benchmarks.

3. **Scoring does not verify manifest registration.** `tools/score_benchmark_tracking.py:39` compares the supplied manifest with config copies, without comparing it to the sealed plan's manifest SHA/counts. The separate fresh metric reviewer invoked the unchanged scorer and original plan with an artificial LasHeR dev97/291-frame manifest; it exited0 despite the registered dev97/54646 input. I inspected that fixture source and verified its raw stream/output hashes. Bind the manifest through `datasets`, `input_sha256` and `split_sizes` before scoring.

4. **Scoring can accept unrelated execution/model metadata.** At `tools/score_benchmark_tracking.py:70`, matching native/config command strings are enough; both can contain an unrelated command. Actual CPU corruption cases also accepted an unregistered variant and different parameter counts across shards. No unrelated project was accessed or executed. Check the expected tracking argv/task/GPU/variant/checkpoint/run and invariant model metadata rather than only comparing duplicated values. The inference runner already has a stronger literal parent/argv check at `scripts/run_benchmark_tracking.py:132`; preserve and carry that evidence into terminal scoring.

## Checks that passed

All **43 actual source bodies** match the plan and preparation. The previous **38 C sources and 37 B sources** are preserved. All43 parse with Python3.7 grammar; the actual CPU test interpreter was Python3.12.12/NumPy1.26.4. The unchanged warm SSH ledger was read; no new environment readiness claim is made. The original plan and all43 source bodies are preserved in a private snapshot before any repair.

The original train/dev/test name sets are disjoint **882/97/245**. Both training plans use the same seed2027 train order; independently recomputed full budget is **917106 tracked frames,182182 updates,1764 initializations**. The five real manifest hashes and every GT array were checked against the archived LasHeR manifest or provided toolkit files:

| Input | Sequences | Frames |
|---|---:|---:|
| LasHeR train smoke input | 1 | 176 |
| LasHeR dev | 97 | 54646 |
| LasHeR test | 245 | 220703 |
| RGBT210 test | 210 | 104706 |
| RGBT234 test | 234 | 116650 |

RGBT210 uses the original common `init.txt` boxes. RGBT234 preserves separate visible/infrared GT and initializes from visible frame0. All initial boxes match their original first annotation; zeros are retained. Actual supplied GT values happen to be integral, but the exporters/loaders preserve float32 without pre-rounding. The metric reviewer additionally used fractional artificial cases to distinguish rawGT from rounded predictions.

The preparation archive contains exactly its two source files and two private annotation exports with matching bytes. Captured export/SCP/remote receipts are exit0 with empty stderr; remote stdout equals the saved report. The helper's rendered remote program reconstructs byte for byte. **Image assurance is only first/last PIL header opens and size comparisons.** `Image.open(...).size` does not decode all pixels; the existing phrase “first/last pairs decoded” overstates the evidence, and the endpoint dimensions were not emitted. This reviewer did not access remote images.

E0 uses the existing `DualRGBT` and exact `dual_rgbt.tracker.Tracker`; its admitted crop alias is set before that tracker imports `crop_pair`. The early `cpu_state` helper imports the global tracker, not the E0 tracker. E1 uses the existing strict `build_candidate_policy`/`CandidateEvidenceTracker`. No fallback constructor, partial net loading or new architecture is introduced. Both instantiate from protected pretrained/executor identities then strictly load the selected net.

The real inference loop passes only the image into `track`; `read_pair` uses image paths only. Only frame0 GT enters initialization, and later GT is written after prediction (`scripts/run_benchmark_tracking.py:183`). Both real trackers propagate predictions/history and update templates from predicted state. Main and reload are `no_grad`; `requires_grad_(False)` and full before/after state comparisons are present. There is no optimizer/backward call in the inference driver. Train smoke is exactly11frames including initialization,10 tracked predictions. Its strict clone restores bbox/templates/history/window/frame/query age and compares bbox, every route field except time, and history/templates on the last frame. That is a source contract; no actual numerical smoke was run.

The scorer keeps the original toolkit math: LasHeR PR/NPR/SR, RGBT210 PR/SR, and RGBT234 per-frame minimum center error / maximum IoU over dualGT for MPR/MSR. Predictions use original integer rounding; dualGT stays raw float32. PR/MPR uses20pixels, NPR0.2, and SR/MSR the mean21-threshold curve, with sequence macro averaging. The original first-frame and LasHeR nonpositive-GT conventions remain unchanged. There is no prediction-statistic score normalization. Prediction/GT shapes, savedGT identity, shard indices, finite positive boxes and complete positive frame timing are checked relative to the supplied manifest, which is why finding3 matters.

## Evidence and limits

The own CPU suite completed **786 checks,18 checkpoint cases and11 runner-admission cases**, native exit0, empty stderr. Its accepted invalid fixtures substantiate findings1–2; “checks passed” does not mean those gates passed review. The separate scorer suite has24 cases:20 correct rejections and4 invalid acceptances. Its arithmetic goldens and raw receipts were inspected; it was not rerun. All112 original read inputs remained unchanged during that suite. The author began revisions afterward; this R1 report uses preserved original plan/source bytes and does not judge the revised version. Evidence is in private ledger `dual_rgbt_benchmark_source_review_20261015_d97c41a6`; exact hashes are in the JSON companion.

The first private checker attempt incorrectly assumed matching RGB/TIR basenames; real LasHeR uses v/i prefixes. Only that checker was corrected, preserving its error streams. No audited source, plan, original annotation, checkpoint or existing report was changed.

Full B, factory and protected CAD checkpoint tensors are unavailable locally; their identities retain captured native hash-assertion qualification. B completion and C M0 are prior provenance, not formal C completion or current E0 status. This review performed **zero torch imports, NN calls, GPU queries, SSH calls and current E0 process/status queries**. No benchmark accuracy, complete-system speed gain, identity memory, query acceptance, reliable template gate or three effective mechanisms is established. Only this new report pair was written in the repository.
