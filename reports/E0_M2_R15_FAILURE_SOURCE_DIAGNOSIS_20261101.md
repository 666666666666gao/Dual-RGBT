# R15 M2 Failure Source Diagnosis

Date: 2026-11-01. Target: host26, `/data2/gb/Dual-RGBT`.

Scope: supplemental engineering diagnosis of the existing R15 integration and a proposed TRAIN experiment sequence. This document gives no SOURCE_PASS, deployment, launch, or scientific acceptance. Existing admission reports and run evidence are preserved.

Attribution: `actual_model=UNATTESTED`; `review_independence=same-family`; `acceptance_status=provisional`; `scientific_result_audit_accepted=false`. The requested `gpt-6-astra` route is unavailable in the exposed tool schema. No Astra execution or independent-family audit is claimed; this does not replace the required formal experiment audit.

## Findings

| Proposed mechanism | Source finding | Causal conclusion for R15 full TEST |
| --- | --- | --- |
| Current E0 box lacks explicit preservation/scoring | No-match retains E0, but the current E0 result is not explicitly scored against generated candidates; no-match labels describe candidate-set absence. | A concrete decision/label semantics gap. Harmful replacement is plausible; its contribution has not been measured. |
| Replacement templates use preceding E0 confidence | The branch uses the same-frame E0 route confidence, computed before replacement, for a different box. | This branch is unreachable under the reviewed full TEST cadence. It cannot explain this regression through same-frame replacement template writes. |
| M2 lacks trained M1 failure gating | R15 constructs direct E0 plus M2; scheduled calls do not depend on M1 state. | A concrete missing integration stage. Gating may reduce harmful interventions, but improvement is unproven. |

Old E0 confidence does enter replacement history on positive selections. Later ordinary E0 template writes can also depend indirectly on the changed trajectory. These are distinct from the unreachable replacement-template branch; neither is an established cause of the measured regression.

## Observed Evidence

The locally archived RGBT210 score records PR/SR `81.768710/61.201206`, versus frozen E0 `88.828508/66.124608`, a difference of `-7.059798/-4.923403` percentage points. The archived receipt records 210 sequences, 104706 frames, 4284 M2 invocations and 4031 positive selections; 253 calls retained E0 through no-match. The 94.0943% replacement frequency is not matching accuracy.

These are existing exploratory TEST observations, not newly scored results or independent validation. They motivate diagnosis but must not select thresholds, seeds, checkpoints, or a new policy. The earlier M1 guard result is a different ablation, not evidence that the complete M1/M2/M3 integration works. See [existing findings](C:/Users/gb/projects/Dual-RGBT/reports/E0_M2_R15_RGBT210_FULL_FINDINGS_20261101.md:10).

The accepted TRAIN metadata reports 706 fit and 176 component-holdout sequences, 18711 candidate sets in total. Epoch9 component-holdout supervised positive top1 is 79.623188%; correct candidate picks are 2685/3318 and correct no-match decisions only 62/132. Those aggregates do not identify reliable-target false replacement, lost-target recovery, or closed-loop tracking gain. Upstream E0 saw this component holdout. See [metadata and limitations](C:/Users/gb/projects/Dual-RGBT/reports/E0_M2_TRAIN_DIAGNOSTIC_METADATA_20261101.md:19).

## Cadence Proof

The complete source chain supports the qualification in the latest findings:

1. Full evaluation supplies `list(range(frames))`, calls `set_sequence`, initializes once per sequence, and invokes `track` exactly once for each image index from 1 through `frames-1`: [evaluator](C:/Users/gb/projects/Dual-RGBT/scripts/eval_e0_m2.py:251).
2. `sampled_m2_frames` filters out frame0 and takes `[::25]`, returning a tuple. With that full input, the tuple is `(1, 26, 51, 76, ...)`, truncated at the sequence boundary. `set_sequence` stores it unchanged: [sampler](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/e0_m2_online.py:10), [assignment](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/e0_m2_online.py:63).
3. Direct E0 initializes `self.frame=0` and increments it exactly once per `track`: [initialization](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/tracker.py:25), [increment](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/tracker.py:44). M2 does not otherwise advance or reset that counter.
4. The wrapper computes `next_frame=base.frame+1`, conditionally snapshots templates, calls `base.track` once, then tests the resulting `base.frame` against that same tuple: [wrapper](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/e0_m2_online.py:83). Thus snapshot timing and invocation timing agree; there is no off-by-one shift to frames25 or50.
5. Every call frame is `1+25k`, hence congruent to1 modulo5. The replacement write requires `frame % 5 == 0`: [condition](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/e0_m2_online.py:110). The two sets never intersect. Ordinary E0's same-frame write also requires divisibility by5: [base condition](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/tracker.py:47).

Consequently, on the reviewed full TEST path, positive selection restores an unchanged pre-E0 template snapshot and then replaces the box/history; it does not execute the replacement crop/write. Later divisible-by5 E0 frames can write templates using their own newly computed confidence and their then-current state. Such later writes are not evidence of the dormant branch executing.

TRAIN smoke uses the manifest's `valid_frames`, not necessarily all consecutive frames. Filtering and striding an irregular list need not produce `1 mod 5`; do not extend the full TEST nonintersection proof to arbitrary TRAIN schedules.

Verification performed here: extracted only the sampler function with Python AST, without importing torch or project models. It returned `(1,26,51,76,101)` for 103 frames; arithmetic checks for frame counts1..2000 found zero divisible-by5 invocation frames. This is a source/arithmetic check, not a runtime trace replay. The server26 benchmark manifest is absent locally, so actual per-sequence invocation totals were not reconstructed; 4284 is taken from the archived receipt.

## Decision And Label Semantics

R15 first runs E0 and obtains its current box and route. Candidate generation receives the previous state and fixed initial templates, not that current E0 box: [online ordering](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/e0_m2_online.py:83). The collector executes one previous-state crop, nine global crops and two refinements, with two modality primary calls per crop: [candidate generation](C:/Users/gb/projects/Dual-RGBT/scripts/collect_e0_identity_train.py:93). Its 24 generated boxes can coincidentally overlap or equal E0's result, but there is no guaranteed current-E0 entry or score.

Argmax index0 returns the E0 result and state as produced by `base.track`; positive indices unconditionally replace `base.state`: [selection](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/e0_m2_online.py:100). Therefore "there is no retain action" would be inaccurate. The actual gap is that retention is represented only by a candidate-set no-match decision.

The labels make the distinction explicit: if any observed candidate has IoU>=0.5, label0 is negative regardless of whether current E0 is already correct or better. No-match becomes positive when there is observed evidence and every candidate IoU is below0.1. Ambiguous cases remain unknown: [labels](C:/Users/gb/projects/Dual-RGBT/scripts/collect_e0_identity_train.py:68). The head's no-match logit is a learned scalar, whereas candidate scores depend on anchor/ROI features: [head](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/e0_identity.py:47).

The training loss rewards positive candidate mass and ignores unknown labels. Best checkpoint selection uses aggregate component-holdout supervised positive top1, not reliable-state protection or closed-loop recovery: [loss](C:/Users/gb/projects/Dual-RGBT/scripts/train_e0_identity.py:21), [selection](C:/Users/gb/projects/Dual-RGBT/scripts/train_e0_identity.py:112). These are coherent identity-set objectives but do not establish a replacement advantage over E0.

Minimal implication: retain the complete current E0 result when the failure gate is closed or identity rejects. Do not relabel no-match as a trained KEEP decision. If explicit E0-versus-candidate ranking is later needed, collect its real descriptor/coverage in the correct crop coordinates and define TRAIN labels for preservation/relative replacement quality. Merely appending E0 to a multi-positive pool does not prioritize it over another positive box. Although the head supports variable candidate counts, the current collector fixes24 candidates/25 labels; changing the set requires corresponding collection/verifier/label review.

## State And Confidence Provenance

The problematic confidence is the current frame's original E0 route confidence, not necessarily the previous frame's confidence. E0 computes it before M2 chooses a replacement. The wrapper then reuses it with replacement motion/size, the original E0 modality/action, and E0 query-counter semantics: [replacement state/history](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/e0_m2_online.py:108).

This produces mixed-provenance history: confidence describes E0's proposed decision, while displacement and scale describe M2's selected box. The returned route remains E0's route. Subsequent E0 prediction consumes this history in modality quality and action-gain context: [quality input](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/model.py:50), [gain context](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/model.py:150), [prediction](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/model.py:182). That dataflow is factual; its effect on R15 PR/SR has not been isolated.

The replacement-template use of that confidence is a latent provenance defect for schedules intersecting write frames, but is dormant for these full runs. Do not substitute an M2 logit into the E0 confidence threshold: the scores have different training semantics. Do not invent a confidence value or claim that retaining old history is semantically solved. Any changed replacement-history contract needs a named TRAIN ablation and closed-loop measurement.

## Missing M1 And M3 Integration

R15 constructs `Tracker` and `E0CandidateIdentity` only: [constructor](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/e0_m2_online.py:52). Invocation depends solely on sampled-frame membership. There is no reliable/uncertain/lost classifier in its decision path.

Reusable M1 code already exists. `M1TemplateGuard` observes the original E0 prediction, normalizes914 features using the trained checkpoint, classifies state, and optionally rolls back only a dynamic-template write. It returns the unchanged E0 current box/route: [guard](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/e0_m1_guard.py:23). It is not an existing M2 invocation gate. M1 labels describe final localization IoU, not identity or recoverability: [targets](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/e0_reliability.py:10). Predicted lost is not oracle lost.

Reuse the guard/observer around direct E0, then consume its record before candidate intervention. An outer `M1TemplateGuard(E0M2Tracker(...))` is not sufficient: the guard expects direct frame/template attributes, and its observation must refer to E0 before M2 changes state. Preserve observer-hook ownership and distinguish template rollback from box replacement. The existing [M1 evaluator](C:/Users/gb/projects/Dual-RGBT/scripts/eval_e0_m1.py:132) is the integration reference.

M1 collection sampled valid divisible-by5 frames: [collection cadence](C:/Users/gb/projects/Dual-RGBT/scripts/collect_e0_reliability_train.py:119). M2 samples every25th positive valid frame. The caches cannot be blindly joined by row, and full-style M2 frames are outside that M1 collection cadence. A new causal TRAIN observation at candidate-decision frames must measure that distribution shift.

The online wrapper retains an initial anchor and initial templates, plus E0's recurrent history/dynamic template; these are forms of retained state. It has no temporal candidate identity association or M3 reliable-memory update/retrieval stage. Adding a fixed anchor is not completion of `protected E0 -> M1 failure -> M2 identity -> M3 memory`.

## Minimal TRAIN Sequence

Everything below is proposed, not executed or admitted by this report. Freeze the protected parent, the existing M1 checkpoint selected by its original TRAIN plan, and epoch9 M2 for the initial diagnosis. Preserve sequence-disjoint706/176 membership, cost0 and disabled TF32. Use TRAIN GT only after causal decisions for labels and analysis; keep TEST out of training, selection and threshold fitting.

1. **Restore evidence access before claiming replay.** Existing metadata says none of the882 accepted M2 NPZ paths are present on26. This review did not re-enumerate remote files or contact25. An authorized transfer of original bytes with acceptance-hash verification is preferable; do not regenerate the full collection merely to replace missing files. Without the cache, report state-stratified replay as unavailable. Metadata alone supplies neither candidate-level decisions nor914-dimensional M1 features.
2. **Establish protected E0 and observation parity.** Reuse direct `Tracker`, `E0Observation`, `load_reliability`, and existing paired-state checks. With M2 intervention disabled and M1 in observe mode, compare boxes, non-timing routes, both templates, history, frame and query counter against E0. Observe M1 at planned M2 decision frames. Separately retain the existing M1 template-guard mode as a named ablation; its intentional rollback must not be mistaken for parity failure. Any necessary fresh bounded TRAIN observation is a new provenance-bearing run, not the recovered original collection.
3. **Measure M1-gated identity before changing weights.** Predeclare lost argmax as the initial gate; preserve reliable and uncertain E0 decisions. Initially keep the existing candidate cadence, anchor, head and argmax rejection unchanged. Run a shadow pass first: record candidate decisions without committing boxes/templates/history. Then compare the existing ungated policy with the lost-gated policy in matched TRAIN closed loops, with M1 template-guard mode held fixed. Add the guard as a separate comparison only when needed to separate gate effects from template protection. Always retain the current E0 decision packet as the rejection path; this is protection, not proof that accepted candidates are correct.
4. **Resolve M2 acceptance and history using TRAIN evidence.** Report reliable-target harm, lost-target recovery, no-match confusion and candidate availability by actual E0 state before deciding that head retraining is necessary. For positive replacements, explicitly review the state/history contract; compare the current mixed-history rule with a stated alternative under identical frozen-state shadow decisions before separate closed loops. Prevent same-frame recovery-template writes until candidate-specific acceptance has been validated; this cannot be credited as fixing the dormant branch in existing full runs. If preservation ranking needs learning, reuse ROI descriptors/head/loss where appropriate, but add real E0 evidence and revised TRAIN labels rather than treating no-match as KEEP. Train only the deficient component and select using predeclared TRAIN protection/recovery/rejection criteria.
5. **Introduce M3 only after failure/identity acceptance is adequate.** Reuse the initial anchor, ROI/projected identity features and trained M1 record for a bounded causal prototype-memory experiment. Keep the initial anchor immutable; admit additional memory only through a separately specified reliable-write rule, never from future frames or GT. Treat retrieval, recovery acceptance and memory writes as separate decisions. Compare the accepted M1+M2 configuration with/without this one memory change, including a no-write control. Temporal associations and memory admission must be specified and reviewed; no existing M3 implementation or completed M3 training is claimed.

For every measured TRAIN comparison, report denominators and per-sequence paired changes: eligible failure frames, actual invocations/replacements, observed positive-candidate coverage, selected-box IoU versus current E0, reliable-to-worse replacements, lost-to-recovered replacements, uncertain cases, correct/false no-match, and subsequent recovery persistence. Stratify by both GT-defined localization state for diagnosis and predicted M1 state for the executable gate; GT must never drive the deployed decision. Distinguish candidate absence from wrong selection.

Frozen E0 replay measures one-step decisions at E0 states. Once replacements alter state, crops/history and future observations diverge; only a separately reviewed closed loop tests the changed policy. A31-frame/93-total-frame smoke tests execution and bindings, not tracking improvement. Existing CPU admission tests likewise do not establish policy quality.

The existing M2 trainer's main entrypoint asserts `target_host == '2025'` and CUDA: [admission](C:/Users/gb/projects/Dual-RGBT/scripts/train_e0_identity.py:248). Reusing its loss/model/data functions is appropriate; invoking it unchanged on26 is not. Any future26 experiment needs its own reviewed plan/entrypoint, exact source/input/checkpoint and absolute argv binding, new task/run IDs, and separate launch authorization. This document launches nothing; any later compute remains within the user's host26 GPU0/1 scope.

## Provenance And Verification Limits

Local source bytes for the following six files exactly match their entries in the archived R15 RGBT210 runtime plan. This binds the diagnosis to that recorded source map, not to a fresh remote filesystem verification.

| Source | SHA256 |
| --- | --- |
| `dual_rgbt/e0_m2_online.py` | `1e4e530860148d4d40f6d8771e271201239d3e4a8ccc8c3d893c91c46f6f5db1` |
| `scripts/eval_e0_m2.py` | `4bc7cb5ec52103773ef7979fe620f19d317c177d1aac60eb61b8587d233eb815` |
| `dual_rgbt/tracker.py` | `406927019ffdcdbc8546755d9f7bff0859aac20766116e4a70f8c89132694fff` |
| `dual_rgbt/model.py` | `c7fae6f893797edd10dcb6e625bb304b5a637c80ed1ac404f58d3d223d1b9f7e` |
| `scripts/collect_e0_identity_train.py` | `78009e849c4618416b680e451d0d2f6d81754ddbbf00f13fc343524596aac9eb` |
| `dual_rgbt/e0_identity.py` | `89b15dea17514e5516dfe43e76295662ae47da8f98ea590e3fc5565925278669` |

Additional local module bytes inspected for the proposed reuse:

| Source | SHA256 |
| --- | --- |
| `scripts/train_e0_identity.py` | `c147e895c9e0c2f94b9ea7755cd215ab685cf72c3b6f773385ea37290cc4019f` |
| `dual_rgbt/e0_m1_guard.py` | `e734935c1591077d96157eafc41b7c313fc9ef5d670b68d0a578887bc2eb39b5` |
| `dual_rgbt/e0_observation.py` | `3fe991f78db09e1a065b9fc1a7d6ed01940dae934bd57968c156232f6a089bdb` |
| `dual_rgbt/e0_reliability.py` | `e179851ee5c35570f86e2d4712463fe4c64048f1847986a9472cc6ec8840ca21` |
| `scripts/collect_e0_reliability_train.py` | `738b17dad5a09cb3b284b5c56b94908719e34fcf569d88edcfd06ccde1921734` |
| `scripts/eval_e0_m1.py` | `f27f564cbd0283a59df1dbeb1f52c991b14ec41c0216b10d0f6fabfec40b35a4` |
| `tests/test_e0_m2_online.py` | `c11ff5a6bc3811244fed3f924369aec4cb1f88e67038dd4eff93c1a838c09dd9` |

Existing local artifact hashes re-read in this diagnosis:

| Artifact | SHA256 |
| --- | --- |
| R15 RGBT210 archived `runtime/rgbt210/plan.json` | `079d269c03c8f407ebe9220de4953fbf75c4e51f4b64845727655d8be20bfff9` |
| Archived `runtime/rgbt210/config.json` | `8f99b3a3c03c5eaf551bba6873043e0edccd99dd15d6b7dda80de58be05b943e` |
| Archived `runtime/rgbt210/COMPLETE.json` | `6420388b22ad3225feda7e69f70cf899bfc775f19f7f6c5690c978d571da6245` |
| Archived `runtime/rgbt210/status.json` | `d76eae793065e5c73b76183fe595c61a92ab60935c266cb97361840a3974fc02` |
| R15 `rgbt210.json` score | `8fb87bd7cf3e1e33fd422bdbff0ad530fdb73dfa5942d4e64c80900c7d623c20` |
| E0 `rgbt210_control.json` score | `cb280cf315bb7b740920ebe7d4ad34b84144e93386df8158b052dd27082cd88a` |
| M1 `rgbt210_guard.json` score | `f3820e47f2561dfce3e613494eee72478bdcd962c0583415c74f41275e46c838` |
| `E0_M2_TRAIN_DIAGNOSTIC_METADATA_20261101.md` | `5928621522a2df2bfe1eeca4c394b5be4f10867a98fbc08b0423443ca83916b3` |
| `E0_M2_R15_RGBT210_FULL_FINDINGS_20261101.md` | `b862f643d87bece7045ba917f6ac1572de386136defeeff6a51c4b29886e2aaf` |

The R15 artifact paths above are under `C:/Users/gb/projects/Dual-RGBT/reports/e0_m2_online_scores_r15_20261101`; control/guard scores are under `reports/e0_m1_full_scores_r1_20261101` at the same local root. Parent checkpoint identity is `ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c`; M2 checkpoint identity is `7e7df5d25d3e8667fe38a4ca8f6ec3662308ffcb52a09bd16aecaab59a9cb4be`, as bound by the archived plan/receipt. Checkpoint tensors were not loaded in this diagnosis.

Only this new Markdown file was written. No source/existing report was edited, no scientific audit or full artifact-tree revalidation was performed, no model/optimizer/scorer was run, and no GPU, remote host, other chat or live task was queried. Current status of other R15 runs is not inferred from historical documents. No state-stratified replay, new TRAIN result, or causal attribution of the regression is claimed.
