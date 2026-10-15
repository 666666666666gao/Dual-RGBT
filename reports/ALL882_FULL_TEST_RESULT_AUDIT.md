# Original E0 C882 full LasHeR test audit

Generated: 2026-10-15T14:30:01.614587+00:00

**Overall WARN; deterministic verification PASS; blocking issues: 0.**

The two original test shards report native COMPLETE / child exit 0. The saved full-test result is reproducible exactly under the supplied legacy CADTrack-directory toolkit: **PR 74.74970272036138, NPR 71.06767196147948, SR 59.08900667625745**. Scope is **245 unique LasHeR test sequences / 220703 frames**, including **245 initializations and 220458 subsequent causal track calls**.

Requested reviewer: `gpt-6-astra` / `max`. Actual model and effort routing: **UNATTESTED**. Fresh native agent `/root/audit_all882_full_test_20261015`; `review_independence: same-family`, `acceptance_status: provisional`. This is not a cross-family acceptance claim.

## Reproduced result and identity

| Item | Verified value |
|---|---|
| Checkpoint | `ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c` |
| Frozen dev selection SHA-256 | `3a97b77fa25f873c18457565a6fd6bfe214e63d793567c936f2d448190cb8527` |
| Selection | final C882 round2 checkpoint; highest dev SR, then PR/NPR, then lower service mean; cost 0.0 |
| Seed / configuration | one trained checkpoint; training and inference seed2027; two disjoint shards of the same configuration |
| Original metric JSON SHA-256 | `cbf041a4f4c6a3dc8852b363e3e603892d6dfe39af78f7fc90c87c9bcb7cd047` |
| Original curve NPZ SHA-256 | `6931f94cb02ad2f10d396b78ab8e3ff29ec53ee3b6f85483926418b7a2a44f05` |
| Service mean / P95 / FPS | 113.81713857860639 ms / 178.66849230485968 ms / 8.786023023319661 |

Service time includes image decoding and initialization; model loading and per-frame result-file writing are excluded. FPS is frames divided by summed service times across the shards, not two-GPU wall-clock throughput. These unpaired observations do not prove a speedup. Evidence: `scripts/run_tracking.py:74`, `:78`, `:84`, `:99`, `:112`, `:124`; `scripts/score_tracking.py:72` and `:74`.

| Native task | PID / GPU | Sequences / frames | Actual completed time (UTC+08) | Child exit |
|---|---|---|---|---|
| `track_test_all882_shard0` | 478338 / 0 | 123 / 107898 | 2026-10-15T20:26:33.912335+08:00 | 0 |
| `track_test_all882_shard1` | 478426 / 1 | 122 / 112805 | 2026-10-15T20:40:09.070616+08:00 | 0 |

Native command, checkpoint/cost/shard arguments, PID and timestamps agree across launch and completed receipts. Config bytes match the launch hashes. This verifies already-collected child completion; it does not attest the original controller final state. Evidence: private `completion_observation.json:8` and `:36`; `scripts/run_task.py:28`–`:32`.

## A–F checks

### A_gt_provenance: WARN

Test sequences are enumerated from the frozen dataset manifest; the tracking entry reads dataset init.txt, and writes the float32 annotation array separately from predictions. All 245 serialized GT arrays / 220703 rows exactly equal the supplied local canonical LasHeR toolkit GT as float32, before and after round(0). The 882 train, 97 dev and 245 test name sets are unique and pairwise disjoint. No GT is derived from predictions. This review did not reread original remote init.txt/JPG bytes or authenticate the local toolkit against an official published release; provenance beyond the supplied local canonical copy is source/receipt traced.

Evidence: `scripts/run_tracking.py:25`; `scripts/run_tracking.py:30`; `scripts/run_tracking.py:34`; `scripts/run_tracking.py:81`; `scripts/run_tracking.py:112`; `scripts/run_tracking.py:113`; `scripts/score_tracking.py:35`; `scripts/score_tracking.py:48`; `scripts/score_tracking.py:51`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/dataset/lasher_dataset.py:13`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/dataset/basedataset.py:22`; `C:\Users\gb\.codex_tmp\dual_rgbt_all882_full_test_audit_20261015\verification.json:110`.

### B_score_normalization: PASS

The scorer uses fixed benchmark thresholds, per-sequence rates and sequence-equal arithmetic means multiplied by 100. NPR normalizes coordinates by GT width/height plus 1e-8, not by prediction extrema. PR is the <=20-pixel rate, NPR the <=0.20 rate, and SR the arithmetic mean of 21 strict-IoU-threshold success rates; SR is not trapezoidal integration. Internal score/sum entropy in model.py is a model input, not reported score normalization. Legacy special-GT/initialization semantics are disclosed separately.

Evidence: `scripts/score_tracking.py:58`; `scripts/score_tracking.py:63`; `scripts/score_tracking.py:70`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:241`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:272`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:285`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:315`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:327`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py:357`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/utils.py:118`; `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/utils.py:156`; `dual_rgbt/model.py:150`.

### C_results_and_identity: PASS

All 743 raw shard files equal their supplied sealed inventory before and after verification. Predictions are finite with positive widths/heights. Exact unique coverage is 245 sequences, 220703 frame predictions, 245 initializations and 220458 contiguous trace rows. Actual native tasks 478338/478426 are COMPLETE/child exit0, with launch/completion task/PID/argv/config identities bound. Stored curves (30135 values), all 735 per-sequence metrics and all aggregate accuracy/timing values exactly match two CPU reconstructions. The checkpoint hash and cost0 match the frozen dev-selected final-C882 configuration. No new checkpoint loading or live native-process observation was performed.

Evidence: `C:\Users\gb\.codex_tmp\dual_rgbt_all882_test_closed_20261015\inventory.json:5`; `C:\Users\gb\.codex_tmp\dual_rgbt_all882_test_closed_20261015\completion_observation.json:8`; `C:\Users\gb\.codex_tmp\dual_rgbt_all882_test_closed_20261015\completion_observation.json:36`; `C:\Users\gb\.codex_tmp\dual_rgbt_all882_test_closed_20261015\receipt.json:3`; `C:\Users\gb\.codex_tmp\dual_rgbt_all882_test_closed_20261015\lasher_all882_full_metrics.json:2`; `scripts/run_task.py:28`; `scripts/run_task.py:29`; `scripts/complete_all_train.py:149`; `scripts/complete_all_train.py:155`; `scripts/complete_all_train.py:166`; `scripts/run_tracking.py:69`; `scripts/run_tracking.py:74`; `C:\Users\gb\.codex_tmp\dual_rgbt_all882_full_test_audit_20261015\verification.json:2`.

### D_active_metric_and_causal_paths: PASS

The scorer instantiates and calls PR_LasHeR, NPR and SR_LasHeR, serializes their curves and derives the reported values. The collected CPU scorer transport records exit0, and the fresh verifier dynamically observed two calls per class while reproducing the stored output. Inference goes through run_with_crop_api to scripts/run_tracking.py. Only frame0 receives GT initialization; later calls are tracker.track(image). Search is centered on the previous predicted state; selected predictions update state/history, and periodic confidence-gated templates crop that predicted state. No per-frame GT forcing is present in the inspected call chain. This is source/trace evidence, without a new NN replay.

Evidence: `scripts/score_tracking.py:18`; `scripts/score_tracking.py:58`; `scripts/score_tracking.py:78`; `C:\Users\gb\.codex_tmp\dual_rgbt_all882_test_closed_20261015\cpu_score.transport.json:1`; `scripts/run_with_crop_api.py:18`; `scripts/run_with_crop_api.py:22`; `scripts/run_tracking.py:83`; `scripts/run_tracking.py:86`; `scripts/run_tracking.py:98`; `scripts/run_tracking.py:101`; `dual_rgbt/data.py:15`; `dual_rgbt/tracker.py:34`; `dual_rgbt/tracker.py:42`; `dual_rgbt/tracker.py:47`; `dual_rgbt/tracker.py:61`; `dual_rgbt/model.py:182`.

### E_scope_and_claims: WARN

This is one full LasHeR test of original E0 after 882-sequence/two-round C training, fixed final checkpoint and cost0 chosen by the preregistered three-dev-candidate rule; training/inference seed2027. It is not dev97, historical C20, candidate E1, RGBT210/RGBT234 completion, a multi-seed study, a module ablation or a speedup experiment. All three updated LasHeR targets remain unmet. Service timing was collected on two concurrent shards and is unpaired; mean/P95/service-FPS are descriptive only. The supplied HANDOFF/ISSUES snapshot still says collection is blocked by SSH; the later closed artifacts supersede that historical observation but do not attest the original controller final status.

Evidence: `reports/all_train_plan.json:5`; `scripts/run_tracking.py:54`; `scripts/run_tracking.py:55`; `scripts/complete_all_train.py:135`; `scripts/complete_all_train.py:149`; `C:\Users\gb\.codex_tmp\dual_rgbt_all882_test_closed_20261015\frozen_selection.json:2`; `C:\Users\gb\.codex_tmp\dual_rgbt_all882_test_closed_20261015\lasher_all882_full_metrics.json:2`; `scripts/score_tracking.py:72`; `scripts/score_tracking.py:74`; `HANDOFF.md:6`; `HANDOFF.md:12`; `HANDOFF.md:16`; `docs/ISSUES.md:529`.

### F_evaluation_type: PASS

Dataset-provided canonical GT scores causal saved model predictions. This is real_gt benchmark evaluation under the supplied legacy toolkit protocol, not model-derived proxy agreement. Source-derived causal behavior and real-GT numeric recomputation support this classification; no new model execution or external official-toolkit certification is claimed.

Evidence: `scripts/run_tracking.py:34`; `scripts/run_tracking.py:86`; `scripts/run_tracking.py:101`; `scripts/run_tracking.py:113`; `scripts/score_tracking.py:35`; `scripts/score_tracking.py:59`.

## Exact legacy semantics retained

1. Load both canonical GT and predictions as float32 and apply NumPy round(0), including its half-to-even rounding.
2. The toolkit replaces each first prediction by its first GT, but keeps that initialization frame in the denominator; it does not remove the frame.
3. After replacement, the toolkit truncates to GT length and carries forward the previous prediction for a nonpositive rounded width/height. Exact original lengths were separately checked and zero predictions required this repair.
4. A GT row with any component <=0 is assigned error/overlap -1. Hence it counts as a PR and NPR success at every threshold and an SR failure at every threshold; these rows remain in the denominator. There are 1260 such rows: 533 with nonpositive width/height, 727 position-only, including one initialization.
5. Centers use (width-1)/2 and (height-1)/2; IoU uses inclusive last coordinates (left+width-1, top+height-1) and +1 intersection widths/heights.
6. PR thresholds are linspace(0,50,51) with <=; NPR thresholds are linspace(0,.5,51) with <= and index20; SR thresholds are linspace(0,1,21) with strict >. All sequences have equal weight.
7. NumPy1.26.4 scalar float32 operands combined with Python -1, /2. and +1e-8 promote to float64. The independent checker preserves this scalar behavior after float32 loading/rounding, while retaining float32 area products. Production scorer/toolkit bytes were unchanged.

Evidence: toolkit `dataset/basedataset.py:19`–`:24`; `utils.py:31`, `:118`, `:122`, `:141`, `:162`; `metrics/metrics.py:246`–`:277`, `:290`–`:320`, `:332`–`:361`. All scorer lines 1–83 and the relevant toolkit/helper code were inspected. No scoring semantic or original input was changed.

## Deterministic verification and actual process completion

- 5331 grouped assertions passed. These contain full-array and full-list comparisons; the number is not presented as a count of individual numeric elements.
- 1286 input files were hash-sealed before and after, including all 743 raw shard files, 17 registered source files, the original request and corrected tracking entry, canonical GT/list and imported toolkit source dependencies. All remained unchanged, including between the failed R1 and completed R2.
- Exact coverage, finite positive-size predictions, float32 canonical GT equality, all contiguous trace frame indices, timing-array/CSV/progress binding and native receipt/config identities passed.
- Direct calls to the unchanged toolkit and a separate NumPy expression both reproduce all 30135 curve values: PR 245×51, NPR 245×51, SR 245×21. All 735 per-sequence metric values and six accuracy/timing aggregates equal the saved report exactly.
- Runtime profiling observed PR_LasHeR, NPR and SR_LasHeR twice each, once per shard. The verifier imported no torch or project neural module and made no neural/GPU/SSH/controller call.
- R2 actually completed with child exit0 and empty stderr: native session **41853**, launch chunk **a5bf58**, terminal chunk **90c233**. Its subprocess receipt is `verify_cpu_r2.transport.json`.

R1 is preserved: session99294 / terminal0939db / exit1, with original checker/stdout/stderr/transport. It stopped at the first-sequence NPR comparison because a float32 vector expression did not reproduce NumPy1.26 scalar promotion. The diagnostic found one threshold comparison discrepancy, maximum float32 error 1.703751868503689e-06, and exact scalar-versus-float64 agreement. R2 corrects only this checker assumption; production scorer/toolkit/prediction/GT files did not change. The path typo `run_tracking.py` in the initial request was corrected to the sealed `scripts/run_tracking.py` by the parent and independently confirmed in the plan; the correction is retained in `audit_context.json`.

## Freeze chronology and causal behavior

The controller scores all three complete dev candidates, selects by the registered rule, writes the frozen selection and copies it to the remote canonical location before launching test (`scripts/complete_all_train.py:149`–`:166`). The tracking entry reads and asserts that freeze against the checkpoint hash and cost before constructing or loading the model (`scripts/run_tracking.py:69`–`:75`). The same selection SHA is present in launch and completed snapshots. This supports freeze-before-inference through source order and artifact binding; the selection itself has no write-time field, so this is not an independent timestamped freeze attestation.

Subsequent frames call `tracker.track(image)` without GT (`scripts/run_tracking.py:98`–`:109`). The tracker crops around its previous predicted state, uses its own selected prediction as the next state, and updates templates only from that state when frame%5==0 and confidence>=0.65 (`dual_rgbt/tracker.py:34`–`:61`). The crop-wrapper and model paths do not add a per-frame GT input. No new neural replay was performed.

## Claim impact

| Metric | Actual test | User baseline (not rerun) | Difference, points | Target | Shortfall, points |
|---|---|---|---|---|---|
| PR | 74.749702720 | 72.74 | +2.009702720 | 78.74 | 3.990297280 |
| NPR | 71.067671961 | 69.16 | +1.907671961 | 75.16 | 4.092328039 |
| SR | 59.089006676 | 57.84 | +1.249006676 | 63.84 | 4.750993324 |

The full E0 test performance claim is supported with the stated protocol. All three targets remain unmet. Baseline differences above are descriptive arithmetic from the user-provided baseline and do not establish a matched rerun or significance. This audit provides no E1, RGBT210/RGBT234, three-module-effectiveness or speedup claim. The historical SSH-blocked HANDOFF/ISSUES statements should be followed by these later completed artifacts while retaining their original failed-observation context.

## Limitations and evidence files

- The local canonical toolkit/GT is hash-sealed, not newly authenticated against an official release; this report certifies exact reproduction of the supplied legacy protocol.
- Remote original annotations/images and checkpoint tensor bytes were not reread. Checkpoint linkage uses config/frozen hashes and the supplied earlier completed-training audit; its tensor-freeze findings are not newly reattested here.
- Freeze-before-inference is supported by controller write/copy-before-launch order, the active pre-model checkpoint/cost assertions and identical launch/completion freeze hashes. The selection JSON contains no standalone timestamp of its write; no independently timestamped freeze event is available.
- Native child COMPLETE/exit0 comes from already collected status/observation artifacts and their source implementation. No live process, controller-final-state or operating-system liveness claim is made.
- Causal search/template updates are source/trace checked. No neural replay or independent capture of in-memory tensors was performed.
- One checkpoint and seed2027 do not establish seed robustness, statistical significance, three module contributions or three-benchmark completion.
- Service mean/P95/FPS include decoding and initialization and exclude model loading and per-frame result-file writing; they are not paired with a baseline or candidate and do not prove a speedup.
- User-provided baseline metrics were not rerun. Numerical differences from them are descriptive only, not a newly matched baseline experiment.

Private verification directory: `C:/Users/gb/.codex_tmp/dual_rgbt_all882_full_test_audit_20261015`.

Primary deterministic evidence: `verification.json`, `recomputed_curves.npz`, `recomputed_per_sequence.json`, `verify_cpu_r1.py`, `verify_cpu_r2.py`, both attempt transports/stdout/stderr, `scalar_precision_diagnosis.json`, `host_native_completion.json`, and `final_input_seal.json`. The JSON audit includes every audited input SHA-256, all A–F evidence references, actual native completions and the complete claim/limitation record.

Only the requested two audit reports and private audit artifacts were written. No experiment source, input, controller, checkpoint or metric was changed; no commit or publication was performed.
