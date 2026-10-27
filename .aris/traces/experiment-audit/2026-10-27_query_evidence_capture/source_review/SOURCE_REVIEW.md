# Query evidence capture: independent source audit

**SOURCE_PASS; blocking_issues = 0.**

Fresh native experiment-audit review; requested gpt-6-astra/max, actual model/effort UNATTESTED. Review independence: same-family. Acceptance: provisional.

This source pass covers the exact hashes in SOURCE_REVIEW.json. No neural network, checkpoint read, remote query, training pause or FPS measurement was performed by the reviewer. The real-TRAIN parity result is still pending; the local TRAIN manifest is absent.

## gt_provenance: PASS

Capture accepts only the model, cost, image geometry, prior tracker box, route and executed box. It does not receive GT. Benchmark GT is loaded after causal sequence prediction. The admission indexes the registered train split and reads real RGB/TIR files; only frame-zero dataset box initializes tracking. Later frames receive images only.

Evidence: scripts/query_evidence_capture.py:15; scripts/query_evidence_capture.py:23; scripts/run_joint_candidate_evaluation.py:127; scripts/run_joint_candidate_evaluation.py:140; tools/check_query_evidence_capture.py:52; tools/check_query_evidence_capture.py:64; dual_rgbt/data.py:15.

## score_normalization: PASS

Recorded selected_gain is the raw predecision signed gain at request_index+1. Cost is the existing action token_cost, and utility is gain minus configured cost_weight times cost. Internal confidence-distribution features are not reported as benchmark accuracy. Existing official metric computation is unchanged.

Evidence: scripts/query_evidence_capture.py:45; scripts/query_evidence_capture.py:50; dual_rgbt_global/candidate_policy.py:43; dual_rgbt_global/candidate_policy.py:84; tools/joint_candidate_results.py:45.

## result_existence: PASS

No new accuracy, parity, FPS or completed-run result is claimed by this SOURCE review. The checker writes CAPTURE_REAL_TRAJECTORY_PASS only after all comparisons and branch checks. Runtime result validation is pending, and the local TRAIN manifest is absent.

Evidence: tools/check_query_evidence_capture.py:89; reports/QUERY_EVIDENCE_CAPTURE_ADMISSION_PLAN_20261027.json:150.

## dead_code: PASS

The runner instantiates capture, begins every tracked frame, finishes it and writes evidence CSV. verify_runs consumes every evidence CSV before scoring; score and freeze both use verify_runs. This is a reachable observation path, not an unused metric helper.

Evidence: scripts/run_joint_candidate_evaluation.py:82; scripts/run_joint_candidate_evaluation.py:127; scripts/run_joint_candidate_evaluation.py:130; scripts/run_joint_candidate_evaluation.py:138; scripts/joint_evaluation_contract.py:214; tools/joint_candidate_results.py:80.

## scope: PASS

Admission is bounded to three fixed TRAIN prefixes, eleven frames each, costs zero and one hundred, each plain and observed: sixty tracked frames per mode. High cost is an explicit STOP fixture, not a deployment choice. Both STOP and QUERY must occur. This establishes only execution equivalence if actually passed, never final-model quality or general performance.

Evidence: reports/QUERY_EVIDENCE_CAPTURE_ADMISSION_PLAN_20261027.json:138; tools/check_query_evidence_capture.py:55; tools/check_query_evidence_capture.py:85.

## observation_only: PASS

Each wrapper invokes the original bound method once, retains references to its result and returns the original output unchanged. No encoder, query, routing or tracker-state update is added. The original query reader consumes primary features without mutating retained box/score; gain selection and template/history transitions remain in the unchanged tracker.

Evidence: scripts/query_evidence_capture.py:17; scripts/query_evidence_capture.py:28; scripts/query_evidence_capture.py:34; dual_rgbt_global/candidate_tracker.py:31; dual_rgbt_global/candidate_tracker.py:38; dual_rgbt_evidence/policy.py:90.

## primary_hann_box_and_original_coordinates: PASS

Captured box is the actual primary box after its Hann-window score and cal_bbox decode. Capture saves search_geometry from the pre-track state, uses the same pixel_xywh conversion and execute_boxes margin-ten float32 clipping against the original image size. It does not substitute an unwindowed peak, crop-clipped GT or final query box.

Evidence: dual_rgbt_global/tracker.py:29; dual_rgbt/model.py:92; dual_rgbt/model.py:99; dual_rgbt_global/candidate_tracker.py:31; scripts/query_evidence_capture.py:24; scripts/query_evidence_capture.py:40; dual_rgbt_evidence/readout.py:49; dual_rgbt_evidence/regions.py:35.

## selected_gain_cost_and_stop: PASS

request_index+1 recovers the original selected gain index. action=kinds+1 recovers the same token cost used for selection. STOP is the explicit gain-zero/cost-zero entry, and finish asserts request zero plus np.array_equal of reconstructed executed primary and final box. CSV verification allows only the existing six-decimal prediction serialization tolerance for this relation.

Evidence: scripts/query_evidence_capture.py:45; scripts/query_evidence_capture.py:48; dual_rgbt_global/candidate_tracker.py:38; dual_rgbt_global/candidate_tracker.py:42; dual_rgbt_global/candidate_tracker.py:66; dual_rgbt/model.py:47.

## csv_completeness_and_sealing: PASS

Each sequence has one evidence row for every tracked frame (initialization intentionally excluded). Verification requires the exact sequence set and ordered complete frame IDs, finite positive primary boxes, finite evidence values, normalized token cost agreement, utility arithmetic and STOP zeros/box equality. Every evidence file is included in recursive run_artifacts_sha256; scored reports bind that map, and freeze recomputes it. Initial GT and prediction contracts remain unchanged.

Evidence: scripts/query_evidence_capture.py:8; scripts/run_joint_candidate_evaluation.py:112; scripts/run_joint_candidate_evaluation.py:120; scripts/joint_evaluation_contract.py:188; scripts/joint_evaluation_contract.py:213; scripts/joint_evaluation_contract.py:217; scripts/joint_evaluation_contract.py:237; tools/joint_candidate_results.py:109; tools/joint_candidate_results.py:135.

## existing_guards_and_training_unchanged: PASS

All ten existing contract functions except verify_runs are AST-identical to the byte-preserved pre-change source; verify_runs only adds evidence requirements. Complete-C/native success, full counts, exact tensor witness, provider-native command, smoke/reload, dev audit and freeze gates remain. All forty protected training plans equal Git HEAD; fifty-seven independently reconstructed source hashes match, with no changed-source intersection. Three protocol input/training/provider/metric/selection fields are unchanged. Current model-specific evaluation plans and their standard output directory are absent.

Evidence: scripts/joint_evaluation_contract.py:21; scripts/joint_evaluation_contract.py:32; scripts/joint_evaluation_contract.py:70; scripts/joint_evaluation_contract.py:94; scripts/joint_evaluation_contract.py:127; scripts/joint_evaluation_contract.py:258; scripts/joint_evaluation_contract.py:284; tools/register_query_evidence_capture.py:14; DETERMINISTIC_CHECKS.json.

## runtime_admission_design: PASS

The source gate binds the exact plan and source maps and verifies fixed input hashes. A native RUNNING parent and exact command/GPU zero are required. One strictly loaded historical candidate_b step-6250 non-sanity model is frozen and reused for plain then observed trajectories. Every box and non-time route is compared exactly; final templates are byte-hashed and history compared exactly for each sequence/cost. Both branches, unchanged state_dict and absent gradients are required. The result binds plan and checkpoint hashes. There is no optimizer, current-final checkpoint path, pause/kill operation or FPS measurement in this checker. Runtime success itself has not been asserted.

Evidence: tools/check_query_evidence_capture.py:26; tools/check_query_evidence_capture.py:33; tools/check_query_evidence_capture.py:45; tools/check_query_evidence_capture.py:55; tools/check_query_evidence_capture.py:80; tools/check_query_evidence_capture.py:85; tools/check_query_evidence_capture.py:90.

## Classification and claim limits

Benchmark scoring remains real_gt. The admission is execution-equivalence only on real TRAIN images; comparing two causal outputs is not being represented as accuracy.

- No new benchmark accuracy, query benefit or FPS result is established by this review.
- Primary-vs-final boxes support same-state immediate executed QUERY IoU comparison only after joining official GT offline.
- STOP missed opportunity, unexecuted alternatives and long-horizon counterfactual trajectories remain unobserved.
- High-cost admission is a branch-coverage fixture, not a selected deployment cost.
- Unchanged incidental evaluation timings are not dedicated profiling or evidence of speed improvement.

## Deterministic checks

- Five changed/added Python files compile without importing or executing NN code.
- Forty protected training plans equal Git HEAD byte for byte; all fifty-seven reconstructed training-source hashes match.
- All admission and protocol source hashes match. Only evidence source bindings, evidence_schema and appended scope text changed in the protocols.
- Complete-C/admission/dev-freeze/source/native guard functions remain unchanged. The three live guards remain byte-identical to their pre-change copies.
- No current model-specific evaluation plan was found; reports/joint_candidate_evaluation is absent.

## Exact bindings

- Admission plan: `377a5f449481e876c737b01a91e197befdbfc7ef655fbb5e0f317399b3257ba2`.
- JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json: `899c14feddceaead8973067e5fe34fdc84284804e6ea389c92bf84d77ad932d2`.
- VRAM18_EVALUATION_PROTOCOL_20261027.json: `dbb353a34bf9d22a2473bec8d521d24b46267703978307ac9fc3a2d0862faf90`.
- SERVER25_JOINT_EVALUATION_PROTOCOL_20261027.json: `9ba535e83faa83cc8b0667d753513570b0c9034f853e9a24d21a4310bb879a5d`.

Four separate guard payloads are under guard_payloads/. They have not been installed into reports/. The live guards are intentionally stale until the executor verifies and installs the exact matching payloads.

## Remaining execution requirements

- Install only payloads whose current plan/protocol and complete source maps still match; reviewer did not overwrite live guards.
- The exact planned real-TRAIN plain/observed runtime admission still needs to pass on its registered provider.
- Existing completed-C, native ownership/capacity, smoke, dev-result audit and frozen-selection gates remain necessary for future model evaluations.

No source blocker or requested implementation change was found.
