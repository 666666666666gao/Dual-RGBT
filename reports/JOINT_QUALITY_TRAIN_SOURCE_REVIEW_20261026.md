# Joint preview/quality TRAIN source review

**SOURCE_PASS; 0 blocking issues.** Fresh delegated source review under run-experiment and experiment-audit. Requested reviewer: gpt-6-astra / max; actual route: **UNATTESTED**. Same-family review; semantic acceptance is provisional.

- Plan SHA-256: `014101511170dc2742c1cca4662035bae7a2b106f0f6e38cc10310eda090d42f`.
- New trainer SHA-256: `a19e7ae96787250ded975e89f2aaeb87a0f9cb3ab7b631e4fe9294ba4d0c2331`.
- All 49 local source hashes/AST parses pass. The inherited 48 exactly match the approved server26 plan and installed files. All 23 bound archived inputs pass SHA-256 verification.

The delta is limited to adapting the existing preview CNN, adding relative-quality ranking, separate preview/head learning rates, expected state checks, loss logging and native wrapper identity. No model architecture, module, deployment interface, teacher or candidate definition changes. Original model/tracker sources are unchanged.

The new rank term is `abs(d) * relu(abs(d) - sign(d) * p)`, with GT quality gap `d` and predicted gap `p`. CPU autograd verifies that descent raises the better modality, reverses correctly for TIR, gives zero rank loss/gradient for ties, and stops after the requested margin. The small-gap case is downweighted. Eight rank cases pass. The old gain/STOP body is AST-identical; six comparisons preserve exact values and gradients, including selected-extra, neutral and threshold cases. Evidence: trainer lines88-115.

The training preview at line191 is computed outside `no_grad`. Local pools and template descriptors reach quality and gain; broad CNN features reach gain through `grid_sample` (model.py:50, global/policy.py:19, candidate_policy.py:47). No detach cuts these paths. `eval()` keeps the frozen model behavior stable without disabling gradients; Preview contains Conv2d/GELU. Archived B has265 state tensors; the selected existing6 preview +4 quality +4 gain tensors leave251 frozen. Actual nonzero CNN gradients and14 changed tensors are required by the forthcoming real sanity, not inferred from scalar tests.

Both runs strictly load the same native-complete B6250 checkpoint, reseed2027 and create fresh AdamW. Sanity weights and old optimizer state are never loaded into formal. Preview/head groups are disjoint at1e-5/1e-4; the CPU scalar optimizer starts empty and reaches four updates correctly. Archived B config/lineage matches its bound plan, with8 historical optimizer states at6250. Evidence: trainer lines269-278,303-323.

The manifest independently yields **882 sequences ×2 rounds,917106 tracked frames,182182 updates,1764 initializations**. The first-sequence sanity yields20 tracked frames/four updates/one initialization. The97 dev sequence names are disjoint. These are verified planned counts, not newly executed training results.

Current-frame GT enters only after `track(image)` advances state; the teacher uses copied pre-track state/templates/history and unchanged pre-update weights. Selected replay retains route/request/kind/rank/geometry/executed-box equality; teacher preserves tracker state. The corresponding helpers are AST-identical to RankSTOP. Existing12-field strict-reload and four-request legacy probes remain in sanity. Evidence: trainer lines173-197,128-138; train_candidate_evidence_policy.py:49-109.

The native identity gate binds the running wrapper parent PID, task, GPU0 and exact argv. One valid and five rejection fixtures pass. Formal checks the exact sanity receipt, native exit0, plan/source/input hashes and strict reload before starting anew from B. Wrapper creation records RUNNING before spawning its child (run_task.py:18-28). The launcher must supply `OPENCV_FOR_THREADS_NUM=1` as specified by the plan.

The previous saved-TRAIN decision audit remains descriptive evidence only; it does not prove this intervention will improve final-weight tracking. No new accuracy, novelty or causal-benefit claim is supported by this review.

**Validation:** 49 local source/AST checks;48 inherited remote hashes;23 archived input hashes; B6250 CPU checkpoint/config/native-status inspection; complete schedule census;21 CPU scalar/native-identity fixtures. Torch1.12.1+cu113 stayed CPU-only with CUDA uninitialized. No trainer import, neural execution, GPU allocation, active dev/process/resource query, remote write, source/plan/HANDOFF edit or Git mutation occurred.

**Remaining admission:** the executor must preserve the reviewed hashes and run the planned native four-update GPU sanity, then the full formal job from B6250 after native sanity exit0. This verdict does not claim runtime admission or training completion. No source fix is required.

Detailed machine report: `reports/JOINT_QUALITY_TRAIN_SOURCE_REVIEW_20261026.json`. CPU evidence and reviewer trace: `.aris/traces/experiment-audit/2026-10-26_joint_quality_source_review/`.
