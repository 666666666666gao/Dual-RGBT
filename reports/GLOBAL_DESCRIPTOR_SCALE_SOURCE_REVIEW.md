SOURCE_PASS; 0 blocking issues. Independent source review of the frozen descriptor64/32 diagnostic, 2026-10-14.

Plan SHA256: 146c29016fbb9a513c35c8ed738433106e6788c15294ee7023d04b1c1d8de52b. New script SHA256: abdf76b9b18f80ac352cc3e7e1cfdcf612f4fc8a9653071270a025ff8c04f4c8. All32 source hashes match; old31 remain unchanged.

179 successful source/data/synthetic/reference checks:85 source/admission bindings,19 sealed-data identity checks,43 NumPy source-extraction checks and32 pinned API/reference checks. Initial56 and supplementary11 inputs are unchanged at closing seals.

Shared three stride2/pad1 Conv2d layers produce nominal8x8 for64 and4x4 for32. New64 uses the exact prior bilinear/align_corners=False resize and[2:6,2:6] mean. New32 uses[1:3,1:3] mean. NumPy source extraction validates crop/indexing and resize signature with explicit feature stubs; no CNN was run.

Same frame0 factor2/128 template crop and normalization, same current global256 crop, same auxiliary role1-old.modality, same valid32x32 grid, same local/wide ROI sides from the old query geometry, and same post-fit ad059943 checkpoint. Global feature is outside the resolution loop and shared. No controller/trajectory/backbone execution is introduced by the diagnostic.

Current baseline path binds the post-fit ranking_after.json SHA2de0a53c..., not the old pre-fit query centre. All1752 sequence/frame/state/auxiliary-role pairs and native-main-outside flags match the sealed old content CSV. Source asserts every64 peak within1e-3 of the post-fit baseline. That assertion has not been executed by this review.

Current-frame GT is used only for nearest-cell rank, score margin and ROI coverage. target/centre/nearest are prepared before the score loop, but do not enter score or peak dependencies. Initial-frame GT is the unchanged shared template initialization. Numerical rank and native-main-outside membership remain native-derived.

Source loads stage=localize/step6250/non-sanity checkpoint, binds warmup-plan SHA, calls eval, disables parameter gradients, clones the full state dict, uses no_grad for diagnose and reload probes, checks all state tensors equal and all gradsNone, and has no optimizer/backward call. Sanity compares4 outputs across2 frozen states; strict reload checks scores64/point64/scores32/point32 on the final sanity probe. Reported265 count agrees with prior259 frozen plus6 preview tensors; actual new265 tensor validation remains native sanity work.

Two sanity states yield4 comparisons; formal1752 states yield3504. The20 diagnostic sequences match the old content plan and belong to train882. Per-frame rows/peaks are written only to runs/.../comparisons.json; printed COMPLETE contains aggregate groups and a private-file SHA. This is no dev/test, PR/NPR/SR, new trajectory, query-benefit, stability, inference-latency or final-method evidence.

Plan/source/input/protected-weight hashes checked before and after; prior native COMPLETE/exit0 and full6250/882/non-sanity marker required; current localize6250 checkpoint is tied to completed full-B-derived warmup plan. Both documented home-conda/GPU1 CLI commands match exact run names and parser. Protocol requires actual sanity COMPLETE/exit0 before formal; that sequencing is a workflow gate, not a sanity-marker read in this script.

Old training/controller sources and initial local all-train status/third-node records are unchanged at closing seal. The diagnostic never invokes the fullC controller. Existing GPU0 allC and owner16936 were not queried, modified or restarted in this review.

Locally cached official PyTorchv1.12.1 source hashes match prior fixed metadata and include all used APIs, including bilinear interpolate and cosine broadcasting. Prior saved native environment reports1.12.1+cu113. Actual SiamFC39572051 source SHA78a29892.../11077bytes matches provenance and uses x_sz=z_sz*instance_sz/exemplar_sz. Its scale design supports a hypothesis only; no SiamFC execution, benchmark reproduction, causal proof or novelty claim.

Fresh same-family/provisional source review; requested gpt-6-astra/max, actual runtime model and reasoning UNATTESTED. Not blind or cross-family.

No torch/project import, actual model forward, CUDA/GPU query, NN job, optimizer/backward, SSH, environment change or installation in this review.

Real native sanity is still required before formal; actual tensor shapes,265 state count, frozen equality, baseline64 peaks, strict reload and real input-byte admission have not been newly measured here.

Checkpoint/native-completion bindings were checked against sealed saved records, not by opening remote weight files. Native execution must enforce the source SHA gates.

Both descriptors retain the same initial template; appearance ageing, distractor identity and descriptor-input resolution versus receptive-field effects are not individually isolated.

Two review-harness failures were preserved privately: defaultGBK decoding and a wrong first20 assumption. Both were corrected in the reviewer code; final checks pass. No reviewed source, plan, protocol, old source or private raw-data input was edited.

Machine gate: reports/GLOBAL_DESCRIPTOR_SCALE_SOURCE_REVIEW.json. Private checkers, first/final seals, all actual stdout/stderr/exit receipts and failure originals: C:/Users/gb/.codex_tmp/dual_rgbt_global_descriptor_scale_review_20261014.

Primary reference links: [pinned SiamFC source](https://github.com/huanglianghua/siamfc-pytorch/blob/39572051d7c3d9850ca6100efa04091e18c2f706/siamfc/siamfc.py) and [PyTorch1.12.1 functional source](https://github.com/pytorch/pytorch/blob/v1.12.1/torch/nn/functional.py). Local cached bytes were inspected; no benchmark or remote environment was run.
