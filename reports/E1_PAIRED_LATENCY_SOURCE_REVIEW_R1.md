# E1 paired latency SOURCE review — R1

**SOURCE_FAIL — 2 blocking issues.** Original R1 bytes remained fixed. Requested `gpt-6-astra` / `max`; actual routing/effort **UNATTESTED**; fresh-context **same-family**, acceptance **provisional**.

1. **F1: remote evaluation-plan SHA is unbound.** `compare_candidate_latency.py:54–56` and `complete_candidate_latency.py:118–122` check only source/input maps; the evaluation file is registered solely in `local_input_sha256` (`e1_paired_latency_plan.json:71`). Actual extracted pre-model admission and dependency code accept changed evaluation bytes. Indirect selection checks use the same mutable remote evaluation SHA. Add the already registered evaluation digest to remote admission, native pre-model and completion checks. This reproduces source admission, not invalid neural execution.

2. **F2: receipt identity is self-reported.** `complete_candidate_latency.py:17–27` only compares native and config command strings to each other. Matching `[python, wrong.py]` strings pass; actual mocked `main()` then launches full (`:139–143`). Wrong plan/manifest/protected-baseline digests, checkpoint path and frozen cost/K also pass. Bind validation to the controller-owned exact argv/run/plan and independently registered evaluation/checkpoint/manifest/frozen selection values; preserve the torch-free controller.

**Passed checks:** all64 source and19 local-input seals match; original46 are unchanged. Python3.7 AST passes on67 files. The actual local controller import introduces no torch. Actual mocked controller/validator cases reject failed nested native status, missing dependencies/owners, wrong owners, incomplete E1, failed/missing native tasks, failed smoke/reload, a full-prefix table, bad row/frame/order, invalid timing, and changed mean/P95/FPS. The valid synthetic full table covers97 sequences,54646 frames,53773 warm frames and54549 tracked frames.

Source inspection and the extracted CPU frame loop support separate model/tracker state, strict final-C loading and frozen dev selection, first-frame-only online GT, independent decoding, alternating order, complete synchronized service/tracking scopes, no baseline accuracy scoring, and no training calls. VRAM is correctly qualified as shared by two resident models; wall time covers the measurement loop rather than loading/post-loop verification. Real CUDA behavior and checkpoint tensors remain untested.

Executed **63 artificial CPU source cases**,51 expected outcomes met. The12 other probes reproduce F1/F2 plus nonblocking boundaries: mutually consistent impossible token/request values are not semantically rejected, and an unsupported top-level FAILED/live-owner schema keeps waiting. The current producer never emits that top-level schema; genuine nested failures reject. These are not12 separate blockers or NN/metric evidence.

| Original R1 artifact | SHA256 |
|---|---|
| Timing plan | `ea73e3565b71a11c17c9db06980d0e1664fbf8202dae0f73e2a6e4b744f342a6` |
| Timing runner | `deec15d2ad5f33a58fb371318020ae26847d24268db7a2ae2ee549d6565b818d` |
| Local controller | `95d0ff7548d44401deacecbed75e604b3b9551e7edeca988438870b8b9a639f2` |
| Protocol | `5bb276a7195bf9f0a972f74249f0c79b37bbe33016883efc240ef06a4433847f` |

All95 audited inputs remain unchanged before/after. Six inputs are remote-only. The static deployment helper SHA is `fe6fb344ffcad5b4347f1bef215cc53b5f5adc2897a0eab2929316be2a3bad7f`; it was only read/parsed, and its line18 SOURCE_PASS gate blocks R1. No SSH/GPU/NN/live status/log/CIM operation or active-handle wait occurred.

The JSON retains exact maps, per-check evidence and all cases. Private original R1 bytes and fixtures: `C:\Users\gb\projects\Dual-RGBT\.codex_tmp\review_e1_paired_latency_source_20261015_r1`. One reviewer-only argparse setup failure and its corrected exit0 are preserved; no production source changed. Preserve R1 before the minimal correction and fresh R2.
