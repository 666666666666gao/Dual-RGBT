# Scale candidate training source review

**SOURCE PASS — 0 blocking issues.** The new sampled-B training and saved-configuration reload path is consistent with the frozen plan. No source correction is required for this declared scope. Runtime admission remains **PENDING**. Generated 2026-10-15T01:59:41.630911+00:00.

Fresh independent context; requested `gpt-6-astra` / `max`. Actual model, effort and routing: **UNATTESTED**. Review independence: **same-family**; acceptance: **provisional**.

Plan SHA256: `410023bc9c687c5a0a7829040908ff2126835b3d70eb9e2a669d6e6f48afdc0c`. All **68** registered source files match; the accepted original **64** are unchanged. **67** Python files parse under the Python 3.7 grammar. All **78** local source/metadata/manifest inputs remain unchanged. The complete source hash map is in the JSON companion.

| Check | Verdict | Finding |
|---|---|---|
| Frozen input identity and syntax | PASS | All 68 registered source hashes match; the accepted formal 64 are unchanged; 67 Python files parse as Python 3.7 syntax; 78 sealed local inputs remain unchanged. |
| Reuse and construction | PASS | The new policy inherits the original CADTrack/SOT executor and 955-input gain head; the two scalar radius attributes add no learned parameter. The new driver calls the scale factory directly and reuses original sampling, optimizer and causal-probe helpers without monkeypatching or changing old factories. |
| Initialization | PASS | Before construction the driver requires registered factory ba49b506 and fitted preview ad059943 identities, the protected CADTrack hash, source/input hashes and independent source report. It strictly loads the original fitted preview, never the M0 checkpoint, for formal initialization. Checkpoint bytes were not loaded in this source review. |
| Training-only sampling and budget | PASS | Actual EvidencePairs NumPy sampling AST reproduced all 50,000 planned choices: all 882 train sources in each of five epochs, 32 distinct M0 sources, strictly earlier templates and finite positive-size float32 targets. Train names are disjoint from dev97/test245. Batch8 gives 6250 updates and two executed labels per sample gives the registered 100000-label budget. |
| Sampled targets and family counts | PASS | Both modalities sample uniformly from actual kind-matched eligible requests. Local/global families use their own requested cell and rank. Source applies original execute_boxes image clipping before main/query IoU, then subtracts query IoU minus main IoU in float32; no nonnegative clamp or GT crop clipping. Per-family eligible count, request cell, ordered-union count, raw/executed boxes and signed label are distinct logged fields. No actual NN label was generated or verified here. |
| Optimization and freeze | PASS | Seed2027, eval mode, no autocast/TF32, zero eight-dimensional offline history, inherited SmoothL1 quality plus mean of two modality gain losses, AdamW lr/weight_decay1e-4 and gradient clip1. Only executor.quality/query_gain parameters require gradients. M0 requires finite nonzero head gradients, real optimizer step count, 257 unchanged frozen states and exactly eight changed head states; these remain runtime assertions, not completed results. |
| Saved configuration and strict reload | PASS | Both checkpoint and config.json use stage scale_candidate_b and retain .25/.50/max8. The loader reads saved radii before factory construction and strictly loads saved net. Capacity is returned in config and explicitly supplied by caller; it is not a hidden model attribute. M0 compares 14 ordered tensor outputs including request cells and family counts, saved config equality and K1 legacy geometry/gain equality. |
| Own-M0 formal gate | PASS | The exact native command, COMPLETE/exit0/GPU1, own stage, plan/source/input/config bindings, strict reload/probe fields, freeze/change counts and ten-frame selected-output replay are checked before seed setup and model construction. Twenty-seven synthetic evaluations of the existing AST gate accept current fixtures and reject actual mismatched command/stage/config/failure conditions. These are source fixtures, not real M0 receipts. |
| Causal probe and scope | PASS | The reused ten-frame tracker probe receives the new model and capacity, uses frame0 initialization then prior prediction/history/templates, replays chosen STOP or one heavy query, and requires all265 model states unchanged. This is an execution probe only. The B dataset remains GT-jittered offline pairs; new C, frozen dev selection, three benchmarks and complete-system latency remain separate pending work. |
| Commands and current priority | PASS | CLI arguments and native command expectations are consistent with scripts/run_task.py. Command templates below are unexecuted. Existing E0/E1 training and benchmark priority is preserved; no task or observer was launched. |

The independent NumPy run extracted only the real `EvidencePairs.__getitem__` sampling prefix. It reproduced **882/882/882/882/882** source coverage and **32** distinct M0 sequences. Each modality receives family counts **12504, 12504, 12496, 12496**. The planned M0 is **4 updates / 32 samples / 64 labels**; full B is **6250 updates / 50000 samples / 100000 labels**. These counts describe the deterministic intended exposure and budget, not completed training.

The target path was reviewed at `train_scale_candidate_policy.py:31–65,198–235`, with `training.py:18–66`, `regions.py:35–49` and `readout.py:49–58`. Targets stay in original-image float32 xywh; predicted boxes use the same margin10 clipping as online execution before IoU. The signed gain remains query IoU minus STOP IoU. GT does not enter candidate proposal/gain inputs. Geometry counts distinguish the actual per-family request set from the shared union, which may contain up to 15 locations under the accepted geometry review.

Configuration/reload evidence is `scale_candidate_checkpoint.py:9–17` and `train_scale_candidate_policy.py:96–119,123–182,240–254`. Capacity is restored as part of the returned config and used explicitly; both radii are restored into model attributes before strict loading. The actual runtime assertions cover 14 prediction fields, K1 legacy equivalence, optimizer steps and frozen states. The unchanged inherited `causal_probe` and `CandidateEvidenceTracker` exercise the new model without asserting that causal C training or evaluation integration already exists.

The **27 synthetic gate cases** execute the existing AST assertions in a private fixture directory only. They cover own current acceptance and mismatched native command/stage/config, failed exits, old 12-field reload evidence and failed causal replay. They introduce no production logic or requirements. No live run directory/status was read. Command argument arrays compatible with the native wrapper are recorded under `command_templates_not_executed` in the JSON; neither command was executed or queued.

Native independent audit: **exit 0**, 2026-10-15T01:56:46.511006+00:00 → 2026-10-15T01:57:05.283931+00:00; stdout **908 bytes**, stderr **0 bytes**; tool chunks `ec746b` / `68fc67`. NumPy **1.26.4** plus stdlib only. Worker SHA256 `14aa740a4f90123f7cb18ca47647bf782e99126230846a6f463f251cc455c5c1`; stdout SHA256 `07fa862675a4a5a42de1b572817eade5d96aebe8a1d72da798621b4d6c975c7d`. No reviewer helper failed. Private evidence is in `C:/Users/gb/.codex_tmp/dual_rgbt_scale_training_source_review_20261015_agent`.

The preparer separately disclosed an initial Python-AST-on-YAML helper failure and retained its first helper. This review independently parses only the 67 registered `.py` files and checks the YAML by its registered hash; that preparation-helper issue is not a production source defect.

| New reviewed artifact | SHA256 |
|---|---|---|
| `dual_rgbt_global/scale_candidate_checkpoint.py` | `482d6a633b5d77180b8c22a0fd334467fbb6ed6dd1c0ab3f14c2bee88ec591ba` |
| `scripts/train_scale_candidate_policy.py` | `8b3ecc793d6e873a841aca313033d1a2ec79db0f3dec180684161ede355b559e` |
| `docs/SCALE_CANDIDATE_TRAINING_PROTOCOL.md` | `2351adb2c8ca2b432c9330cecbcd37c8ed7eeedd6a54828e858871db2599fde1` |
| `reports/scale_candidate_training_preparation_20261015.json` | `aafb5417bd3bd4b1e002daf08800faa55685b2e2b20bcde4aa8ee56b74c06db1` |

**Limits and remaining evidence:**

- SOURCE verdict only: no Torch import, model construction, weight load, image loader, NN forward/backward/update, actual checkpoint reload, causal tracking, GPU/server/process/status read or environment installation occurred in this review.
- The 100000 query labels and M0 64 labels are future executed-label budgets. NumPy sampler exposure and gate fixtures do not demonstrate those neural executions.
- Actual own M0 remains pending. Formal B still requires its real native exit0, COMPLETE/SANITY/config records and the reviewed gate. SOURCE PASS is not runtime or formal-result admission.
- Separate scale-aware C training, evaluation integration, held-out selection, full LasHeR/RGBT210/RGBT234 metrics and complete-system timing are explicitly pending. They are not blockers for this sampled-B source preparation.
- The .25/.50 radii are an internal configuration comparison, not a selected dataset optimum or evidence of three effective mechanisms, improvement, target attainment or speed.
- Fresh context is independent but the model family is the same; requested gpt-6-astra/max, actual model/effort/routing UNATTESTED, acceptance provisional.

Only the review MD/JSON are public outputs. No reviewed source, plan, protocol or accepted result was edited. Existing formal E0/E1 work retains GPU priority.
