# Sampling scope helper: independent SOURCE-ONLY review

Date: 2026-10-13 (Asia/Shanghai)

```yaml
review_mode: SOURCE_ONLY
verdict: WARN
source_verdict: PASS
blocking_findings: 0
required_code_changes: 0
review_independence: same-family
acceptance_status: provisional
requested_model: gpt-6-astra
requested_reasoning_effort: max
actual_model_route: UNATTESTED
native_manifest_execution: NOT_RUN
neural_execution: false
```

The requested model/effort are recorded as a request, not an independently attested backend route. This separate reviewer inspected the supplied source and completed local artifacts directly. This is same-family provisional review, not cross-family acceptance.

**Verdict: WARN, zero blocking findings.** The helper correctly reconstructs the sampled search-target identities and Stage A batch routes under the current-manifest/source assumptions. No source correction is required for the specified completed runs. The warnings below bound the evidence; they are not observed implementation failures.

## Reviewed scope and actual checks

Reviewed `scripts/audit_sampling_scope.py`, `dual_rgbt/data.py`, `scripts/train_executor.py`, `scripts/train_policy.py`, project `AGENTS.md`, and `C:/Users/gb/.codex/skills/shared-references/local-codex-policy.md`.

Read only these completed experiment artifacts: the config, COMPLETE and step CSV in `reports/stage_a_spatial`, `reports/stage_a_full`, and `reports/full_pipeline/stage_b_refreshed_20261013`; the config and COMPLETE in `reports/full_pipeline/stage_c_complete_20261013`. Directory entry names were listed to locate the approved files; no training log or dev-file content was read.

Actual reviewer checks used local PowerShell JSON/CSV reading and SHA-256 calculation. The helper entry point was not run or imported. No manifest, image, checkpoint, live status, live log, or live dev file was read; no SSH, GPU query, neural forward, optimizer update, evaluation, or remote process operation was performed. The executor's reported AST check is not presented as this reviewer's own execution.

| Completed run | Saved epochs | Samples/epoch | Batch | Actual CSV rows and saved steps | Audit result |
| --- | --- | ---: | ---: | --- | --- |
| A short | 1, 2 | 2,000 | 4 | 1,000 rows; 0–999 | PASS |
| A full continuation | 3–7 | 10,000 | 8 | 6,250 rows; 1,000–7,249 | PASS |
| B refreshed | 1–5 | 10,000 | 8 | 6,250 rows; 0–6,249 | PASS |

Every CSV was independently checked for contiguous step order and the expected epoch row order; mismatches were zero. Each A row was checked against `modality = step % 2` and `action = (step // 2) % 5`; mismatches were zero. The short A CSV has 100 batches per route and the full A CSV has 625 batches per route. Consequently the completed configs/CSV support 4,000 + 50,000 = 54,000 A sampled items and 50,000 B sampled items; each A route receives 400 + 5,000 = 5,400 sampled items. These are draw counts, not unique-frame counts.

## Source findings

1. **PASS — RNG draw order and target selection.** `dual_rgbt/data.py:52` creates `RandomState(self.seed + index)`, selects a sequence with the first `randint`, and selects `valid_frames[randint(2, len(valid_frames))]` with the second. The helper at lines 43–48 does precisely these operations, in the same order. Template sampling and crop jitter occur after the search target is chosen, so omitting them does not change these two draws. The count refers to the actual selected frame value, not its ordinal within `valid_frames`.

2. **PASS — epoch seeds use each run's own sample count.** Both training sources create the dataset with `seed + (epoch - 1) * samples`. The helper uses the saved absolute epoch and the corresponding config. In particular, resumed A epoch 3 starts at `2027 + 2 * 10000 = 22027`; it does not continue using the earlier 2,000-sample epoch offset. The reviewed A/B configs all use seed 2027. Counts and target sets are combined without summing per-epoch unique counts.

3. **PASS — A sequential batches and saved routes.** `train_executor.py:71` explicitly uses `shuffle=False` and `drop_last=True`; all reviewed sample counts are divisible by batch size. The helper asserts this divisibility, CSV lengths, epoch order, contiguous saved steps and the route formula. Mapping item `index` to CSV row `epoch_index * batches + index // batch` therefore agrees with the source. The continuation's saved step 1,000 is preserved. Route-specific sequence/target sets are distinct from the aggregate A sets.

4. **PASS — B has no per-batch route partition.** `train_policy.py:63` uses the same seed formula, and its DataLoader supplies no shuffle or custom sampler. The reviewed 10,000 items divide evenly into batches of 8. For each batch, label construction evaluates both primary modalities and actions 1–4 for each modality, with primary losses supplying the quality targets and gain baselines. Passing `routes=None` and reporting one B coverage set correctly avoids applying A's alternating route schedule to B.

5. **PASS — coverage quantities are kept separate.** A/B `unique_search_targets` are sets of `(sequence_name, frame)`; sequence participation is a separate count. The eligible-search denominator is `sum(len(valid_frames) - 2)`, corresponding to the search sampler's allowed positions. `train_noninitialization_frames` instead uses `sum(len(boxes) - 1)`. Neither denominator is a template count. The helper explicitly excludes template identity, crop coordinates and pixel coverage from its unique-search-target claim.

6. **PASS — bounded C scope.** The saved C config lists 20 distinct sequence names and `rounds=2`; COMPLETE also records two rounds, 20 sequences per round, 17,708 tracked frames and 3,504 steps. The helper requires these selected names to belong to the current 882-name train manifest and checks `2 * c_total == tracked_frames`. Its C count is limited to those 20 sequences. `2 * total` is explicitly labeled a hypothetical whole-train budget; it is not a launched or completed 882-sequence result.

7. **PASS — implementation stays within CPU reconstruction.** The helper imports CSV/hash/JSON/path utilities and NumPy, not the training modules or Torch. Source inspection shows no image loading, checkpoint loading, neural operation, subprocess, or network call. It reads the designated completed run artifacts and current manifest and writes `reports/sampling_scope_audit.json`. Its intended `runs/...` inputs differ from the local `reports/...` copies used for this review; equality with the eventual native inputs remains to be established by their reported hashes.

## Warnings and acceptance boundary

- **WARN W1 — historical identity is not attested.** Current manifest and current source hashes bind the inputs used by the eventual audit. They do not establish that those bytes or the NumPy environment equal the launch-time inputs. The helper accurately labels the result as a current-manifest reconstruction, not a historical `__getitem__`, image-load or pixel trace.
- **WARN W2 — native reconstruction is pending.** No actual manifest-dependent unique counts, coverage fractions, missing-sequence lists, manifest SHA or output JSON are accepted by this SOURCE-ONLY review. The `882` manifest assertion and the C `2 * c_total` assertion were inspected, not executed. Their real outcomes must be recorded after the native CPU run. This source verdict does not waive the existing diagnostic schedule or authorize a neural experiment.
- **WARN W3 — C frame coverage is a configured-manifest calculation.** C config/COMPLETE establish the saved 20-sequence/two-round scope and aggregate counts. This review did not inspect trajectory rows or independently trace every visited frame. The helper's `distinct_noninitialization_frames` denotes the selected sequences' current-manifest noninitialization-frame capacity; it is not a separately observed per-frame visit trace. From COMPLETE, 17,708 / 2 = 8,854 is the expected capacity if the helper's consistency assertion passes, not a manifest result computed in this review.
- **WARN W4 — no scientific acceptance from coverage.** Sequence participation and sampled-target counts do not demonstrate full-frame traversal, unchanged pixels, checkpoint reload correctness, accuracy, latency or module effectiveness. The helper's limitations preserve this distinction.

No FAIL finding or source blocker was found. Subsequent native results should retain the above limits and record actual exit status, output and input hashes; no native execution is claimed here.

## Reviewed byte identities

All SHA-256 values below were calculated by this reviewer from the allowed local files.

| Path | SHA-256 |
| --- | --- |
| `scripts/audit_sampling_scope.py` | `627f47572c8ffb17ee1ce9da5ce08ef0f99f8d9a38a7a6caa30c26c6b54fd4f7` |
| `dual_rgbt/data.py` | `858e41c8e5218c7f5323a6e18cec00997a48d0323101384b71a73d61ea0280cb` |
| `scripts/train_executor.py` | `c5a41ff3924daf9568b8c8e848dd1e1686bb0e166ecb327e6813879f5c605d80` |
| `scripts/train_policy.py` | `2fa070434680704c984d9a1b5b992973103971c47d692a6fcbea734571922913` |
| `reports/stage_a_spatial/config.json` | `ff754568d2521a334800f74c7e9328011a94f6349084321a2366b225844be8d5` |
| `reports/stage_a_spatial/COMPLETE.json` | `a15100ac875061b948ab1b9a123004824bd76a301cd15197f8dac2a27a82fbee` |
| `reports/stage_a_spatial/steps.csv` | `cd22613eb6f00cd89b6978666cbf015b1372bf995ef2120fa52c951afb88c6a4` |
| `reports/stage_a_full/config.json` | `e61a0180443f544d89139f99f8d30e9b85604756df19a1afb6a24000b63f134d` |
| `reports/stage_a_full/COMPLETE.json` | `f958f106ff24605a5127536bf90b6774ba83005b64c113209055e8af9cee07ae` |
| `reports/stage_a_full/steps.csv` | `63b6ae31d2ac025a726cb325a832eaeeb5a76a8e34fcace409d984a9be3088f3` |
| `reports/full_pipeline/stage_b_refreshed_20261013/config.json` | `4317f9cbfc10c40cc60efe9b947fbeedf6abfa2a55dea21837fe7fd33e534dd3` |
| `reports/full_pipeline/stage_b_refreshed_20261013/COMPLETE.json` | `f7a1db8148769bd7e63944e719802675eddde7a388c1512e3668153cf76259da` |
| `reports/full_pipeline/stage_b_refreshed_20261013/policy_steps.csv` | `62c296440b35c6ff19fad5e3f30b5b1853dbf576afa7158c8aaa944ca7582792` |
| `reports/full_pipeline/stage_c_complete_20261013/config.json` | `9b163a02d6bdf122e5d4e12a33332f9b8ddfec9a4f0e387b2ce164b16496906d` |
| `reports/full_pipeline/stage_c_complete_20261013/COMPLETE.json` | `cac98d116b1a28b1a48015781234598e39915fd3809da2bb657aa64494de3e86` |
