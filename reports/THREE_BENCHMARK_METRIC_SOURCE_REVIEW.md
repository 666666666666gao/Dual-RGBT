# Three-benchmark metric/source review — R3

**SOURCE_PASS — 0 blocking issues, limited to the exact reviewed CPU source/metric protocol.** All demonstrated R1/R2 defects are closed. This is not a benchmark-performance or neural-execution result.

- Review attribution: same-family, provisional; requested `gpt-6-astra` / `max`; actual route `UNATTESTED`.
- Plan SHA-256: `fa9bd08a922978aa79a6d597a115f7654aaf84996e64baba8850eb59ce07ed25`.
- Inputs/helper SHA-256: `5af8f4f03f5ee0c1c02bc415eb2ea004a696e40c95a10993896abfa5bb246a2e`.
- Scorer SHA-256: `64ae2e3b142855bdea180f2b37ef13d39fd0090d035990726ee1fc1851e52d68`.
- Freezer SHA-256: `80786bd0d5239348dd178c15b7036effc86eacb2c6b6d884223af308a27eb6eb`.
- All 44 source files, plan, seven toolkit code files and five registered manifests are verified. Source/plan hashes were checked again before this report. R1/R2 reports remain byte-exact.

## Incremental execution evidence

| Executed private CPU check | Result |
|---|---|
| Three full dev paths, costs 0.0 / 0.05 / 0.15 | PASS, each 97 sequences / 54,646 frames |
| Original swapped-shard counterexample | REJECTED |
| Missing prediction table | REJECTED |
| Modified scalar SR submitted to real freezer CLI | REJECTED; no frozen file written |
| Modified per-sequence score submitted to real freezer CLI | REJECTED; no frozen file written |
| Valid freezer CLI | PASS; all three complete recomputed JSON reports equal their inputs |
| Omitted prediction hash in receipt artifact table | REJECTED by shared validation |
| LasHeR full test | PASS, 245 sequences / 220,703 frames |
| RGBT210 full test | PASS, 210 sequences / 104,706 frames |
| RGBT234 full test | PASS, 234 sequences / 116,650 frames |

The 11 CLI cases produced 7 expected accepts and 4 expected rejects. One additional shared-validator omission test rejected. No unexpected result occurred. All three full-test metric outputs equal R2 when using the exact same artificial prediction arrays.

The freezer executes the source-bound CPU scorer for all three full-dev reports, compares the complete parsed JSON—including sequence rows, aggregate values and raw artifact table—and validates the full run inputs before writing the frozen selection. The shared gate requires exact native/config/COMPLETE identity, registered sequence/shard coverage, prediction and GT file sets, finite positive timing, matching totals, complete artifact hashes and common model capacity.

The selected configuration is shared across all three benchmarks and its SHA is passed explicitly in the canonical test command. Future dataset-specific configuration selection needs its own registered protocol.

## Reused checks with verified unchanged code

AST comparison proves every R2 scorer statement remains in the same order. R3 adds only the exact shard-slice assertion and `run_names` output. `evaluation_command`, `groundtruth`, `load_manifest` and `sha` are unchanged. The old LasHeR scorer and all seven toolkit code files are byte-identical.

Therefore the four R1 rejection regressions, R2 command/capacity checks and seven independent metric goldens are reused instead of repeated. R1/R2 established all raw toolkit GT comparisons; the actual R3 full-score paths again match all saved test GT to the manifest and current toolkit and check RGBT210/RGBT234 annotation-file hashes.

Metric semantics remain: per-frame min-CLE/max-IoU for RGBT234; original PR/SR for RGBT210; original LasHeR PR/NPR/SR and special-GT treatment; float32 prediction rounding; unrounded modality GT; fixed frame denominators and sequence macro averaging. No denominator comes from prediction statistics.

## Scope and limits

**Every executed prediction, timing record and native receipt in these tests is an explicitly private artificial CPU protocol fixture.** The fixture uses real registered GT/manifests and manually specified predictions; it is not a new tracking result or baseline. No torch/model import, NN inference/training, GPU query, SSH or original E0 progress query occurred.

This SOURCE_PASS covers the reviewed source protocol only. Separate NN SOURCE checks, actual inference smoke, authentic full-C/checkpoint evidence and actual frozen full benchmark evaluation remain necessary. It does not establish accuracy, target attainment, model capacity or latency/FPS. Interior image pixels were not decoded.

## Evidence

- Private exact source/fixtures/raw CLI streams: `C:\Users\gb\.codex_tmp\dual_rgbt_metric_source_review_r3_20261015_5b82ce4d`.
- `test_r3.py`, `sealed_inputs.json`, `protocol_results.json`, `incremental_reuse_verification.json`.
- `valid_freeze.stdout.bin` contains the real three-score recomputation and frozen-selection receipt; all accepted/rejected CLI stderr and stdout files are retained.
- The machine report includes the exact 44-source map, plan/toolkit/input hashes, case exit codes, complete recomputation receipts and primary artifact hashes.
- Original R1/R2 reports and their private evidence remain unchanged.
