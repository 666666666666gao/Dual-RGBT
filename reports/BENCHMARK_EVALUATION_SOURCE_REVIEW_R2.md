# Benchmark evaluation SOURCE review — R2

**SOURCE_FAIL — 2 blocking issues.** Reviewed exact plan `85757fd0ddb23855ed9861eb81cf90a4db863e7cea32f255c124804d43d7b498` and all 44 source bodies. Requested Astra/max routing is **UNATTESTED**; this is a same-family, provisional source review. No NN admission is granted.

## Blocking finding: incomplete development evidence can freeze a test configuration

The unchanged R2 freezer and `frozen_selection` accepted three artificial reports declaring the complete LasHeR dev split (97 sequences / 54,646 frames), with the three planned costs, but each report hashed only an unrelated marker file. There was no `runs` directory and no prediction, saved GT, timing, config, COMPLETE or native receipt. The full freezer main wrote a selection, and the shared validator admitted it.

This is an executed CPU protocol counterexample to `scripts/benchmark_inputs.py:22` and `tools/freeze_benchmark_selection.py:12`, not a model result. The source and plan seals passed. The required correction is to require complete registered development artifacts and actual CPU score recomputation before producing the fixed selection SHA. Current R3 edits are outside this R2 review.

## Closed R1 checks and retained behavior

The original R2 checkpoint gate was executed with JSON-only load stubs. Both legitimate E0 and E1 producer schemas were accepted; 26 corruptions were rejected, including unrelated native task/GPU/command, wrong checkpoint run/disk config, E0 order/integrity failures, and E1 source/input/factory/K/sanity binding or duplicate 882-name coverage. This closes the demonstrated R1 B1 metadata failures. It does not verify remote tensors or current training completion.

Eleven runner-prefix cases accepted the valid 11-frame smoke and rejected failed/stale SOURCE or metric reviews, wrong parent/task/canonical command, GPU7, and wrong smoke settings. Twenty static checks cover Python3.7 grammar, original E0 tracker and crop alias order, existing E1 tracker, strict model loading, frozen/no-grad inference, GT only at initialization and after prediction, and reload of causal state. All 44 source files and the helper parse as Python3.7 grammar; execution used local Python3.12/NumPy, not Python3.7.

The helper hash is `52367bfe6ffc83a1ceefb51504d63d15a7e4eaa719ef62d5e6faa47d201823d0`. Its source launches E0 then E1 through `run_task.py` on GPU1, 11 frames each, waits 180 seconds before the first observation and 240 thereafter, collects only its own tasks, and requires zero optimizer/backward counts. Rendered remote snippets parse but were never executed. There were no old E0 process/log/status queries.

Old 38 source bodies and plan input/toolkit seals are unchanged. The five actual manifests were rehashed; the R1 complete GT/provenance audit is retained. Counts are LasHeR dev97/54,646, test245/220,703; RGBT210 test210/104,706; RGBT234 test234/116,650. RGBT210 retains common raw GT and RGBT234 separate visible/infrared GT. Image receipts establish first/last PIL headers only, not full image decoding. Full checkpoint bytes remain remote and are qualified by prior native hash receipts.

The sibling metric reviewer completed six legal full-data scoring paths and rejected 20 corruptions, closing all four R1 scorer counterexamples. Its new actual counterexample is the second blocker: swapping the configured shard ids and matching canonical commands while retaining the old sequence files/lists is accepted. Aggregate97/54,646 coverage stays complete, so numerical scores are unchanged, but the reported command does not produce that shard under the runner's manifest slicing rule. Require exact `config.sequences == manifest[shard::shards]` names. This reviewer read that mutation and verified all primary evidence hashes, including raw subprocess streams and output; the execution is attributed to the sibling.

## Evidence

Private ledger: `D:\Program Files\UserCache\gb\codex_tmp\dual_rgbt_benchmark_source_review_r2_20261015_37ac92e1`. `freeze_gap_result.json` and `freeze_gap.receipt.json` record the actual failing admission. `admission_result.json` and `admission.receipt.json` record 28 checkpoint cases, 11 runner cases and 20 static checks. Both processes exited 0 with empty stderr. `original_R2_source44_plan_helper.tar.gz` seals the reviewed inputs. All hashes are in the JSON companion. The earlier R1 report pair is unchanged.

NN calls, torch imports, GPU queries, SSH calls and current E0 observations: **0**. This source-only result establishes no performance, numerical reload equivalence or full-C completion.
