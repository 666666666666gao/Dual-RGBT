# P25 DropMoE FiLM Runtime Archive R1

Observed at 2026-11-04 03:25:34 UTC. This archive restores remote runtime
metadata for the four completed P25 DropMoE FiLM TRAIN collection shards.
It contains no checkpoint, weight, or sequence-artifact payload.

Local archive:
`reports/c2_film_replay_runtime_archive_r1_20261104/`

Transfer result: 24 files, 234,065 bytes. Every local file SHA256 was compared
with a hash computed from the corresponding P25 remote bytes; all 24 matched.
The archive contains each task plan, native status, `COMPLETE.json`,
`config.json`, `precision.json`, and `progress.json`. No standalone log file
was present in the inspected task/run directories.

| Shard | Native task | Status | Sequences | Frames | Records |
| --- | --- | --- | ---: | ---: | ---: |
| s0 | `collect_c2_policy_dropmoe_film_s0_full_r1_p25_20261103` | COMPLETE/0 | 221 | 98,613 | 19,574 |
| s1 | `collect_c2_policy_dropmoe_film_s1_full_r1_p25_20261103` | COMPLETE/0 | 221 | 103,761 | 20,578 |
| s2 | `collect_c2_policy_dropmoe_film_s2_full_r1_p25_20261103` | COMPLETE/0 | 220 | 128,483 | 25,541 |
| s3 | `collect_c2_policy_dropmoe_film_s3_gpu3_full_r1_p25_20261103` | COMPLETE/0 | 220 | 127,696 | 25,398 |
| Total |  |  | 882 | 458,553 | 91,091 |

Local validation found for each shard: native task/status and receipt agree;
receipt plan SHA equals the archived task-plan SHA; receipt is complete,
non-smoke, executor-preserving, and explicitly has
`benchmark_evidence=false`; archived config matches the task plan; all three
precision flags are false. No `.pth`, `.pt`, or `.tar` file is present.

## Archived File SHA256

| File | SHA256 |
| --- | --- |
| `task_plan/s0.json` | `5e4ce47af18e1c5a11841b4c499446034c35acecaf3b3e2035b1375633e2254c` |
| `native_status/s0.json` | `42e847d2da586846d4eff2e87a38be0909829d74cb9b8591a9dc0607be1b76b9` |
| `runtime/s0/COMPLETE.json` | `4c262a68df8b0bf5f704ebe3e47a204e24f99573c96ba4351a8bd203a757808b` |
| `runtime/s0/config.json` | `5e4ce47af18e1c5a11841b4c499446034c35acecaf3b3e2035b1375633e2254c` |
| `runtime/s0/precision.json` | `8fba9d3d90e81208ddaf86ac94bffd762ee8834b463586421b9c74e7c7a4fecc` |
| `runtime/s0/progress.json` | `247d138bdfa251c0806897ff13a97080879dd42efb9809fe29f312c06d28db56` |
| `task_plan/s1.json` | `fe9addecb1dace24435eb9929144c2279ecbfeb6852f6e2f32d8a3609338ffe9` |
| `native_status/s1.json` | `e7f5d0b14d863be69cc5f75c3eda9df11ff564e8a9a01d80a3a9204dcd2b0f3a` |
| `runtime/s1/COMPLETE.json` | `43465e590976bc90478867894fb32e52fa82956a42440aba6483309788411d3c` |
| `runtime/s1/config.json` | `fe9addecb1dace24435eb9929144c2279ecbfeb6852f6e2f32d8a3609338ffe9` |
| `runtime/s1/precision.json` | `8fba9d3d90e81208ddaf86ac94bffd762ee8834b463586421b9c74e7c7a4fecc` |
| `runtime/s1/progress.json` | `86d53f3f42dd1283c38bc531d978b73df40bb3d7a2343840f0064f983086169f` |
| `task_plan/s2.json` | `a8dbca765a47223b5b392efd105ce894aacb0411ea2a23615a29981e4bb494ad` |
| `native_status/s2.json` | `69601bdaf04c5ce0fcea2f9a5a74a14ad360f4b8697f48a3f64046715742cf55` |
| `runtime/s2/COMPLETE.json` | `9a9c9edf021fbf6b24d2c30c4cc706ddf9081f8494f1df1383eef00fa8a7aeac` |
| `runtime/s2/config.json` | `a8dbca765a47223b5b392efd105ce894aacb0411ea2a23615a29981e4bb494ad` |
| `runtime/s2/precision.json` | `8fba9d3d90e81208ddaf86ac94bffd762ee8834b463586421b9c74e7c7a4fecc` |
| `runtime/s2/progress.json` | `e74065917d4c93c4c9daf4d626d4500fd68d5a4eaee98371a75fe23360bc6dcb` |
| `task_plan/s3.json` | `fc9ce52a01e819c2802d055c9e5f93e9f44a3a77bf22bdd1fb084340cc1523fd` |
| `native_status/s3.json` | `8293e3e4bcc170ddaa79526b43ca01d46930e0646ea21cab352c9be7a1448450` |
| `runtime/s3/COMPLETE.json` | `3a7ca917f3cbc299436bfd75dedd3e991db33a08b897195bb794379ce2b55677` |
| `runtime/s3/config.json` | `fc9ce52a01e819c2802d055c9e5f93e9f44a3a77bf22bdd1fb084340cc1523fd` |
| `runtime/s3/precision.json` | `8fba9d3d90e81208ddaf86ac94bffd762ee8834b463586421b9c74e7c7a4fecc` |
| `runtime/s3/progress.json` | `eb4f9d5bb5bf5df749dc8591875d72f22adb11edc0aa512de4ea27ee52c6731e` |

## Remaining Admission Blockers

This transfer resolves the prior review's missing-local-runtime-bytes blocker
for the four collection receipts/status/config/precision files. It does not
resolve the two source-plan raw-byte mismatches: the previously reviewed s0
and s2 raw SHA256 values were
`68a188a1e46374668d4cdf1f2f915fe4fe2bdef27aa4814d44677931669c0c46` and
`987e46551ec3c843a46f8c5d6ca8a6f6f239d4d2e0ad16b5a15aff6b9a856e1e`.
Neither exact source-plan file was found under its expected isolated-root
path on P25 (`/data2/gaob/Dual-RGBT` or `/home/gaob/Dual-RGBT`). The archived
task plans are runtime copies and are not represented as those original
reviewed source-plan bytes.

The existing replay plan also remains incompatible with the current launcher:
it lacks top-level `launcher` and `review` objects. The disabled plan and
failed R2 review remain unchanged. No replay, policy FIT, benchmark, or model
selection task was launched. P25 GPU availability does not grant plan
admission; obtain a new exact source review and canonical runtime admission
before using any free GPU.
