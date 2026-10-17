# E1 三数据集完整 test 最终结果（审计带警告）

E1 full C882、冻结 `cost=0.15 / max_candidates=8` 的三个 test 已全部完成。六个 shard 均 native COMPLETE/exit0，覆盖和指标已 CPU 独立重算，审计结论为 WARN / RESULT_INTEGRITY_PASS_WITH_WARNINGS，无阻塞问题。

| 数据集 | 序列 / 帧 | 指标 | mean ms | P95 ms | FPS |
|---|---:|---|---:|---:|---:|
| LasHeR | 245 / 220703 | PR 67.612357 / NPR 62.545797 / SR 50.842354 | 138.345 | 211.688 | 7.228 |
| RGBT210 | 210 / 104706 | PR 84.429457 / SR 54.041235 | 100.592 | 141.951 | 9.941 |
| RGBT234 | 234 / 116650 | MPR 87.774864 / MSR 57.523560 | 91.507 | 133.602 | 10.928 |

LasHeR 仍未达到 PR 78.74 / NPR 75.16 / SR 63.84；相对用户 baseline 为 −5.127643 / −6.614203 / −6.997645 个百分点，相对已核实 E0 C882 为 −7.137346 / −8.521875 / −8.246652 个百分点。

审计文件：`reports/e1_three_benchmark_test_20261017/EXPERIMENT_AUDIT.md/.json`。审计确认三数据集均为 real_gt，保存曲线与汇总指标精确一致；警告包括单 seed、并行未配对计时、CADTrack legacy toolkit/NumPy 兼容语义，以及本地未封存正式 checkpoint tensor/native training receipt。上述结果不支撑速度提升、稳健性或模块因果贡献。

LasHeR trace 已暴露路由失败：E1 TIR 主模态比例约52.16%、辅助查询约6.72%，而 E0 分别约10.86%和83.93%。这支持优先修复质量路由与 STOP/gain 校准的研究方向；它是诊断证据，不是因果消融。
