# 完整裁剪 API 的均值计算候选

2026-10-14，SOURCE 正确性 PASS/整体 WARN/0 blocker 后，在原 conda 运行纯 CPU 三臂对照。
原生 wrapper693753 从00:04:09.378运行到00:04:54.357 BJT，真实 exit0，用时44.979秒。
wrapper 的 gpu=1 仅是现有任务记录字段，实际子命令通过 `/usr/bin/env CUDA_VISIBLE_DEVICES=`
隐藏全部CUDA设备，无模型构造、NN前向或优化器更新。当前参考测试与封存全882入口均未改变。
记录见 `reports/sum_crop_cpu_native.json`；源码及五项依赖SHA见 plan/launch/native报告。

## 真实输出一致性与计时

10条真实dev序列，各取首个/中间有效帧，20模板+60搜索共80 cases。
所有border mean、图像Tensor、target、geometry逐位一致，60搜索的完整五部分RNG状态一致。
输入为uint8、六通道，两路通道整数总和低于2^53；实际断言OpenCV求和除法与NumPy mean相等。
每臂480次完整API调用，共1440次；连续化复制、均值、warp、归一化、Tensor构造、搜索RNG初始化均计入。
解码、六通道输入准备和精确性断言排除；PyTorch32/OpenCV64线程，三种循环轮转重复两次。

| 同次运行的实现 | API均值 ms | P95 ms |
|---|---:|---:|
| 原 crop_pair | 38.571554 | 64.660076 |
| 已准入 six_channel_crop | 21.149685 | 47.060887 |
| 新 sum_crop | 12.338135 | 21.452628 |

新候选相对同次原实现平均减少68.0123%，相对six_channel减少41.6628%。
80/80 case的均值比原实现低，只有35/80比six_channel低，并非每种情况都改善。
原/现有候选本次均值与历史44.066/24.298ms不同，不能把不同运行间差异归因于新增改变。

严格匹配全部80个case的sequence/frame/size/trial/seed/image_shape，复用历史guard判定作分组；
下面三臂时间全部来自本次运行，没有混合历史时间。

| 历史guard分组 | cases | 原实现 ms | six_channel ms | sum ms |
|---|---:|---:|---:|---:|
| 不计算均值 | 47 | 40.058661 | 10.756279 | 12.603384 |
| 仍计算均值 | 33 | 36.453553 | 35.952414 | 11.960355 |
| 模板 | 20 | 38.548182 | 12.841740 | 10.237637 |
| 搜索 | 60 | 38.579345 | 23.918999 | 13.038300 |

候选总是取均值，取消逆变换guard：在47个原本无需均值case中平均增加1.847ms，
在33个保留均值case中减少23.992ms。这说明当前混合分布平均有收益，也显示代价。
约35.95ms是整次crop时间，不是NumPy mean单算子时间；差值同时包含guard删除及均值实现改变。
分组与原始计时见 `reports/sum_crop_cpu_analysis.json`、`reports/sum_crop_cpu_profile.json`。

## 后续执行与解释边界

当前只完成CPU组件验证；没有实际NN准入、完整轨迹等价或最终权重整机延迟结论。
当前两路参考test仍共享机器CPU资源，且轮转没有遍历六种排列、未完全平衡两两先后。
不把本次结果作为独占机器性能、正式显著性、PR/NPR/SR提升或新核心模块证据。
新sum候选未接入排队全882的17项封存配置，不修改已有训练选择。
以后先做同冻结模型的真实状态/更新准入，再决定完整评价和最终同设备延迟测量是否采用。
最终延迟脚本已扩展完整97条dev，协议见FULL_DEV_LATENCY_PROTOCOL.md；该协议本身尚未执行NN。

## Instrumentation changelog

| 文件 | 改动 | 用途 |
|---|---|---|
| scripts/profile_sum_crops.py | 新建 | 独立CPU候选、输出精确性检查、三臂API计时 |
| reports/sum_crop_profile_plan.json | 新建 | 实验范围、源码SHA、预估节点 |
| reports/sum_crop_profile_launch.json | 新建 | 真实wrapper PID及CPU子命令 |
| reports/sum_crop_cpu_native.json | 新建 | 真实退出码、时间与17项封存一致性 |
| reports/sum_crop_cpu_profile.json | 新建 | 全部80 case/1440次计时 |
| reports/sum_crop_cpu_analysis.json | 新建 | 身份匹配与条件分组统计 |

独立实验文件保留以复现；没有给活动训练/跟踪源代码添加计时探针。
