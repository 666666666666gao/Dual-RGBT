# 冻结动作收益结果的配对分析

2026-10-14新增实际结果：正式诊断完整结束、原生task及owner exit0；CPU分析93a7b3 exit0。
1752状态/14016对完整输入核验PASS，统计见TRAIN_ACTION_REWARD_RESULTS.md与
reports/train20_action_reward_analysis.json。以下NOT_RUN描述保留为原源码准入时的历史记录。

入口 scripts/analyze_policy_rewards.py 只读取已完成正式诊断的 config.json、COMPLETE.json、
action_rewards.csv，不导入 PyTorch、不连接服务器、不读图像或执行 NN。正式诊断目前未到启动节点，
所以只有源码和 CLI 检查，不存在本分析的实际收益统计。

源码 SHA `5439d4322d10c6ebd70527d074582453b9232dab1826fc58799ae2f44b4d5ef4`。
本地 CLI PID47124 真实 exit0、stderr空；analysis_executed=false。
后续在原队列收回正式 JSON/CSV 并证明 native exit0 后执行：

```powershell
uv run --python 3.12 python scripts/analyze_policy_rewards.py --run reports/full_pipeline/reward_diag_train20_20261014 --output reports/train20_action_reward_analysis.json
```

分析只接受20条/8854次track/1752个监督状态/14016个收益对的正式 COMPLETE，拒绝 sanity。
每状态严格验证RGB/TIR各action1..4共8行、相同路由/目标flag和每模态相同主误差；
重复行、动作缺失、不完整正式结果会直接失败。gain必须与保存的两项loss差完全一致，
输入三个文件 SHA 记录在报告中。计划、检查点、原生结束和未更新权重证据仍由原实验回执证明。

## 要回答的问题

1. **标签变化来自哪一步？**分别计算 clipped→uncensored（GT截断影响）、
   uncensored→executed（像素映射/边界约束/float32状态影响）及总变化。
   各模态/动作输出收益正值数、预测MAE、收益差和严格sign flip幅度分位数。
   未滤近零，不能把一个小数值符号变化直接解释成有效的策略改变。
2. **是否改变预算动作？**固定实际选择的主模态，比较旧clipped与实际executed误差下的
   STOP及四个动作诊断oracle，报告行动分歧、STOP↔query和执行效用regret。
   不改模态以为当前策略制造额外收益，不跨序列搜索阈值。
3. **当前访问状态有没有定位收益？**报告保存路线选中动作的执行loss/原图IoU改善、
   调用分布和平均辅助tokens，同时用另一主模态的实际主框误差作调度诊断。
   后者只是当前状态下的模态差，不是训练后切换模态的长期改善证据。

cost仍是原计划0.05，STOP增量0、local193/385、full1；oracle的标量计算用Python float64，
同值按STOP→action顺序选择。真实路由直接读取CSV，不声称逐位复现GPU float32 argmax。
regret是相同当下状态的诊断效用差，不是完整跟踪轨迹上界，不是可部署策略。

互斥目标分区为：四角都在源crop内、中心在内但角点超出、中心在外。
报告全体与各分区及逐序列统计。均值按监督状态/动作对加权，**不是toolkit序列等权SR**；
八个动作对来自同一状态，不能当作八个独立训练样本做置信区间。
这份报告不计算测试PR/NPR/SR、不估计真实延迟，也不证明修改监督后精度已经提高。

正式结果和对应源码审查通过后，再根据变化幅度、行动分歧及可获得的真实辅助收益判断
是否需要修改模块二监督；不提前改已封存的20seq测试或882完整训练。
