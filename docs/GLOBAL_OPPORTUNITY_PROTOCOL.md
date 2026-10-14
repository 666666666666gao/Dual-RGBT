# 全图候选与修正器收益诊断

SOURCE_PASS / SANITY_STARTED，2026-10-14 10:23 BJT。原正式 A 已真实完整退出0；其 source30不变。

R1 strict-reload clone未显式关requires_grad的冻结契约阻断已保留；没有发生更新的证据。
最小两行修正后R2源检查29 PASS/0 FAIL、真实exit0，同family/provisional、实际routing UNATTESTED。
仅新driver/plan部署，source31及CUDA隐藏CLI实际exit0。GPU1只读10帧sanity已单次启动，准入尚待实际终态。
必须实际核验10track/2状态/28比较、265冻结state、七项strict probe、0 backward/optimizer和原生exit0。

此检查使用已完整训练的全图 A 或 B 正式权重；必须原生任务 COMPLETE/exit0、50000对/6250更新、sanity=false。
运行命令必须提供实际 checkpoint SHA。M0权重禁止进入此诊断，诊断不训练或产生新权重。
先独立源码审查，再10帧只读GPU检查：全部state不变、无梯度/优化器、实际STOP反事实逐位复现、strict重载7项预测一致。
执行器的梯度/Adam准入已在同源全图A10/B2/C2中实际完成，本诊断必须保持0 backward/0 optimizer。

## 状态与对照

固定seed2027，原train顺序前20条，完整跟踪8854帧，每五帧正宽高标注监督，共1752个状态。
参考轨迹使用现有廉价主模态调度，但强制STOP，只执行所选主模态完整编码。
首帧GT初始化，之后预测决定搜索和主模板更新；未读取辅助不更新辅助模板。
这不是最终学习策略的轨迹，也不是独立RGB/TIR轨迹；只是明确可复现的同历史/同状态访问分布。
每次比较保存跟踪前template/history/box，实际所选STOP反事实必须逐位重现当前预测。

每个状态下比较两种主模态各五条执行路径，共10结果：
STOP、LOCAL_MAIN、LOCAL_ALTERNATE、GLOBAL_LOCAL、GLOBAL_WIDE。
另对实际所选主模态比较四个控制，共14结果/状态、24528对：

|控制|源区域/重型输入|要回答的问题|
|---|---|---|
|DENSE_SAME_SEARCH|完整辅助主搜索源/256|现有信息是否必须完整辅助编码|
|RANDOM_LOCAL|独立均匀随机中心、同局部边长/128|候选是否优于同token预算随机读取|
|GT_CENTER_LOCAL|GT中心、同局部边长/128|正确查询位置下局部修正器有没有能力|
|GT_CENTER_WIDE|GT中心、同宽查询边长/256|局部输入不足时更大上下文是否有效|

完整辅助控制仍是本模型共享执行器，不重测用户CADTrack baseline精度。
四控制仅在实际选中主模态评价，不补称“两个主模态全部控制”或联合最优oracle。
GT中心查询只做不可部署的定位诊断；它不影响之后的参考状态、模板、路由或任何训练。
它不是候选空间的最优动作，也不是完整轨迹性能上界。多个区域收益不能相加。
所有收益为uncensored GT与margin10/float32实际执行框的原图IoU差；不把距离变小称IoU恢复。
输出每个动作实际IoU、收益分布、源区域GT覆盖以及主源内/外状态，便于区分候选与执行器问题。
CSV delta是已计算float32 IoU读回后在Python相减；不声称与训练float32差值逐位相同。

## 证据与后续决定

模型全eval，265个state在诊断前后必须逐位不变，全部参数grad=None，checkpoint/初始权重/源哈希不变。
config/SANITY/COMPLETE保存原字节；完整action_comparisons.csv保留服务器/本地私有，不能上传Git。
公开只发布代码、计划、聚合结论与审查证据，不发布GT中心的查询几何或原始annotation。
轨迹总耗时只用于诊断运行安排，不作为部署系统FPS/延迟数据。

如果GT中心局部/宽控制没有收益，先查修正器或区域尺度，不继续用门控训练掩盖问题。
如果GT中心有效而当前global候选覆盖低，下一步训练/改进廉价候选；A阶段global预览未训练，不能据其当前结果否定方向。
若辅助有益但学习策略不如同预算简单规则，才进一步处理收益预测。B完成后同协议再检查候选及收益变化。
诊断不是最终目标；完整882因果训练、dev97预算校准、test245 PR78/NPR72/SR62与实际效率验证必须继续。
