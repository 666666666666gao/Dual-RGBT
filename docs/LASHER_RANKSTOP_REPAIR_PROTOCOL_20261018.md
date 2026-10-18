# LasHeR rank/STOP policy repair protocol（2026-10-18）

当前 E1 C 只用 quality 与 signed gain 的 SmoothL1。失败统计显示 STOP 与辅助查询的竞争是关键，但固定 RGB 和强制查询都不能修复 dev，因此不改执行器或网络结构。本实验仅改变 C882 轨迹训练目标：对每个 sampled query 的 signed gain 增加相对 STOP 的 margin 约束。

- margin：0.05；权重：0.25
- executor、backbone、preview、quality/gain/query head 接口和模块连接保持不变；只训练原有 quality/gain 头
- 目标：有益 query 的预测 gain 至少为正并超过 margin，有害 query 被压到 STOP 以下
- 训练：同一 882 条序列 × 2 rounds、seed2027、label stride5；先 sanity 再 formal

完整结果必须经过 LasHeR dev 三成本选择，再冻结后跑三数据集 test；若 dev 不超过 E1 原策略，停止该 repair。
