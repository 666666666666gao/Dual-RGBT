# Dual-RGBT

根据当前目标的定位需求，决定先完整处理哪个模态、是否追加辅助信息，以及只编码辅助模态哪些区域。

研究目标：LasHeR **PR≥78 / NPR≥72 / SR≥62**，同时降低实际在线推理成本。
这是一份进行中的实现，不代表目标已达到或三个模块的创新性已经验证。

当前已完成A执行器7250次更新，正在从A7执行器训练完整B，后续接续真实C、
完整97条验证与245条测试。详见[完整训练协议](docs/FULL_TRAINING_PROTOCOL.md)。
截至目前，新方法正式PR/NPR/SR仍待完成。

用户提供的去模块 CADTrack baseline：PR **72.74**、NPR **69.16**、SR **57.84**。
按要求不重测 baseline 精度。初始权重位于 `/home/gaob/CADTrack_ep0014.pth.tar`。

三个模块：廉价预览主模态调度；定位收益驱动的区域证据请求；共享主干的局部证据修正。
核心区别需要由实验验证：在辅助重型编码前预测区域定位收益，而非仅做置信度门控。

详见 [方法与实验计划](docs/EXPERIMENT_PLAN.md)、[问题记录](docs/ISSUES.md)
和 [交接文档](HANDOFF.md)。

`lib/`、`experiments/cadtrack/`、`tracking/` 为服务器 `/data2/zhouy/CADTrack-main`
的去模块版本源码副本。其原始作者项目为
[IdolLab/CADTrack](https://github.com/IdolLab/CADTrack)。原源码说明保存在
`docs/CADTrack_SOURCE_README.md`。新方法放在 `dual_rgbt/`，不改原模型前向。
