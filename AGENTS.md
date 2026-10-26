# Dual-RGBT

- 先实现最简单、直接的正确方案；不加无证据的 fallback、try/except 或兼容层。
- SSH: `2025`；仅物理 GPU 0、1，每卡最多一个训练任务。不读取温度、功率。
- 项目、环境、输出均在 `/home/gaob/Dual-RGBT`。
- Python: `/home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python`。
- 初始权重：`/home/gaob/CADTrack_ep0014.pth.tar`，CADTrack 去掉论文三个模块、SOT 训练的 backbone，保留原文件。
- 数据使用项目内符号链接，指向现有 `/data2/wangwj/dataset/LasHeR-0428`。
- LasHeR 优先。用户提供 baseline PR/NPR/SR=72.74/69.16/57.84，不重测 baseline 精度。
- 当前接续 LasHeR 目标 PR≥78、NPR≥72、SR≥62（历史严格规划78.74/75.16/63.84保留记录），同时完成 RGBT210、RGBT234 和完整在线系统延迟；尚未达到。
- 保持整体架构及三个模块职责，允许按数据集调整模块内部参数、宽度或层数；匹配训练权重并登记验证选型、容量和延迟，测试前冻结配置。
- 不用 LasHeR test 标注训练、产生收益标签、选择阈值或选权重。
- 固定 seed 2027。长任务先估计诊断节点，临近节点查询，重复查询间隔180–300秒。
- 新实验先独立源码审查，再真实数据 GPU 前向、反向、优化器更新与重载检查。
- 交接文档：仓库 `HANDOFF.md`，同步到 `/home/gaob/Dual-RGBT/HANDOFF.md`
  和 `C:/Users/gb/Desktop/document/Dual-RGBT交接文档.md`。
- 代码、计划、问题与真实结果及时提交到用户指定 GitHub 仓库；权重、环境、原数据不提交。

code_sync: scp
wandb: false
