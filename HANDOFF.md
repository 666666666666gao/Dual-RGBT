# Dual-RGBT 交接文档

## 用户目标与约束

2026-10-13：25服务器仅GPU0、1；项目和conda环境在`/home/gaob/Dual-RGBT`。
优先LasHeR，目标PR≥78、NPR≥72、SR≥62，并明显降低完整在线推理成本。
用户确认去模块CADTrack baseline PR/NPR/SR=72.74/69.16/57.84，按要求不重测。
唯一初始权重`/home/gaob/CADTrack_ep0014.pth.tar`（epoch14，原文件不动）。

## 路径

- 本地代码：`C:/Users/gb/projects/Dual-RGBT`
- 服务器：`/home/gaob/Dual-RGBT`
- 环境：`/home/gaob/Dual-RGBT/conda-envs/dual_rgbt`
- 数据源：`/data2/wangwj/dataset/LasHeR-0428/trainingset`及`testingset/testingset`
- 仓库：https://github.com/666666666666gao/Dual-RGBT
- 本地文档：`C:/Users/gb/Desktop/document/Dual-RGBT交接文档.md`

## 方法

廉价预览决定主模态；完整主路径之后预测各辅助区域的定位收益；
从原图取少量patch再共享编码并修正候选。核心是收益请求与局部执行的连接。
主模态质量、辅助价值与信息预算分开学习。三个模块仍是待验证候选贡献。

## 已完成与当前进程

已建立本地仓库与服务器项目；找到数据、权重和去模块CADTrack源代码。
CPU权重结构读取成功，保存`reports/checkpoint_structure.json`。
旧环境clone失败；conda-pack API任务正在执行（本地原工具会话34465）。
尚无新模型训练、正式精度或延迟指标。原只读SSH会话73515限定GPU0、1。

## 接续顺序

完成home内环境迁移及真实GPU见证；核验数据与原权重严格加载；
实现与独立审查M0四路径、均衡A训练，完成短程运行后按实际问题推进B/C/D。
不得跳过M0，也不得把短程IoU、oracle或GT维持搜索的结果当作LasHeR正式成绩。

问题、修复与证据见`docs/ISSUES.md`；实验顺序见`docs/EXPERIMENT_PLAN.md`。
每次源码/结果更新后，将本文件同步到上述服务器、本地文档与GitHub三个位置。
