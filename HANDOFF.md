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
环境迁移完成，GPU0真实M0会话61116退出0：10组合实际更新、有限梯度、
严格重载逐位一致、STOP仅385个token主路径，局部辅助193个token。
耗时15.93秒，峰值2999788544字节；这是执行验证，不是推理提速指标。
独立代理逐字复现文档命令：GPU1会话80288退出0、同项检查PASS，耗时14.78秒。
独立源码审查PASS，0阻断/0非阻断，同family/provisional、实际模型路由不可独立确认。
源代码修复裁剪扰动与逐样本收益/区域覆盖记录，细节保留在审查报告及问题记录。
数据实测882 train/97 dev/245 test，帧数/GT审计issues为空。
阶段A于服务器时间2026-10-13 13:16:41单次启动：GPU0，真实PID3035566，
源代码提交4ed99e2080fa9382a351c615b731449fd1987b6a；
2轮×2000样本、1000次真实AdamW更新，每种模态/动作100次。
13:17启动核查确认该PID实际命令与CUDA_VISIBLE_DEVICES=0一致；
已完成第50步（至少51次更新），该步loss=0.847779、训练样本mean IoU=0.825947。
这些是训练批次诊断，不能替代LasHeR PR/NPR/SR。日志为
`runs/stage_a_diagnostic_20261013.log`，记录与逐样本结果在同名run目录。
按已观测每步时间0.127秒和额外校准，粗估数分钟；下一次观察不早于13:20:30。
阶段B脚本与闭环Tracker仅准备，源码审查中，尚未执行；正式D评价尚未执行。
当前尚无新方法LasHeR PR/NPR/SR或完整延迟指标；目标保持未达成。

## 接续顺序

按预计节点查看原任务而不是重复启动；只读观察器`scripts/observe_stage_a.py`。
完成短程后读取逐样本收益、负收益、目标覆盖和损害数，先根据实际问题决定A充分预算，
再审查并推进B/C/D。不要将M0十步状态当作A初始权重；A从用户原epoch14开始。
不得跳过M0，也不得把短程IoU、oracle或GT维持搜索的结果当作LasHeR正式成绩。

问题、修复与证据见`docs/ISSUES.md`；实验顺序见`docs/EXPERIMENT_PLAN.md`。
每次源码/结果更新后，将本文件同步到上述服务器、本地文档与GitHub三个位置。
