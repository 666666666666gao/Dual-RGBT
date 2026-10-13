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
阶段A于服务器时间2026-10-13 13:16:41单次启动：GPU0，原PID3035566，
源代码提交4ed99e2080fa9382a351c615b731449fd1987b6a；
2轮×2000样本、1000次真实AdamW更新，每种模态/动作100次。
13:29权威读取确认COMPLETE.json为epochs=2、steps=1000，PID已不存在，
CSV恰有1000条真实更新。后台启动未保存原生exit码，不虚构exit0。
这些是训练与校准诊断，不能替代LasHeR PR/NPR/SR。日志为
`runs/stage_a_diagnostic_20261013.log`，记录与逐样本结果在同名run目录。
第二轮同100个GT裁剪校准样本：RGB/TIR STOP mean IoU=0.689434/0.623036；
主候选局部补充为0.684997/0.619012，平均signed定位收益=-0.010859/-0.009632。
主候选ROI平均GT覆盖0.902/0.870，但尚无净改善；次候选分别纠正3/4、损害40/44。
原始逐样本CSV、步日志和两轮JSON已收回到本地reports/stage_a并保留服务器原件。
latest.pth为epoch2/1000；另存first_preserved.pth，实际核验也是epoch2，
只是第二轮副本，不宣称保留第一轮权重。原用户epoch14权重始终保留。
阶段B源码复审0阻断，但原生sanity和C真实轨迹尚未执行；正式D评价尚未执行。
固定epoch2证据诊断已完成，GPU1原生会话33980退出0，800个配对动作结果，
无优化器更新或新权重。池化归一化熵PRIMARY RGB/TIR为0.999060/0.998949，
真实摘要置零后loss反而低0.002779/0.005314；错配摘要只使框均值变化约0.0007。
次候选切换gain=-0.880998/-0.674639，修正gain=0.003180/-0.001131；
主要损害由未经验证直接切框造成。GT前景池化也未挽救固定MLP，不能只调温度。
已查阅OSTrack开源CenterPredictor：保持空间图并在响应峰处取size/offset。
下一步局部辅助token按原坐标填入辅助网格，与已算主特征复用原decode/head，
避免池化丢失空间定位及强制切候选；这仍是轻量空间融合。
空间执行器新上下文源码审查PASS，0阻断/0非阻断；真实M0与同预算A尚待执行。
新A保存latest及十路径dev定位loss最低的best；不兼容旧池化版state，
旧诊断与代码完整保留在41534b8及之前提交，不添加自动加载fallback。
当前尚无新方法LasHeR PR/NPR/SR或完整延迟指标；目标保持未达成。

## 接续顺序

原A与固定权重诊断已终态，不重复启动。按真实负收益修复局部空间证据读取，
先同预算重新训练执行器验证，再依据结果决定A充分预算，
再审查并推进B/C/D。不要将M0十步状态当作A初始权重；A从用户原epoch14开始。
不得跳过M0，也不得把短程IoU、oracle或GT维持搜索的结果当作LasHeR正式成绩。

问题、修复与证据见`docs/ISSUES.md`；实验顺序见`docs/EXPERIMENT_PLAN.md`。
每次源码/结果更新后，将本文件同步到上述服务器、本地文档与GitHub三个位置。
