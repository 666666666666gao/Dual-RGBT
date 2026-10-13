# 冻结训练轨迹的实际动作收益诊断

M29 已发现 dev 搜索几何下存在被裁成零面积的 GT，但 dev 几何不能证明训练动作收益符号改变。
本实验在训练序列上保存同一状态的真实主框与四种辅助动作框，分别比较三种定位误差：

- 当前代理：原始模型提案与 clamp 后的目标。
- 未截断代理：同一个原始提案与完整目标，损失其余计算相同。
- 实际执行误差：提案经过 Tracker 原有像素映射、margin=10 的 clip_box 与 float32 状态写入后，
  用同一搜索坐标中的完整目标计算同一种 GIoU/L1 损失；另存原图 IoU 改善。

比较采用两模态各四种辅助动作，每状态八个收益对。所有反事实动作共用 track 前的模板、历史、
框与搜索裁剪；实际路线继续决定下一帧状态。GT 只用于离线诊断标签与初始化，不维持后续搜索。
原有 margin=10 执行约束未改；删除 GT clamp 本身并不代表已经得到实际执行收益。

## 固定输入与预算

计划为 reports/reward_diagnostic_plan.json，SHA
`f0edc89d98077c49a031dec0f86c4091e5a6255187ff6de0f881a9cb8d85374b`。
输入正式 B 为 runs/stage_b_refreshed_20261013/latest.pth，SHA
`ba49b506cd4bdcdfb355e8741feaf6c5c61e2edf9cd0a5c91459c3e5957c80ed`。
manifest SHA `389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc`。
seed2027、成本0.05、stride5、首20条固定训练序列，与原 C 及全882计划前20条一致。
一轮完整轨迹预期20次初始化、8854次track、1752个有效监督状态、14016个动作收益对。
1761个stride5候选中9个GT无效而不产生标签；不跳过其正常跟踪。

实验是实际 NN 前向，**0 optimizer / 0 backward / 0 新权重**。固定 B 在 eval/no_grad 下执行，
开始与结束比较全部265个 state 张量及输入权重 SHA。未实施收益监督修复或状态写入机制。
冻结 B 的轨迹不是原 C 更新过程中每一步的重放，也不能作为完整新方法指标。

## 准入与队列

独立 SOURCE R1 发现 manifest 键名错误；已仅修改为现有 test_sequences 名称列表，
R2 PASS_SOURCE_ONLY/0 blockers；保持 R1 FAIL 历史，见 REWARD_DIAGNOSTIC_REVIEW.md。
新两脚本加原封存17项，共19项 SHA 固定；不修改原参考测试或全882队列的17项源码。
使用已准入六通道 crop API，独立进程在导入 Tracker 前替换别名；没有采用未准入 sum 候选。
远端计划19项、B、manifest 和计划文件 SHA 已在部署时核对一致。
原conda CUDA隐藏 --help 与本地队列 --help 已真实 exit0；这不证明 NN 实验已经通过。

唯一新队列 scripts/queue_reward_diagnosis.py 只写 reports/reward_diagnostic_status.json。
首次检查为 **2026-10-14 04:30:50.010 BJT**，原参考控制器必须 COMPLETE_PIPELINE 后才允许入场。
已知失败或原 owner 消失则停止，不自动重启；仍运行时240秒复查。
只用物理GPU1，launch要求该卡 memory.used<500 MiB；全882主训练仍用GPU0。
不提前查当前参考 test 的帧进度、精度或 GPU 状态。

首先运行 reward_diag_input_sanity_20261014：10次track、2个标签、16对收益，预估小于一分钟。
验证全部265张量保持、实际选中框逐位等于 Tracker 状态、GT四角在范围内时两代理逐位相同。
再严格加载同一原 B，验证 predict 的模态、动作、框及 gains 逐位相同。没有优化器，
因此不执行反向或优化器更新检查。实际 NN sanity 尚待队列在参考测试结束后执行。
通过后另起进程从同一原 B 做 reward_diag_train20_20261014，不使用 sanity 的状态。
正式预算预计20–30分钟，首次检查1800秒后，必要时240秒；只收回 JSON/CSV，不收图像或权重。

## 解释范围

raw sign flip 定义 clipped_gain*executed_gain<0，未滤除近零数值波动。
报告同时保存收益幅度与差异；不能把任意符号变化直接叫作有意义的策略翻转。
这个实验能定位当前代理与真实执行目标是否不一致；不能证明更换标签会改善长期轨迹，
不能代替完整训练、最终245序列 PR/NPR/SR、同设备整机效率或三模块消融。
