# Relative M2 full TRAIN 失败诊断与下一次最小训练提案

日期：2026-11-02。结论：**真实训练已经完成并通过终态文件/保留权重验证，但 12 个 epoch 全部未通过保护与恢复 gate；不能进入 online integration。** 保留 protected E0，`best_qualifying_epoch=null`、`online_allowed=false`。本文件是工程失败诊断和待审 TRAIN 提案，不是新的 SOURCE_PASS、科学验收或启动许可。

## 1. 本次证据与验证边界

本次依据本地真实归档，读取了 acceptance、完整 history、config、frozen control、COMPLETE、native status 及 `reports/E0_M2_RELATIVE_FULL_TRAIN_FINDINGS_20261102.md`。以下 SHA256 均按本次读取的文件 bytes 计算。

| 证据 | SHA256 |
|---|---|
| `reports/E0_M2_RELATIVE_FULL_TRAIN_ACCEPTANCE_R1_20261102.json` | `5fdefad2fb8f12ad269befa85c22aa00711beed575db9b508ec3b0aafd4af2e5` |
| `reports/e0_relative_full_training_r1_20261102/plan.json` | `2af5ac9854c6da986d9ba42274f56136793f8348f13b73ccb8b738c6831079db` |
| `terminal/history.json` | `6c1279bae1e4492af2d85374ecf10079470e4e58417461b80f6e78f3d0fbd445` |
| `terminal/config.json` | `eb74324b3b8f28bda73af7a7877fedf7fcc752e0e2de791e46b67d546e9c6e6a` |
| `terminal/frozen_control.json` | `eb164330dad5c0514ff4c708358f99e1c52b055e0e0a4fdb662a4448775beade` |
| `terminal/COMPLETE.json` | `ae94a7e71321954116917c909e72ae98928d5117ec4fd61a7d61142412a9bce6` |
| `terminal/native_status.json` | `c1ac0f6d48241cd8c8e2724b97e7c098623a1680142fa8b0a689da354bec1d01` |
| `reports/E0_M2_RELATIVE_FULL_TRAIN_FINDINGS_20261102.md` | `fd9a9c9cb5ae6ca7b602535372781b33216b8e2cb6b40f03a31f54c4b5fa7d8b` |

上表 `terminal/` 均指 `reports/e0_relative_full_training_r1_20261102/terminal/`。五份 terminal JSON 的 bytes 分别匹配 acceptance 的 artifacts/native hashes；config 中的 plan 与实际 plan 完全一致。另逐项比较 history 的 epoch 8-12 与 acceptance 的 CPU replay，除允许存在数值舍入差异的 loss 外，所有决策指标完全一致。

acceptance 状态为 `RELATIVE_CURRENT_E0_FULL_TRAINING_VERIFIED`。其含义是终态执行、文件和保留权重的重放验证成功，**不是训练策略合格**。该 verifier 已重放 frozen initialization、保留的 epoch 8-12/latest；epoch 1-7 权重已按原保留策略删除，其 history 只有结构验证。本次没有再次加载权重或重放全部缓存，不能声称独立重放了已删除的早期权重。

真实 task 为 `train_e0_relative_full_r1_p26_20261102`，原始运行根为 `/data2/gb/Dual-RGBT`，native `COMPLETE/exit0`。训练共 12 epoch、348 次更新，头部总参数 116482，engine receipt 用时约 50.28 秒。这是缓存特征上的头部训练，`protected_e0_loaded=false`，不是新的 E0 端到端训练。归档中的原 GPU 编号不构成当前资源状态；本次未查询 GPU。

## 2. 失败发生在哪里

数据为已验收的 882 条 TRAIN sequence、459435 帧、18711 个候选集合。既定 component split 下，fit 为 706 sequence / 14762 sets，holdout 为 176 sequence / 3949 sets；holdout 含 reliable 2899、lost 800。

原 gate 保持不变，见 `tools/relative_e0_train_data.py:63-68`：

- reliable 状态从 IoU >= .5 变为 < .5 的数量必须等于 0；
- reliable sequence-macro IoU delta 必须 >= 0；
- lost 状态从 IoU < .1 恢复到 >= .5 必须 > 179，即至少 180/800。

这里的 179 是预先冻结的资格基线；不能改为 0，也不能在看到 frozen relative 的 197 后替换资格基线。全 KEEP 可满足保护部分，但恢复为 0，仍然不合格。

以下使用原 history 的训练设备评估 loss；CPU terminal replay 的决策计数相同，loss 有很小的数值差异。`H` 为 reliable 损伤数量，`R` 为 lost 恢复数量，`可靠宏 ΔIoU` 为 reliable sequence-macro delta。

| epoch | fit loss | held loss | fit H | held H / 2899 | fit R | held R / 800 | held 可靠宏 ΔIoU | 合格 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| frozen | 2.159810 | 2.205623 | 833 | 306 | 1126 | 197 | -0.082746 | 否 |
| 1 | 2.021724 | 2.125970 | 806 | 332 | 1145 | 199 | -0.088565 | 否 |
| 2 | 1.967755 | 2.116964 | 822 | 359 | 1199 | 199 | -0.086044 | 否 |
| 3 | 1.920013 | 2.106006 | 789 | 361 | 1216 | 195 | -0.084387 | 否 |
| 4 | 1.873859 | 2.110632 | 761 | 347 | 1241 | 193 | -0.080207 | 否 |
| 5 | 1.829002 | 2.110736 | 702 | 344 | 1271 | 198 | -0.075390 | 否 |
| 6 | 1.790719 | 2.109130 | 650 | 338 | 1298 | 182 | -0.068314 | 否 |
| 7 | 1.737380 | 2.115337 | 616 | 351 | 1356 | 189 | -0.072708 | 否 |
| 8 | 1.680954 | 2.124661 | 551 | 353 | 1383 | 178 | -0.070579 | 否 |
| 9 | 1.633150 | 2.143887 | 535 | 392 | 1396 | 167 | -0.076702 | 否 |
| 10 | 1.572870 | 2.158036 | 519 | 389 | 1444 | 178 | -0.077950 | 否 |
| 11 | 1.517610 | 2.172778 | 458 | 389 | 1453 | 173 | -0.076642 | 否 |
| 12 | 1.459788 | 2.203595 | 500 | 419 | 1545 | 169 | -0.085241 | 否 |

由此可以确认：

1. 早停不是现成解法。epoch 3 的 held CE 最低，但仍有 361 次可靠损伤；epoch 1-7 恢复数达到门槛，却全部损坏保护条件。不能挑一个“相对较好”的失败 epoch 推进部署。
2. 存在后期泛化恶化：fit loss 持续下降，而 held loss 在 epoch 3 后总体回升，最终几乎回到 frozen 水平；held 损伤从 frozen 的 306 增加到 419，恢复从 197 降到 169。
3. 问题不只是 holdout 上的过拟合。epoch 12 的 fit 本身仍有 500/10073 = 4.96% 可靠损伤，可靠宏 delta 为 -0.021286。fit 平均 IoU delta +0.081320 和 lost 恢复 1545/3477 = 44.43%，掩盖不了该保护失败。
4. epoch 12 的 held 可靠损伤率为 419/2899 = 14.45%，lost 恢复率 169/800 = 21.125%；整体宏 delta 为 -0.050357。held 替换 3181/3949 = 80.55%，fit 替换率为 73.61%。高替换率是行为描述，不足以单独证明因果机制。

## 3. KEEP 有真实证据，但没有独立的动作效用语义

`dual_rgbt/e0_relative_identity.py:14-23` 把 current descriptor 放在 24 个候选前面，与候选一起送入继承的 matcher，再丢弃旧 scalar `no_match`。因此“KEEP 没有输入”或“仍然只是一个 scalar no-match”都不准确。

实际结构为：

```text
KEEP score       = m(anchor, current)
candidate j     = m(anchor, candidate_j)
decision        = argmax(KEEP score, candidate_1 ... candidate_24)
```

`dual_rgbt/e0_identity.py:47-68` 的共享网络先将 768 维描述子投影为 128 维并归一化，使用与初始 anchor 的绝对差和逐元素乘积，经共享 MLP 得到每模态分数，再按完整 coverage 求平均。候选分数不接收 current 描述子、current score 或当前运动状态；current 只在最终比较时与候选竞争。

这能比较“谁更像初始目标”，却没有显式学习“替换这个当前状态的收益与风险”。完全相同的 current/candidate 证据会打平，index 0 的 KEEP 优先，这是正确的保守 tie 行为；问题在于真实近似、遮挡、相似干扰物和不同 crop 的 descriptor 可以不同，而分数差并未被定义为定位收益。不能仅凭这些汇总指标断言 descriptor 在信息论意义上不足，或确认某个具体干扰物机制。

旧 `no_match` 参数被冻结且不参与当前 action logits，见 `e0_relative_identity.py:12,19`。因此调这个旧 scalar 无法修复本次动作判断；增大学习率、重复 seed 或继续训练也没有现有证据支持。

## 4. CE oracle 与 gate 不矛盾，但优化风险不对齐

`relative_action_targets` 在 `dual_rgbt/e0_relative_identity.py:26-40` 选出满足 coverage、IoU >= .5 且严格优于 current 的最佳候选，否则 KEEP。对 reliable current，oracle replacement 一定提高 IoU。**若该 oracle 被完美预测，它并不违反保护 gate。** 不能把失败直接归为“训练标签要求破坏可靠状态”。

实际不对齐在 surrogate 和错误代价：

- reliable current IoU .800000、候选 .800001 时，CE 仍把 KEEP 当作错误类别；它要求一个几乎无收益的替换，而 gate 首先要求不损伤。
- 单一最佳候选 CE 没有把“另一个安全候选”和“使 reliable 跌破 .5 的候选”赋予不同的定位风险；它们都不是目标类别。
- 按 set 平均的 CE 不等于 sequence-macro 保护目标。长 sequence 的更多采样集可以对训练占更大权重，不能用平均 loss 下降推导宏保护成立。
- 24 个候选取最大值时，误替换的动作集合风险没有单独约束。仅提高最佳类别的概率不等于可靠状态零损伤。

### 已证实的训练/决策支持集差异

`tools/relative_e0_train_data.py:16-29` 将原 identity label 为 -1 的候选从目标生成及 loss denominator 中屏蔽；`tools/relative_e0_training.py:29-43` 的实际决策却直接对未按该 label 屏蔽的 logits 做 argmax。原 label -1 包括完整观测但 IoU 在 [.1,.5) 的模糊 identity，以及不完整观测，来源见 `scripts/collect_e0_identity_train.py:68-89`。

本次用实际源码中的 target/loss 函数做了一个小型 CPU 合成验证，无模型前向或 optimizer：current IoU .8，两个候选 IoU 为 .2/.800001，labels 为 `[0,-1,1]`，logits 为 `[0,1,0]` 与 `[0,100,0]`。两者 target 都为 2、loss 都为 0.693147，unknown logit 的梯度都为 0，但未屏蔽 argmax 都选择第一个候选。该例证明损失没有约束这个样本的 unknown logit；不代表真实训练中所有 unknown 分数的来源都相同。

真实 epoch 12 选择原 unknown 候选的次数为 fit 959、held 294；分别占替换的 8.82%/9.24%。这不是“这些候选全都错误”的证据，必须与具体 IoU/coverage 交叉统计。同时 held 419 次可靠损伤大于全部 294 次 unknown 选择，因此**至少 125 次可靠损伤来自原 label 非 -1 的选择**，unknown 差异不能解释全部失败。

不能把 GT identity mask 加到推断路径，也不能把所有 -1 伪造为负 identity 标签。若新任务改为预测定位收益，应明确区分“identity 未定”与“已执行框具有可计算的当帧 TRAIN IoU”，另建动作效用监督，保留原 identity labels。

## 5. 时序、几何与泛化的证据

训练的五个输入仅为 anchor/current/current coverage/candidates/candidate coverage，见 `tools/relative_e0_training.py:13-17`。数据加载只保留九个字段，见 `tools/relative_e0_train_data.py:80-104`，未输入 current/previous box、candidate box、crop geometry、候选 detector score、E0 route 或历史。

这些信息已有缓存，不必先重新收集：

- `scripts/collect_e0_current_full.py:106-120` 已保存 current/previous box、normalized box、geometry、914 维 observation 及 route。
- `dual_rgbt/e0_observation.py:39-61` 的 observation 使用当前图像和传入的历史，包含 preview、ROI、coverage、统计、框及模态；它不是未来轨迹。历史的具体位置需按实际 layout 解析，不能猜切片。
- `scripts/collect_e0_identity_train.py:93-135` 已保存候选框、score、query/modality、crop geometry 与 refinement parent。候选先执行，随后才读当帧 TRAIN GT，见 `:311-318`。
- current evidence 以 previous-state crop、固定初始 templates 编码当前 E0 框，见 `dual_rgbt/e0_current_evidence.py:8-33`；使用这些缓存时应继承该坐标及模板语义，不能混同 observation 中来自原策略路径的 primary ROI。

缺少联合上下文是明确的源码事实；“加时序几何一定能达到 gate”仍只是待检验假设。fit 的残余失败同时可能来自目标代价、支持集差异、外观可分性或模型表达能力，不能从当前表格排出唯一原因。

当前数据始终位于原 protected E0 轨迹上；缓存评估是单步反事实选择，不是替换后轨迹的闭环恢复。即使未来某个头通过这些 gate，也不能直接声称模板/历史干预后的长期恢复或 benchmark 提升。

既有 706/176 component 划分应保留，不能重新随机拆帧。上游 E0 见过这些 TRAIN 数据，holdout 也已用于逐 epoch 选择和本次诊断，所以它是 head 开发用的 TRAIN holdout，不是独立最终验证集。新训练不得使用 TEST 指标、TEST 标签、未来帧标签或目标名称特征。

M1 gate 也不能事后替代原资格条件。`dual_rgbt/e0_m1_relative_m2_online.py:41-68` 在 M1 判定后才调用 relative head；该预测 gate 不等于真实 lost 标签。将坏头按 M1 筛选后只报较好的子集，不能升级当前失败结果。

## 6. 下一次最小 TRAIN 提案：冻结外观，学习带上下文的动作优势

建议只做**一个新候选、一个 seed、一个预先锁定预算**。不从拒绝的 epoch 12 继续微调，不改 E0/candidate collector，不做 seed/LR/阈值网格。以下是具体待审设计，尚未实现或获准运行；它同时改变动作表示和损失，若改善也不能声称已由单因素消融证明原因。

### 6.1 训练前只做 CPU 数据诊断

在原缓存上按 fit/held 分开输出：

1. 现有 frozen 和保留 epoch 的逐 sequence/component 损伤与恢复，区分可靠轻微下降、跌破 .5、lost 恢复；报告 score gap。
2. 原 unknown label × coverage × current 状态 × 被选框 IoU 的交叉表，核对损伤集中度；不将 unknown 自动记为错误。
3. lost 集合中可观测候选 IoU >= .5 的 oracle 数量、relative 最佳候选可用率、reliable 安全候选/KEEP 分布。若 held 可用 oracle 不足 180，当前候选集合无法满足恢复 gate，应停止该方案，保留 gate。
4. 新输入字段的 frame/prediction/crop/template 对齐、有限值、原坐标系和因果来源；仅用 fit 估计连续特征标准化量。禁止把 current/candidate IoU、identity labels、sequence index 或未来结果送入网络。

以上真实缓存交叉表尚未在本次重新计算，不能用合成例子或汇总指标替代。它们是提案进入训练前应补齐的 CPU 证据，不是本文件隐含的已通过检查。

### 6.2 最小模型与 KEEP 定义

冻结原 epoch-9 identity checkpoint 的全部外观权重，初始化 SHA 为 `7e7df5d25d3e8667fe38a4ca8f6ec3662308ffcb52a09bd16aecaab59a9cb4be`。冻结 E0、anchors、24 候选及其顺序。只训练一个共享的 `28 -> 32 -> 1` GELU MLP，共 961 个参数。

每个候选 j 的 28 个标量输入固定为：

| 输入组 | 维数 | 来源 |
|---|---:|---|
| frozen current 分数、candidate 分数、两者之差 | 3 | 现有 matcher，完整 coverage |
| candidate detector score | 1 | 原候选缓存 |
| current/candidate 的双模态 coverage | 4 | 已验收缓存 |
| current、candidate 各自相对 previous box 的中心位移与 log 尺寸比 | 8 | 同帧/前帧预测框，位移按 previous 宽高归一化 |
| 传入 E0 的上一状态 history | 8 | 已缓存 observation 的真实 history layout |
| 当前 E0 modality/action/confidence | 3 | 已缓存 route |
| candidate 与 current 的 box IoU | 1 | 两个预测框的几何重叠，不是与 GT 的 IoU |

令 MLP 输出 `a_j`，其语义是预测“相对 KEEP 的 IoU 优势”。KEEP logit 固定为 0；从 `[0,a_1,...,a_24]` 做 argmax，tie 选 KEEP。输出层零初始化，使初始模型确实全 KEEP；不是给旧外观 logit 直接加一个 residual 后沿用已知损伤的初始策略。

current 没有任何完整模态观测时仍 KEEP；candidate 没有完整模态观测时在训练和推断一致地排除。这个 eligibility 只使用在线可用 coverage，不使用 TRAIN GT 或 identity label mask。连续特征按 fit 统计标准化；离散 route/coverage 保留明确编码。

只使用 history/route/几何与已有外观分数，不先训练整个 914 维新表征或扩大 backbone。这是对现有证据下改动范围的限制，不是已经证明这 28 个标量充分。

### 6.3 明确的监督与保护项

设 `q0` 为当帧 TRAIN GT 对 current 的 IoU，`qj` 为同一 GT 对已执行且完整观测候选的 IoU。动作监督使用真实 `delta_j=qj-q0`，不再强制分类到唯一最佳候选。

```text
L_adv     = mean SmoothL1(a_j, qj - q0, beta=0.1)
L_protect = mean softplus(a_j)
            over reliable current (q0 >= .5) and degrading candidates (qj < q0)
L         = L_adv + L_protect
```

两项先在每个 set 内平均，再在每条 sequence 内平均，最后对 fit sequences 等权；保护项只在具备该监督的 set/sequence 上聚合，无该项时贡献 0。系数固定为 1，不在 holdout 搜索。保护项惩罚可靠状态的所有预测退步，而不限于跌破 .5，有助于与可靠宏 delta 的要求对应。

对完整观测且原 identity label 为 -1 的候选，仍可用已经执行的框和当帧 TRAIN GT 计算连续定位优势。这是一个**新声明的定位效用监督**，不是把 -1 改为负 identity；原 labels 保留。无完整观测的候选不补造标签，训练与推断使用一致的 coverage 排除规则。

推断只读预测特征和 MLP 输出，不读取 `q0/qj`、损失分组或监督标签。决策边界 0 来自 KEEP 的效用原点，提前固定，不能在 held 或 TEST 上调节。该可微保护项不是零损伤保证；原经验 gate 仍逐项决定是否拒绝。

### 6.4 固定一次训练和验收

- 保留同一 706/176 split、14762/3949 sets、初始 anchor、24 候选、采样 cadence 和 seed 2027；仅 fit 参与梯度与标准化。
- 单次 AdamW，lr 3e-4、weight decay 1e-4、batch 512 sets、12 epoch；不加额外训练轮数。sequence 权重按完整 fit 的 set 数计算，避免每个 minibatch 重新定义权重。
- 记录零初始化全 KEEP control、原 frozen relative control 和全部 epoch 的三项 gate、原有 IoU 指标、unknown 交叉表、eligible oracle 及损失分项。头部总/可训练参数与冻结外观参数分别列明。
- 仍只允许在原三个 gate 全部成立的 epoch 中按原 `acceptance_rank` 排序，见 `tools/relative_e0_training.py:74-78`。若 12 轮仍无合格头，保留 protected E0，停止该次提案；不得事后减掉困难 sequence、只验 M1 子集、降低恢复数或容许可靠损伤。
- 保存 latest、最后五个 epoch 和合格 best；沿用终态 CPU 决策重放。明确区分 `execution_verified` 与 `qualifying_head_available`。早期权重若仍删除，就继续明确其指标未独立重放的限制。

该方案的核心实验问题是：在不增加 encoder 查询、候选或数据采集的情况下，显式动作收益监督与很小的联合上下文能否消除训练内可靠损伤并泛化到既定 TRAIN holdout。没有保证它会通过 gate，也不将未通过的头包装成可部署版本。

### 6.5 实施范围与后续准入

实施应使用新的 additive context-head、data/loss、训练 entry/contract 和测试文件，生成新的 disabled draft/source map、task/run 与终态 receipt；现有 sealed 训练和 runtime 源码保持原样。具体文件名应在实现时一次确定，不在本诊断中伪造已存在的入口或 launch plan。

必要 CPU 测试应覆盖：KEEP 零初始化/tie、完整 coverage 的训练推断一致性、unknown identity 与连续效用监督的区分、标签不进入 forward、原 gate 拒绝样例、frozen 外观权重及 fit-only 标准化。新源审和真实 update smoke 在获准训练前完成。

即使新头离线通过 gate，也需要单独审查适配后的 context runtime，并做独立模型实例的 paired TRAIN 闭环验证，观察首次替换后恢复持续性、模板/历史状态及完整成本。当前五输入 runtime 不会自动兼容 28 维 context 设计，现有 online review 不提供新版本准入。不得直接跳到 TEST 选型。

## 7. 诊断依据的源码绑定

以下实际读取的核心文件均与本次 full plan 的 sealed source map 一致。本次只新增此文档，没有修改这些文件或旧报告。

| 源码 | SHA256 |
|---|---|
| `dual_rgbt/e0_relative_identity.py` | `07be9e173e1881d27c24cc896e5bd3ee5226c3aff6e36cd282aec725a37c41a0` |
| `dual_rgbt/e0_identity.py` | `89b15dea17514e5516dfe43e76295662ae47da8f98ea590e3fc5565925278669` |
| `tools/relative_e0_train_data.py` | `477a8430e2d359817e23bda78b5ff79b53cf03085826b79fd93619eccb0500b7` |
| `tools/relative_e0_training.py` | `6bb3fcb00f603c1c7fe9e83b783061132006fea7d1ecef0040fef52d20182d53` |
| `scripts/train_e0_relative.py` | `13b5efa102e4390fa08af33a3387a47c20ff4b7af7deca18c8eda6965686b69f` |
| `scripts/collect_e0_current_full.py` | `3f1098523c68d8833a368f44ca1234cb724da6e24df1571a3ff7c740de12faf7` |
| `scripts/collect_e0_identity_train.py` | `78009e849c4618416b680e451d0d2f6d81754ddbbf00f13fc343524596aac9eb` |
| `dual_rgbt/e0_observation.py` | `3fe991f78db09e1a065b9fc1a7d6ed01940dae934bd57968c156232f6a089bdb` |

已验收 current collection SHA 为 `ebacf099dc881c0b11fc3c26cd7da01afcad40fc34006ade563af30c0f6b7ab7`；collection plan SHA 为 `61f3dc02d5727093740547666cbf2af95765efa683736f3fa074e7cc7167feef`。这些绑定来自真实 full plan/receipt；本次未重新遍历原始图片、全部缓存或 checkpoint bytes。

模型归属与结论上限：`actual_model=UNATTESTED`，same-family、provisional；请求的 `gpt-6-astra` 路由未在工具 schema 暴露，未执行或声称 Astra 审核。`scientific_result_audit_accepted=false`，本文件不替代正式 experiment audit。未授权或执行任何新的部署、GPU 查询、GPU 训练、benchmark/TEST 选择或其他 chat 通信。下一次训练仍需新的具体实现及准入流程，本诊断本身不允许启动。
