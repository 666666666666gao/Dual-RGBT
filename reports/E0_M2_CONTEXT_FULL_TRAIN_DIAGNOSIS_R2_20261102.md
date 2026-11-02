# Context R2 实际完训诊断

诊断日期：2026-11-02。结论：实际训练执行成功，候选资格失败；停止原样重复此 Context 配方，保留 protected E0。本文件是已完成 R2 的工程诊断和一个待验证的 TRAIN-only objective 建议，不是新实验准入、科学接受或在线部署许可。

归因标记：`actual_model=UNATTESTED`、`review_independence=same-family`、`acceptance_status=provisional`、`scientific_result_audit_accepted=false`。当前工具未暴露请求的 `gpt-6-astra` 路由，不声称 Astra 或独立模型家族审计。

## 真实执行与失败

读取了实际 findings、terminal acceptance、recovery-gap audit，以及本地归档的 plan/config/history/COMPLETE/native/initial control。检查这些 JSON 的 SHA 和相互绑定，没有重跑 verifier、scorer、归因脚本或模型前向，没有查询 GPU 或联系其他聊天。

实际 full 位于 P25 的 `/home/gaob/Dual-RGBT/snapshots/context_advantage_r2_20261102`，task 为 `train_e0_context_full_r2_p25_20261102`，run 为 `runs/e0_context_full_r2_p25_20261102`。native 记录 GPU2、`COMPLETE/exit0`；真实 full 为 12 epochs、348 updates。此前真实 smoke 为 5 updates、`COMPLETE/exit0`；full 从新的 zero-KEEP head 开始，不是继续 smoke 权重。输入为已接受 TRAIN tensors：fit 14762 sets / 706 sequences，held 3949 sets / 176 sequences，每 set 24 候选、28 特征。

只训练 961 参数的 context head；116482 参数的 appearance scorer 已冻结。归档包含参数确实改变、finite gradients、fit-only normalization、strict reload bit equality。执行成功不能替代资格成功：`best_qualifying_epoch=null`、`retain_protected_e0=true`、`online_allowed=false`。

下表来自原始 `full_terminal/history.json`；loss 是每轮训练过程中累计的均值，不是 latest checkpoint 的固定全分区重放损失。

| Epoch | 训练 loss | Held harmful / 2899 | Held recovery / 800 | Held replacements | 合格 |
|---|---:|---:|---:|---:|---|
| 1 | 1.116576 | 1 | 1 | 3 | 否 |
| 2 | 1.023077 | 1 | 1 | 3 | 否 |
| 3 | 0.893888 | 1 | 1 | 3 | 否 |
| 4 | 0.739826 | 1 | 2 | 7 | 否 |
| 5 | 0.661411 | 2 | 4 | 16 | 否 |
| 6 | 0.639648 | 5 | 21 | 85 | 否 |
| 7 | 0.622184 | 7 | 46 | 264 | 否 |
| 8 | 0.608096 | 12 | 63 | 647 | 否 |
| 9 | 0.597573 | 14 | 66 | 912 | 否 |
| 10 | 0.589736 | 13 | 69 | 1083 | 否 |
| 11 | 0.583782 | 11 | 63 | 1158 | 否 |
| 12 | 0.578999 | 12 | 52 | 1251 | 否 |

原资格 gate 完整保留：held reliable harmful crossing 必须为 0，reliable sequence-macro IoU delta 必须非负，lost recovery 必须大于 179，即至少 180。定义见 [relative_e0_train_data.py:63](C:/Users/gb/projects/Dual-RGBT/tools/relative_e0_train_data.py:63)。12 轮全部失败，不能挑一个仍不合格的 epoch 进入下一阶段。已删除的 epochs 1-7 只有归档 history/结构核验；epochs 8-12/latest 有既有 terminal CPU replay，本次不声称再次独立重放。

Epoch12 的 fit 为 harmful 56、recovery 245/3477，reliable macro delta 为 -0.0024068184；held 为 harmful 12、recovery 52/800，即 6.5%，reliable macro delta 为 -0.0018108419。Held 总体 sequence-macro delta 虽为 +0.0071965706，也不能抵消可靠子集退化或恢复 gate 失败。Epoch8→12，held replacements 从 647 增至 1251，recovery 从 63 降至 52；fit replacements 从 2466 增至 4700，recovery 从 306 降至 245。因此不能仅以“整体过于保守、增加替换即可”解释后期失败。

## 实际恢复缺口

以下直接引用已完成的 recovery-gap audit；本次没有重新计算该 audit。

| Latest checkpoint 的 TRAIN 诊断 | Fit | Held |
|---|---:|---:|
| Lost 状态总数 | 3477 | 800 |
| 至少一个已观察候选 IoU >= 0.5 的 lost 状态 | 1810 | 374 |
| 上述状态中选择 KEEP | 952 | 187 |
| 上述状态中选择错误候选 | 613 | 135 |
| 上述状态中实际恢复 | 245 | 52 |
| 上述状态中所有正确候选 advantage 均 <= 0 | 1476 | 293 |
| 存在正 advantage 正确候选，但被错误候选压过 | 89 | 29 |

两种分解各自闭合：held 的 187+135+52=374，以及 293+29+52=374；fit 对应为 952+613+245=1810，以及 1476+89+245=1810。它们不是可以相加的两组错误类别。尤其“正确候选均非正”不等于选择 KEEP，错误候选仍可能得到正分并被选中。具体定义见 [audit_context_e0_recovery_gap_r2.py:43](C:/Users/gb/projects/Dual-RGBT/tools/audit_context_e0_recovery_gap_r2.py:43)。

在已有可恢复候选的状态上，fit 有 81.55%、held 有 78.34% 的状态未给任何正确候选正 advantage。实际恢复率占各自可恢复状态的 13.54% / 13.90%。恢复候选并非完全缺失，且分数问题同时存在于 fit，不能仅归因于 held 泛化。反过来，held 另外 426/800 个 lost 状态在当前已观察候选集合中没有正确候选；该固定集合的 374/800 只是逐状态可用性上界，不是可实现的闭环跟踪恢复率。

| 已观察动作组 | Fit 数量 | Fit target / prediction 均值 | Held 数量 | Held target / prediction 均值 |
|---|---:|---:|---:|---:|
| Reliable degrading | 217471 | -0.623736 / -0.653972 | 61880 | -0.619239 / -0.632505 |
| Reliable improving | 23977 | +0.053015 / -0.129016 | 7611 | +0.056908 / -0.126999 |
| Lost recovering | 8452 | +0.751093 / -0.248333 | 1602 | +0.730808 / -0.258858 |
| Lost other | 74957 | +0.019075 / -0.408052 | 17594 | +0.022706 / -0.429523 |

这些是未加 sequence weight 的动作计数和均值，不是梯度份额；四组也没有穷尽中间质量状态等所有观察。Fit lost-recovering 动作仅占全部 354111 个 observed actions 的 2.39%，reliable-degrading 动作数是它的 25.73 倍。Held 对应为 1.69% 和 38.63 倍。恢复动作目标明显为正而预测均值为负，支持“恢复信号未被充分拟合”的描述，但均值不代表每个动作都预测为负。

## 源码机制与归因边界

**按候选平均确实保留了组内数量失衡。** [e0_context_advantage.py:100](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/e0_context_advantage.py:100) 对 observed 动作回归 `candidate_iou-current_iou`，SmoothL1 的 `beta=0.1`，再按每状态 observed 数量求平均。准确说分母不是硬编码 24；本次 fit/held 的平均 observed 数约为 23.99。一个状态内若只有少量恢复动作，其回归贡献按该比例进入 state loss，没有单独的恢复动作配额。SmoothL1 超出 beta 的分数梯度会饱和，较大的正 target 也不会按 target 幅度无限放大恢复梯度。

这构成数量稀少动作被平均稀释的具体机制，实际动作分布和 fit 错误与之相符；但“除以 24”本身不必然导致恢复失败。如果特征足够可分、模型和优化足以分别拟合各动作，正动作仍可以拟合正确。当前没有按组、按 sequence weight 统计的梯度范数或方向，也没有控制消融，不能把 25.73 倍原始计数解释为 25.73 倍梯度优势，更不能给该机制分配因果百分比。

**Sequence weighting 平衡的是序列，不是状态或动作类别。** [context_e0_train_data.py:17](C:/Users/gb/projects/Dual-RGBT/tools/context_e0_train_data.py:17) 对 eligible set 使用 `w_i=N/(S*n_s)`；全分区 mean 后，每个有 eligible set 的序列等权。Advantage eligibility 为任一 observed，protection eligibility 为任一 reliable-degrading 动作，两项分别归一化，见 [context_e0_train_data.py:77](C:/Users/gb/projects/Dual-RGBT/tools/context_e0_train_data.py:77)。这没有让 reliable/lost 状态或恢复/其他动作等权；同一序列内的稀少恢复动作仍会被 set 数量与候选平均影响。两项 eligible 序列集合也可能不同，现有 audit 没有报告其分组加权质量。

**Zero-KEEP 与 protection 有明确的保守方向，但没有直接惩罚 lost recovery。** Head 最后一层零初始化，KEEP 恒为 0，argmax 初始全 KEEP，见 [e0_context_advantage.py:61](C:/Users/gb/projects/Dual-RGBT/dual_rgbt/e0_context_advantage.py:61) 与 [context_e0_training.py:46](C:/Users/gb/projects/Dual-RGBT/tools/context_e0_training.py:46)。初始对称 tie 选择 KEEP 是已验证控制；最后层可学习，并非死网络。零输出层会使第一步隐藏层权重的任务梯度为零，但不阻止输出层开始学习；记录中参数改变、loss 下降和替换增加也不支持“网络从未训练”。

Protection 对 `current_iou>=0.5 && delta<0` 的 observed 动作施加 softplus，其梯度总推动这些分数下降，且单独最小化时不会停在有限的真实负 delta。它不直接作用于 lost 状态；通过共享 961 参数影响其他状态，是待验证的竞争机制。Latest 固定全分区 audit 中，fit advantage/protection loss 为 0.150554/0.426091，held 为 0.147217/0.429398；loss 数值较大不等于该项梯度必然占主导。不能据此直接取消保护项或断言初始化是主因。

**回归目标与资格目标还有结构差异。** 原损失拟合全部 observed IoU delta；lost 状态中改善但仍低于 0.5 的候选也可有正 target，却不算 recovery。它没有直接保证正确恢复候选超过 KEEP 和所有错误候选，也没有把“可靠状态绝不跌破 0.5”实现为硬约束。后期 loss 持续下降而 recovery 回落，与目标不完全一致相容，不证明单独哪一项应负责。

现有证据不能区分或排除特征区分力、961 参数容量、优化过程、冻结 appearance scorer 的可分性及共享参数冲突；没有证据证明延长 epochs 或换 seed 会通过 gate。Held harmful 12 个中有 11 个原标签 unknown，fit 为 42/56；这也是描述性关联，不能把依赖 TRAIN GT 的 unknown mask 搬进在线决策。当前指标基于固定 protected E0 状态，不是修改动作后传播状态的闭环结果。

## 一个最小 TRAIN-only 修改建议

建议只增加一个 **fit 可恢复 lost 动作的 sequence-balanced SmoothL1 辅助项**，保留现有 advantage loss 与 protection loss。它同时让恢复状态获得独立的归一化份额，并使其正确动作不再在新增项内被其他 23 个候选稀释；不同时改模型、阈值、采样或保护系数。

具体定义：对 fit set `i`、候选 `j`，令 `R_ij = observed_ij && current_iou_i<0.1 && candidate_iou_ij>=0.5`，`n_i=sum_j R_ij`，`eligible_i=(n_i>0)`。复用现有 `sequence_weights(fit_sequence_index, eligible)`，得到固定 fit 权重 `w_rec_i`。设 `ell_ij=SmoothL1(score_ij, delta_ij, beta=0.1)`：

```text
L_rec = mean_over_all_fit_sets(
    w_rec_i * sum_j(R_ij * ell_ij) / max(1, n_i)
)
L_new = L_adv_existing + L_protection_existing + L_rec
```

没有恢复动作的 set 在新增项中贡献 0；只在具备该监督的序列间做 macro 平均。新增项系数固定为 1，不由 held 结果搜索。原始全动作回归与可靠退化保护都保持原系数 1，避免用删除负动作监督来换恢复数。这是目标修改假设，不是已证实修复；它可能增加错误替换、与保护竞争，也不能单独保证正确候选排序或解决候选缺失。

GT 只用于 fit loss 的目标/分组和既有 TRAIN 指标，绝不进入特征、在线 mask 或 action rule。权重只由 fit 计算，不读取 held/TEST 来定系数。保持原始 706/176 split、accepted caches、28 特征、observed 定义、冻结 appearance scorer、961 参数结构、fit normalization、zero-KEEP 初始化、KEEP=0 和 argmax 决策；seed2027、AdamW、LR0.0003、weight decay0.0001、batch512、12 epochs/348 updates 也固定。保持全部资格 gate 和既有 qualifying-only checkpoint 选择规则。

可验证的下一步是先为新 objective 写独立 CPU 检查：空恢复组贡献为零、unobserved 不参与、复制非恢复候选不改变新增项、fit sequence weight 不依赖 held，以及正恢复监督给分数的梯度方向正确。新源与输入、objective 定义和 task/run 经单独审查、真实 update smoke 及准入后，才考虑一次上述固定预算的单因素 trial；本诊断不执行它，也不授权占卡重复 R2。

该 trial 的诊断读数应预先包含 fit 恢复动作的 target/prediction gap、可恢复状态的非正分数与 outranked 分解，以及不变的 held gate。若 fit 恢复拟合改善而 gate 仍失败，只能接受“辅助项改变了拟合”的有限结论；没有 qualifying epoch 就继续 retain E0，不挑拒绝 head、不放松 gate、不延长原配方或追加 seed/threshold grid。Held component TRAIN 已参与开发，且上游 E0 已见过相关 TRAIN 数据，不能把它称为独立最终验证，更不能据此做 TEST 调参。

## 精确证据绑定

下列 SHA 均由本地现存文件 bytes 计算；full plan、config、history、COMPLETE、native 的绑定与 acceptance 一致，gap audit 的 acceptance/dataset/checkpoint 引用也一致。五个分析相关源文件与 full plan/export receipt 的 sealed source map 一致。本次没有重新读取远端 tensors/checkpoint bytes；其内容重放结论来自已完成的 acceptance/audit，不冒充本次独立模型重放。

| 证据 | SHA256 |
|---|---|
| `E0_M2_CONTEXT_FULL_TRAIN_FINDINGS_R2_20261102.md` | `f9ea1ad3d1e9513819d55917629ec13631247b2eadae7bf00746b156441fb7fa` |
| `E0_M2_CONTEXT_FULL_TRAIN_ACCEPTANCE_R2_20261102.json` | `cf596d4a5bbe5bd8dcb09f50403f1177ec5326186c90f904628b89a8a33836cc` |
| `E0_M2_CONTEXT_RECOVERY_GAP_AUDIT_R2_20261102.json` | `3fbc68ec5f93aa4a44c660e4f83b664396ea6b961160419b4275f553e0ce0213` |
| Full plan | `7d899f37f5b3e6fb762238a78203f32660691f52669e4456b14f42c746df9cca` |
| Accepted dataset，引用绑定 | `b79b4a3db2bd7d1c2d639b737a5cd0b7d6b8c10857a59a5a1fbd7af3f2b031f3` |
| Latest / epoch12 checkpoint，引用绑定 | `b49bdd553464a64381a2e36e4060df37e4f3dbb648b23397a7faa7ea285340a5` |
| Full native status | `4647e92f2093920893c31433fdf14fb49e350f868aaca717011bbda22fd253ae` |
| Full history | `0392b97e454a538bb55f67adab110983b35308bed169a201cd4186cb5b7add92` |
| R2 execution source review | `c289d6aed7e3e234c66996dfffb6128fffe3aea7fd524c91f60f67cd8fc2fe2c` |
| `tools/audit_context_e0_recovery_gap_r2.py` | `95a55e3bbee7c081fc155a8d7cfe496b3fcf790cf88ff0230361f569938ce0b2` |
| `dual_rgbt/e0_context_advantage.py` | `8c2c3762d433dd85e359b9a1202630cdcc04d701f6938d1b2d9116847838fbe1` |
| `tools/context_e0_training.py` | `39492b8bd8a866665dc91f74e5b14b32b7eb4eef79bc4e458a5c105691cd06fb` |
| `tools/context_e0_train_data.py` | `341ba73e1af8c958e78c6d47da68d27037d58b972b276800c3e5be6c15247c76` |
| `tools/relative_e0_train_data.py` | `477a8430e2d359817e23bda78b5ff79b53cf03085826b79fd93619eccb0500b7` |
| `tools/relative_e0_training.py` | `6bb3fcb00f603c1c7fe9e83b783061132006fea7d1ecef0040fef52d20182d53` |

只新增本诊断文件；sealed 源码、旧报告和训练终态归档不变。没有 GPU 查询、部署、训练更新、verifier/scorer 重跑、在线 head 提升或 paired/TEST 启动。
