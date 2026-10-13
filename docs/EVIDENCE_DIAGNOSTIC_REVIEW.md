# 冻结执行器证据诊断源码审查

- 日期：2026-10-13
- `review_independence: same-family`
- `acceptance_status: provisional`
- `review_context: same source reviewer, new diagnostic script`
- `requested_model: gpt-6-astra`
- `requested_reasoning_effort: max`
- `actual_model: unavailable to reviewer (runtime metadata not exposed)`
- `actual_reasoning_effort: unavailable to reviewer (runtime metadata not exposed)`
- `source_verdict: PASS`
- `open_blocking_issues: 0`
- `open_nonblocking_issues: 0`
- `runtime_acceptance: PENDING`

本次仅源码审查 `scripts/diagnose_evidence.py`，对照此前已审查的 `DualRGBT.correct()`、数据与定位损失及 `docs/EXPERIMENT_PLAN.md:40–46`。未修改实验代码，未访问 SSH/GPU、运行诊断、训练或重测 baseline。A 已完成的进程状态来自执行代理通知，本审查不另行查询进程。

## BLOCKING / NONBLOCKING

没有发现需要修复的具体源码问题。脚本适合在完成的 A epoch 2 执行器上进行这次有限范围的 dev 裁剪诊断；源码 PASS 不代表诊断已经运行或已有正面结果。

## 核对结果

1. **权重固定，无优化或新权重。** `diagnose_evidence.py:34–42` 限定物理 GPU 1，以 eval 模式构建模型，读取固定 A `latest.pth`，断言 epoch 2 / step 1000 后严格加载并冻结参数；`:52` 使用 `torch.no_grad()`。脚本没有 optimizer、backward 或 checkpoint 保存。输出中的 `optimizer_updates=1000` 来自所加载 A checkpoint 的历史更新数。

2. **多次 MLP 调用不会污染实际输入缓存。** `:44–47` 的 norm hook 在每次主/辅编码后记录归一化 token；`:63–67` 在实际辅助执行结束后立即克隆 corrector 输入，并取得该次辅助编码的搜索 token。后续 `boxes_from_input()` 只调用 corrector，不调用 backbone.norm。其 pre-hook 确实会覆盖字典中的 `features`，但本轮使用的是已经克隆的局部 `features`，`evidence` 也已保留局部引用。下一动作再次执行 `model.correct()`，会重新更新两个缓存。

3. **重算 summary 的检查对应真实执行。** `:68–71` 从实际 corrector 输入提取主候选 768 维特征，并用所捕获辅助 token 重算 cosine-softmax pooling；`torch.equal` 比较对象是实际进入 MLP 的 `[768:1536]` 切片。该断言能在运行时验证本诊断追踪到了真实 summary，而不是一个另造的替代值。`:26–30` 的框重建取 `[1536:1540]` 候选框，重复原修正器相同的中心 tanh 与尺寸 exp 变换。

4. **ROI 与 GT 坐标一致。** `:73–77` 将原 16×16 搜索网格的索引转换为每个 patch 的归一化左上/右下边界，和当前裁剪下 GT xywh 求交。每 patch 面积为 `1/256`，所以乘 256 后得到该 patch 被 GT 覆盖的比例。`weighted_foreground_mass` 为 pooling 权重对这些比例的加权和；`uniform_foreground_mass` 为同一组 patch 的均匀权重结果。二者比较使用完全相同的区域和 GT，不会把区域面积不同误当成相同的覆盖定义。

5. **eligible 掩码及 oracle 配对正确。** `:90–100` 只对与 GT 有正面积交集的样本执行 GT 面积加权 pooling；features、evidence、target 使用同一个布尔掩码，返回损失通过同序 nonzero 索引写回原样本。未 eligible 样本保留 `None`，其正常路径结果仍记录。`:136–139` 仅在 eligible 行中计算各自 `refined_loss - oracle_foreground_loss`，没有混用全组正常均值。已提供的同 dev 集各组平均覆盖大于零，足以保证各组至少一个 eligible 样本；当前固定任务无需增加空组 fallback。

6. **样本配对及状态范围正确。** `:48–59` 缓存固定 seed 2027 的 100 个 CPU dev 裁剪，与 Stage A epoch 2 校准集一致；每个主模态重置 sample_id，每批四种追加动作共享 templates、search、零 history、preview、primary 和候选。model.eval() 下复用一次主路径与为各动作重复主路径具有相同定义。输出应有 2×4×100=800 行，每主模态/动作 100 行。全部数据来自官方训练目录的 dev 划分；GT 仅用于离线裁剪、损失及明确标记的诊断干预，不被用于正常候选选择或部署策略。

7. **损失分解方向正确。** `:60,80–81,110–112` 依次比较真实 GT 下的主框、所选候选框与修正框：`candidate_switch_gain = L(primary)-L(candidate)`，`correction_gain = L(candidate)-L(refined)`，`total_gain = L(primary)-L(refined)`。前两项相加对应总收益，保留负值，能区分收益来自候选切换还是修正器。

8. **干预范围准确。** `:82–98` 仅替换 MLP 输入中的辅助 pooled summary，主候选特征、候选框和区域坐标保持当前样本值。zero 为零向量；shuffle 为当前四样本批内 summary 循环移动一位；GT pooling 使用已实际编码的同一区域 token。固定 100 样本、batch 4 的当前设置没有单样本尾批。所有干预均复用现有辅助编码，没有重新运行主干或进行参数更新。

9. **输出保留了分析所需证据。** `:101–118` 保存逐样本正常/干预损失、三阶段框 IoU、分解收益、pooling 熵、前景 mass 及框变化；`:128–145` 按主模态/动作汇总，并给出 eligible 样本数与配对 oracle 收益。scope 明确声明冻结 dev 裁剪诊断、oracle 不可部署、zero/shuffle 为表示干预。没有将该输出称为在线 PR/NPR/SR、实际缺失模态实验或延迟提升。

## 结果解释的边界

`normalized_attention_entropy` 指修正器的 **cosine-softmax pooling 权重**，并非 Transformer 自注意力矩阵。GT pooling 是使用真实目标区域的面积权重诊断，不保证给出最优 pooling 或完整在线跟踪上界。前景 mass 衡量权重落到 GT 覆盖 patch 部分的程度；Transformer token 已包含区域内上下文，不能仅凭该值声称特征信息完全来自目标像素。

zero/shuffle 的损失差只能说明固定 MLP 对其输入表示干预的响应；它们没有移除真实辅助编码过程。当前数据仍为 GT 维持搜索裁剪的离线 dev 样本，因此任何正收益都需要后续真实闭环及完整延迟测量验证。LasHeR PR≥78、NPR≥72、SR≥62 与实际系统提速目标仍未由本诊断证明。
