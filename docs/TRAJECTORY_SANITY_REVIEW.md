# C 原生 sanity 增量源码审查

日期：2026-10-13。结论：**PASS；新增 `--sanity` 的源码阻断项 0。** 这是 same-family/provisional 的源码结论，实际 C sanity、正式 C 训练及精度/延迟验收均未由本审查执行或确认。

```yaml
review_scope: source-only, new Stage-C --sanity branch
review_context: delegated reviewer with parent-supplied brief and local artifacts
review_independence: same-family
acceptance_status: provisional
requested_model: gpt-6-astra
requested_reasoning_effort: max
actual_model_id: UNAVAILABLE
actual_reasoning_effort: UNAVAILABLE
actual_metadata_note: no independently attested runtime model or effort identifier exposed
reviewed_head: 00d3e7732fa249d826ca84f26beabf0d6f058fef
reviewed_target: current uncommitted scripts/train_trajectory_policy.py --sanity additions
source_verdict: PASS
open_blocking_issues: 0
runtime_acceptance: PENDING
ssh_gpu_model_training_execution: NOT_RUN
benchmark_or_latency_measurement: NOT_RUN
```

本次使用 `code-review-excellence` 的具体正确性审查方式，直接核对当前 diff、`AGENTS.md`、`docs/TRAJECTORY_POLICY_REVIEW.md`、`docs/REFRESHED_POLICY_RUN_REVIEW.md`、实验计划及 `dual_rgbt/{model,tracker,losses,data}.py`、`scripts/train_policy.py`。只写本文件；未连接服务器、查询 GPU、载入模型或 checkpoint tensor、执行前向/反向/优化器或改动训练源码。请求的 Astra/max 与不可独立证明的实际运行标识分开记录，不声称取得外部模型审查证明。

## 核对结果

1. **两次更新来自同一条真实 C 路径。** 新参数位于 `train_trajectory_policy.py:36–37`；没有另造样本、损失或训练循环。`:69–84` 从所选 train 序列读取真实图像，只以首帧 GT 初始化，之后每帧先实际 `track`；保存的 `previous`、`templates.clone()` 和 `history.clone()` 是跟踪前状态。标注帧再用相同搜索几何、历史和 Hann 枚举两模态四动作，沿原有质量/收益监督执行 backward 与 AdamW。第 2 次成功更新且 CSV 行写出后才进入检查并返回（`:120–132,156`）；不是两次空循环或仅做推理。

2. **两头的非零梯度、有限性与参数变化有对应断言。** `:114–117` 在每次 backward 后分别要求 quality/gain 至少一个参数有非零梯度；`:111,118–120` 要求 loss 和覆盖这两个头的裁剪前总梯度范数有限，然后才执行 step。`:136–138` 将第 2 步后的每个头与初始 CPU 独立副本比较，要求各至少一个参数变化。当前 quality/gain 都是 Linear/GELU，无该前缀下的 buffer，故这里确实是在验证参数变化。它证明的是每个头参与两次反向及两步后的净变化，不声称每个标量参数都变化。

3. **冻结检查包含 preview 与持久 buffer。** `:45–52` 严格载入 B，令模型 eval，冻结 backbone/decode/box_head/preview；AdamW 只包含 quality/gain。初始快照使用 `detach().cpu().clone()`，不会被后续原位更新同步污染。`:133–135` 对全部非 quality/gain 的 state_dict 项逐一 `torch.equal`，覆盖执行器、preview 参数、BN 持久 buffer 和 `token_cost`。`Tracker` 及 `predict` 的 no_grad 路径与 eval 状态吻合。这里的冻结对象是模型持久状态；Tracker 的框、历史和模板按真实轨迹正常推进。

4. **保存后严格重载与同输入输出比较正确。** `:139–149` 使用第 2 次更新后的模型，在保存的跟踪前 templates/search/history 及同一 Hann、cost_weight 上求 expected；随后实际保存 `latest.pth`，构造新模型并从该文件 `strict=True` 重载，再比较 modality、action、box 与 gains。两个 `predict` 都是 no_grad，两个模型均为 eval；输入没有经过新的 `track` 或模板更新。该检查比较的是保存后模型在同一状态上的输出，不要求它等于 CSV 中更新前的原始 route。`DualRGBT` 构造器仍会读取保留的原 CADTrack 文件，属于已有明确启动依赖。

5. **短检查的输出与正式完成标记分开。** 达到两步时先完成冻结/变化/重载断言，再写 `SANITY.json(status=PASS)` 并返回；因此这条默认 sanity 路径不会写正式 `COMPLETE.json`，也不会进入正式的每轮保存分支。`return` 退出 CSV 的 with 块时会关闭并刷新第二行。目录仍以 `exist_ok=False` 新建，不能覆盖已有 run。sanity 的 checkpoint 仅保存 net/step/config，不是优化器续训验收材料。

6. **不带 `--sanity` 时，正式 C 的行为与原审查一致。** diff 只有参数、受该参数控制的快照/断言/提前返回；默认参数仍为 seed=2027、20 条 train、2 轮、label_stride=5。监督、状态推进、正式 optimizer 和 COMPLETE 写入没有改变。既有“裁剪后的 GIoU/L1 代理收益，不等于完整图像 PR/NPR/SR 收益”的边界仍适用；本增量不扩展原方法结论。

## 输出证明范围与正式运行前提

**无需要先修复的源码阻断项。** `SANITY.json` 的 `saved_state_reload_bit_equal=true` 对应的实际断言范围是：严格加载成功，以及上述同一输入的 modality/action/box/gains 一致。源码没有把重载后的全 state_dict 再逐项与保存前比较，也没有直接比较 quality 数值、所有响应图或多帧重放；不得把该字段扩写成这些未检查事项。此前文档要求“严格重载并对同一保存状态核对输出”，当前分支满足该范围，不据此增加无证据的防御性分支。

正式 C 必须**不带 `--sanity`**，并从**已完成的、由 A7 初始化的 fresh B** 原始 checkpoint 重新开始，使用新的 C run 和新 AdamW。不能以旧 A2/B 诊断权重、C sanity 的两步权重或未完成 B 的中间权重代替。当前 `--policy` 只执行严格 tensor 加载，不自行验证阶段来源；执行代理须以 fresh B 的实际 config、完成记录及 checkpoint 身份核对。新 B 审查给出的五轮预期为 epoch=5、step=6250，本审查没有核验其实际完成或远端权重身份。

真实 C sanity 仍须实际执行并收回 PASS、两行 CSV、checkpoint 与配置；本报告不能替代它。依据已有数据计划，默认首条 `11runthree` 有343帧、无无效宽高，预期两个监督点为 frame=5、10，即2次更新、10次非初始化 track。这里是从已收回报告推导的预期，没有读取真实图像或远程 manifest。

正式默认两轮预期仍为 **17,708 次非初始化 track、3,504 次更新、40 次初始化**；CSV 应有3504个数据行，最后 step 索引为3503，checkpoint/COMPLETE 的完成步数应为3504。审查者从本地 `reports/trajectory_data_plan.json` 的20个长度重算了每轮8854次 track；每轮1752个有效 stride-5 监督点沿用该采集报告，本地 `data/manifest.json` 不存在，未独立重建有效帧索引。这些都不是已运行结果。完整 dev 校准、test 评价和在线延迟测量仍属后续工作。

## 静态检查与证据固定

只用现有 Python 3.13 的 `ast`/`hashlib`/`json` 标准库读取源码和已收回报告，没有导入项目、Torch 或运行模型。下列六份源码 AST 解析均通过；独立执行 `git diff --check -- scripts/train_trajectory_policy.py` 返回0。

| 文件 | 审查时 SHA-256 |
|---|---|
| `scripts/train_trajectory_policy.py` | `d45ba30d89c4400f82d6900cd23155f23568c8b03075be5855e898b00b462740` |
| `scripts/train_policy.py` | `2fa070434680704c984d9a1b5b992973103971c47d692a6fcbea734571922913` |
| `dual_rgbt/model.py` | `434da9f818158c16bd7563e93ef44c65b2b726d881f35f94284fb6e8f9966e50` |
| `dual_rgbt/tracker.py` | `406927019ffdcdbc8546755d9f7bff0859aac20766116e4a70f8c89132694fff` |
| `dual_rgbt/losses.py` | `d3b941610e666a236036fd67079af15d5003e366a56adf8acab8b34d8ac584b1` |
| `dual_rgbt/data.py` | `858e41c8e5218c7f5323a6e18cec00997a48d0323101384b71a73d61ea0280cb` |
