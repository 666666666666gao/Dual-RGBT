# 空间执行器源码审查

- 日期：2026-10-13
- `review_independence: same-family`
- `acceptance_status: provisional`
- `review_context: fresh agent; direct current-source review`
- `requested_model: gpt-6-astra`
- `requested_reasoning_effort: max`
- `actual_model: unavailable to reviewer (runtime metadata not exposed)`
- `actual_reasoning_effort: unavailable to reviewer (runtime metadata not exposed)`
- `reviewed_base: 41534b8107e18416e6122db861ece10fbfe5f291`
- `reviewed_target: current uncommitted worktree, including scripts/run_stage_a_spatial.sh`
- `source_verdict: PASS`
- `open_blocking_issues: 0`
- `open_nonblocking_issues: 0`
- `runtime_acceptance: PENDING`

直接读取当前 `AGENTS.md`、`docs/EXPERIMENT_PLAN.md`、`docs/ISSUES.md` 的 M04/M08，以及 `dual_rgbt/{model,losses,tracker}.py`、`scripts/{train_executor,sanity,train_policy}.py` 和新启动脚本，并对照上述基准提交的完整差异。进一步核对了数据采样、原 CADTrack 主干、patch embedding、decode/CENTER head、原训练损失及数据清单生成源码。只写本报告；未执行 SSH、GPU 前向、反向、训练、基线重测或测试集评价。

## 结论

没有发现当前空间执行器 M0 与短程 A 范围内需要修复的具体源码问题。实现对应已有 M04/M08 证据：取消候选框直接切换和摘要 MLP，保留局部辅助空间证据，让原定位头重新选出最终位置。可以进入真实数据 M0；短程 A 仍以新 M0 实际通过为前提。

## 核对依据

1. **辅助选择发生在投影之前，空间索引一致。** `model.py:68–90,119–142` 先对原搜索像素 unfold，再按原 16×16 网格索引 gather，之后才执行共享 patch 投影和 12 层 ViT。当前 patch embedding 是 16×16、stride 16、norm 为 Identity，因此展开权重后的线性投影与原卷积的所选位置一致。位置编码也按相同索引提取。局部索引是无重复的 8×8 区域，扩大动作覆盖全部 256 个位置；两个模板共 128 token，故主路径/扩大辅助各为 385，局部辅助为 193。

2. **空间回填与 RGB/TIR 顺序正确。** `model.py:136–140` 将 `[B,K,768]` 辅助搜索 token scatter 到 `[B,256,768]` 的原位置，未查询处为零。RGB 主路使用 `[primary RGB, sparse TIR]`，TIR 主路使用 `[sparse RGB, primary TIR]`，与原 `cadtrack.py:54–59` 的 `[RGB,TIR]` 顺序一致。转置后 reshape 成 `[B,1536,16,16]` 保持每个通道的行列位置。完整主特征被直接复用，没有再次运行主干。

3. **最终定位不受候选框强制。** `correct()` 中 candidate 只用于生成辅助区域，保留的 box 形参不参与计算。输出来自共享 decode 和 CENTER head；`head.py:129–155` 在新的完整响应图上取最大响应位置，再读取该位置的尺寸/偏移。因此 PRIMARY、SECONDARY、PREVIEW 是查询位置动作，EXPAND 是全辅助查询，均不会把对应候选框直接当作最终框。现有调用签名保持一致，无需兼容层。

4. **损失与梯度路径正确。** `losses.py:7–32` 的每个定位目标是 `2×(1−GIoU)+5×L1`，再加原 GT 热图 focal loss；请求路径的最终框和最终 score 使用与主路径相同的目标。STOP 只使用主路径目标，请求动作累加主/辅融合两项目标。回填没有 detach 或 no_grad，scatter 的 source 接收梯度，融合目标可以经辅助 ViT、所选 patch 投影回到辅助原始像素；同时也回传到复用的主特征。CENTER head 的离散峰位置由 focal 监督，尺寸/偏移由框损失监督，与原头的训练方式一致。未引入 GT 候选选择或 IoU 伪标签。

5. **新 A 从原权重开始，预算与动作均衡符合计划。** `model.py:34–43` 以 `training=False` 构建原 CADTrack，再严格加载用户原文件的 `net`，不会走 YAML 的训练预加载分支。`train_executor.py:28–48` 默认 seed 2027、原 checkpoint 路径，冻结 preview/quality/gain，仅优化 backbone、decode 和 box_head；主干学习率为其他执行器参数的 0.1。`:61–80` 保留训练模式、固定 BN 与梯度裁剪 0.1。新 launcher 使用物理 GPU 0、项目解释器、独立 run 名，参数为 2×2000 样本、batch 4，即 1000 次更新。`:71` 的 `(step % 2, (step // 2) % 5)` 每十步覆盖两模态×五动作，各组合总计 100 次；不会接着旧 A 或 M0 的更新权重训练。

6. **best 保存和 dev 配对正确。** `train_executor.py:92–157` 每个 epoch 使用相同 seed 的 100 个 dev 样本。`data.py:51–63` 按 `seed+index` 独立生成采样，dev DataLoader 不 shuffle，十路径具有相同样本顺序。best 标准是十个等样本数路径的平均定位损失，与计划一致；严格下降时同时保存该轮 `net/epoch/step/config` 与 `best.json`，latest 另含 optimizer。best 可直接供当前 `train_policy.py:49–50` 的严格 `net` 加载使用。定位损失降低、signed gain、IoU0.5 纠正/损害与查询覆盖仍来自真实 GT，并保留逐样本记录；没有把训练总损失误作 best 标准。

7. **M0 检查针对实际空间路径。** `sanity.py:31–61` 对两模态×五动作检查十二层主/辅 token trace、有限损失/参数梯度、末层主干非零梯度和实际 AdamW 参数变化。新增辅助像素梯度检查先清除 `search.grad`，且执行器损失没有通过 preview/gain 连续反传的辅助捷径：preview 只通过离散候选索引影响查询。因此该断言检查的是所选辅助像素经过编码与融合定位损失的真实梯度。`:64–81` 保存新 M0 文件、严格重载并比较相同输入的框逐位一致，同时强成本调用实际 predict 路径并验证 STOP 只有一次 385-token 主编码。这些是待运行断言，本审查不声称已经通过。

8. **调用处与数据边界一致。** 当前 `train_policy.py:51–83` 冻结空间执行器的三个模块，并从新的 `correct()['box']` 重新计算训练样本的 signed 收益；不再引用已移除的 corrector。`tracker.py:42–43` 追加动作读取融合 score，STOP 读取主 score，不再访问被移除的 quality_logit。活跃调用处未发现残留该字段。`diagnose_evidence.py` 已明确限定为 41534b8 的历史池化诊断，不要求其适配新架构。训练/M0/B 只读取 train，best 只读取 dev；清单生成从官方 trainingset 读取这些标注，test 仅取序列名做分离检查，不参与训练、收益标签或选权重。

`git diff --check 41534b8` 无输出、退出 0；新 shell 脚本为 LF、无 UTF-8 BOM、末尾有换行。未执行模型或训练脚本。

## 验收范围

本次 PASS 仅支持空间执行器的源码正确性，不代表辅助净收益、充分训练、在线精度或延迟已经改善。新旧 A 同 seed、样本与更新预算的比较同时改变了空间表示、输出规则和融合路径损失，按计划只能作为系统级比较，不能归因成三项独立消融。

原有 M06 的 Hann 解码缺项和模板阈值校准、M07 的真实延迟成本、B/C 的策略学习与真实闭环仍按原计划处理；它们不阻断本次 M0/A，也未因本报告被宣告解决。正式 PR/NPR/SR 与完整系统平均/P95 延迟须在 C 完成、最终权重和预算冻结后验证；本次不启动 D，也不重测用户 baseline。
