# B 策略 M0 验证源码审查

日期：2026-10-13。结论：**PASS；源码阻断项 0。** 这是两次 B 更新及 checkpoint/在线检查器的准备审查，尚无本报告核验的 B 原生执行结果。

```yaml
review_scope: source-only, B M0 additions and checkpoint checker
review_context: continuing independent reviewer, separate narrow source task
review_independence: same-family
acceptance_status: provisional
requested_model: gpt-6-astra
requested_reasoning_effort: max
actual_model_id: UNAVAILABLE
actual_reasoning_effort: UNAVAILABLE
actual_metadata_note: runtime does not expose independently attested model or effort identifiers
source_verdict: PASS
open_blocking_issues: 0
runtime_acceptance: PENDING
native_B_M0_execution: NOT_RUN_BY_THIS_REVIEW
```

直接读取当前 `scripts/train_policy.py` 和新 `scripts/check_policy_checkpoint.py`，检查其差异，并对照已读取的 model/tracker/data 实现及原 `scripts/sanity.py` 的重载检查。只写本报告；没有 SSH、GPU 查询、神经网络运行或实现修改。已接受的 `TRAJECTORY_POLICY_REVIEW.md` 保持不变。

待运行的 B M0 是从空间 A 的 `runs/stage_a_spatial_20261013/best.pth` 初始化，以 seed=2027、epochs=1、samples=8、batch-size=4、workers=2 完成两次 AdamW 更新，然后运行检查器。以下均是源码所能支持的预期，不是已完成结果。

## 正确的检查路径

1. **各策略模块实际参与反向。** `train_policy.py:101–107` 先清零梯度、执行 backward，再分别要求 preview、quality、gain 至少一个参数具有非零梯度。随后对全部可训练参数裁剪梯度，并断言返回的总范数有限，再执行 optimizer.step。非零梯度断言本身并不排除 Inf，但后续有限范数断言覆盖这一点；不需要再增加逐参数重复检查。要求每个模块有非零梯度合理；B 的 history 恒为零，不能据此要求每个输入列都有非零梯度。

2. **更新次数不会先于更新递增。** `train_policy.py:107–119` 在 optimizer.step 和当前 CSV 行写入之后递增 step，epoch 末将 step 随 net/config 保存。默认计划的 DataLoader 有 8/4=2 个完整批次，故完成后 checkpoint step 与 COMPLETE.json steps 应均为 2，CSV 有两行且 step 索引为 0、1。检查器要求保存的 step>0（`:30`），并如实把具体数值写入报告；采集时仍应核对它是否为本次计划的 2。

3. **冻结比较覆盖参数与 BN buffer。** `check_policy_checkpoint.py:27–34` 将 B checkpoint 的所有 backbone/decode/box_head state_dict 项与所给 A executor 逐 tensor `torch.equal` 比较，包括这些模块的 buffer。随后 preview/quality/gain 各要求至少一个 state tensor 与 A 不同。前者验证最终冻结状态，后者结合训练中的非零梯度断言验证各策略模块发生更新。模型的 strict load 在 `:36,38` 检查完整 key/shape，不会把缺失模块当作可接受的部分载入。

4. **确实执行 predict 比较。** 两个新模型都严格载入保存的 policy，再在相同真实 train 图像产生的 templates/search、零 history、cost_weight=.05 上调用实际 `predict`（`:39–50`）。比较包括 modality、action、最终 box 和五项 gains 的逐位相等。此处未传 window，因此是 B 的无窗 predict；后续 Tracker 检查才覆盖 Hann 在线路径。重载结论的精确范围见下文。

5. **闭环过程没有注入后续 GT。** `:51–66` 仅使用首帧 GT 初始化，随后按帧 1–32 读取两路图像并调用实际 Tracker。Tracker 从预测状态生成后续搜索裁剪，并按其预测 confidence 更新模板与 history；检查器没有用后续标注重置状态。它只访问 manifest 的 train[0]，没有使用 dev/test。

6. **token 检查观察真实执行。** `:54–63` 将 hook 装在 backbone 第一个 block 的 attention 前；每帧清空 trace，执行 track 后按实际 route 断言：STOP 为 `[385]`，局部动作 `[385,193]`，EXPAND 为 `[385,385]`。这些长度含两个模板的 128 tokens 和 CLS，而非仅搜索 tokens。hook 会捕获意外多跑的完整主/辅路径。它检验的是本次 32 帧实际出现的动作，不保证四种查询动作和 STOP 都出现，也不是重新执行原 M0 的所有组合。

7. **框/history 断言和输出范围准确。** `:64` 要求实际预测框所有坐标有限且宽高至少为 10，与 Tracker 使用的 clip_box margin=10 一致；最终 `:65` 明确检查 history 的全部元素有限。history 是包含方向/尺度变化的有符号状态，不要求元素均为正。每帧报告保留预测框、实际 heavy token trace 和 route。只有 32 帧检查全部完成后才写 status=PASS。报告明确说明不作为精度或速度证据；hook 和检查器本身也不构成完整计时实验。

## 验收口径与已关闭的两处澄清

**加载一致性是两个文件加载副本之间的一致性。** `:35–50` 的 model 与 clone 都先从同一保存文件加载权重；它没有拿训练过程中保存前的 model 输出作参考。原 `scripts/sanity.py` 的 `strict_reload_bit_equal` 比较保存前输出与文件重载后的输出。执行代理已将新检查器的字段改为 `strict_load_instances_bit_equal`（`:70`），审查者重新读取了实际源码；当前命名准确表示“两次严格载入后的 predict 逐位一致”。严格加载和实际 predict 构成有效的 checkpoint 可用性检查，本次验收不宣称保存前后输出逐位一致。

**history 的有限性检查已补齐。** 初读版本的 `tracker.history.abs().sum()>0` 只检查非零，模态 one-hot 本身已保证正常 history 非零，而且 Inf 会通过。执行代理已将它直接替换为 `torch.isfinite(tracker.history).all()`（`:65`），审查者重新读取了实际源码。最终检查明确拒绝任一 NaN/Inf，没有添加 fallback，也没有声称运动/尺寸字段必须非零。此次修正是补齐声明的验收条件，并非声称曾观察到实际非有限 history。

## 数据与运行验收边界

检查器固定访问 train[0] 并执行到索引 32。源码的数据清单生成器只保证每序列至少 3 个有效 GT，不能单凭这一规则推导存在 33 帧。执行代理随后报告了实际远程 CPU 预检：train[0] 为 `2up`，共 176 帧，首帧有效，因此此次固定 32 帧范围有实际数据支持。本审查未独立 SSH 复查，也没有把这项 CPU 预检写成神经网络运行证据。

两个新审查 Python 文件的本地 AST 解析通过；上述两处修正后的 checker 再次解析通过；`train_policy.py` 的 `git diff --check` 通过。B 的真实有限损失/梯度、两次更新、执行器完全不变、各策略模块变化、两次严格加载的实际 predict 比较以及 32 帧 hook/框/history 检查，仍必须由后续实际运行产物确认。该 B M0 通过也不能替代 C 的真实历史策略训练更新、dev 校准或正式测试。
