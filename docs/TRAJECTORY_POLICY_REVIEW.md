# C 真实训练轨迹策略刷新源码审查

日期：2026-10-13。结论：**PASS；源码阻断项 0。** 本结论只覆盖当前 C 训练实现，不代表 C 已执行、预算已校准或正式测试已通过。

```yaml
review_scope: source-only, stage-C preparation
review_context: fresh reviewer reading supplied local artifacts
review_independence: same-family
acceptance_status: provisional
requested_model: gpt-6-astra
requested_reasoning_effort: max
actual_model_id: UNAVAILABLE
actual_reasoning_effort: UNAVAILABLE
actual_metadata_note: runtime does not expose independently attested model or effort identifiers
reviewed_base: bdf2fd0eff32ee94acf9a97187682b660ea01b6e
reviewed_target: current local uncommitted C script and supporting model/tracker/B-label changes
source_verdict: PASS
open_blocking_issues: 0
runtime_acceptance: PENDING
stage_c_neural_execution: NOT_RUN_BY_THIS_REVIEW
benchmark_or_latency_measurement: NOT_RUN
```

直接读取了 `AGENTS.md`、`docs/EXPERIMENT_PLAN.md`、`scripts/train_trajectory_policy.py`、`scripts/train_policy.py`、`dual_rgbt/{model,tracker,data,losses}.py`，并核对清单生成器、CENTER head、Hann 定义、框裁剪函数和现有 B/Hann 审查。后续读取了执行代理新采集的 `reports/trajectory_data_plan.json`。本审查只写本报告；未执行 SSH、GPU 查询、神经网络、训练或 baseline 精度评价，未修改实现。A 的已部署固定源码 `3fdf8ab` 不在修改或重新验收范围。

## 核对结果

1. **GT 与在线状态隔离正确。** `train_trajectory_policy.py:61–80` 每轮每条序列只用第 0 帧 GT 初始化。之后每一帧先调用 `tracker.track(image)`，后续搜索中心、框大小、模板与历史全部来自真实预测。GT 在可标注帧仅变换到已确定的搜索坐标中计算损失，没有替换 tracker state、候选位置或模板裁剪框。非标注帧同样先执行 tracker，不能被误读为每五帧才跟踪一次。

2. **快照确实是决策前状态。** `train_trajectory_policy.py:68–72` 对 NumPy state 使用 `copy()`，对 templates/history 使用 `clone()`，然后才让 tracker 提交当前选择。`tracker.py:47–54` 会原位写第二模板；独立 clone 阻止该写入污染标签。每个标注帧的 search 由同一 image、保存的 previous、相同 factor=4/size=256 再生成；`crop_pair` 此调用没有 RNG，几何与实际跟踪一致。标签和反向前没有 optimizer 更新，模型始终 eval，故不存在因 BN/dropout 或权重变化而换了反事实执行器的问题。

3. **Hann、候选与实际执行一致。** C 的 `labels(..., history, tracker.output_window)` 将同一窗口传给两路 primary；枚举四个 correct 动作时也传同一窗口（`:82–94`）。`model.py:99–101,144–146` 先对各自响应加窗，再由同一 CENTER `cal_bbox` 解码。candidate 和 gain 状态读取加窗后的 primary，STOP 与追加动作都遵循 tracker 的在线约定。`output_window` 在一次 initialize 后不被 track 修改，因此不需要另存窗口副本。原 centered Hann 的 16 个采样点均为正，原 CENTER score 也有正下界；当前 float32 熵输入没有因窗口引入零概率。

4. **监督的符号、次序和维度正确。** 每个 modality 复用一份 primary/candidates，质量目标为 `-main_loss`，四个动作的收益依次为 `main_loss-action_loss`（`:84–97`）。`action-1` 与 PRIMARY/SECONDARY/PREVIEW/EXPAND 顺序一致，学习端 `gain_predictions(... )[:,1:]` 与 `[1,4]` 标签对齐；质量目标是 `[1,2]`。收益允许为负，没有裁为正收益。STOP 的收益与增量成本由模型固定为 0；训练不把成本扣进监督，再在实际决策时重复扣除。

5. **冻结与梯度路径正确。** C 构造新模型并严格载入 policy 的 `net` 后，将 backbone/decode/box_head/preview 全部 `requires_grad_(False)`；AdamW 仅包含 quality/gain（`:43–48`）。标签枚举在 no_grad 内，实际跟踪也在 no_grad 内；学习侧在 no_grad 之外重新运行 `preview_state` 并计算 quality/gain 损失（`:98–112`）。quality/gain 是 Linear/GELU 网络，eval 不关闭其 autograd；preview 无需反向。没有 preview 空间 CE，因此真实轨迹中目标中心越出搜索区不会被错误转换为越界分类索引。冻结参数、BN 状态以及两个头的实际非零更新仍需 C 原生 sanity 证明。

6. **B 的调用契约保持明确。** `train_policy.labels` 新增显式 history 和可选 window；B 在 `train_policy.py:69–70` 显式传零 history，未传 window，继续原有 GT 周围裁剪、无窗监督。C 显式传真实 history 和 Hann。两个调用都与新签名匹配。C 的来源应是已接受空间 A 初始化并训练后的 B policy；源码严格载入所给 checkpoint，但本次没有读取实际 checkpoint 来替代该阶段来源检查。

7. **序列、轮次和训练边界正确。** `train_trajectory_policy.py:38–42` 用独立 `RandomState(seed)` 从 manifest 的 train 一次性选固定 20 条，默认 seed=2027；两轮保持相同序列次序，每次 initialize 后遍历 `range(1,len(boxes))`。不是 GT 抖动小片段。清单生成器只从官方 trainingset 读取 train/dev 标注，并检查 train/dev/test 名称不相交；C 不访问 dev/test 项。每次 optimizer.step 后后续帧使用更新后的策略，这是训练轨迹刷新，不能作为某个固定 checkpoint 的基准轨迹或正式精度结果。

8. **计数与保存对应实际工作。** `frames` 每次 track 后加 1，故排除初始化帧；`step` 每次成功 optimizer.step 和日志写入后加 1。CSV 含轮次、序列、原始帧号、损失、负收益比例、目标中心是否位于搜索区及真实 route。日志 step 从 0 起，checkpoint/COMPLETE 的 step 是完成更新次数。每轮保存 latest 的 net、optimizer、round、step、args；选中序列名保存在 config.json。只有全部循环结束才写 COMPLETE.json。当前脚本没有恢复入口或 best 选择，不应把保存的 optimizer 描述成已实现恢复，也不把末轮权重描述成 dev 最优。

## 实际数据前提与预计计数

执行代理提交的 `reports/trajectory_data_plan.json` 记录了相同 seed=2027、train、20 条所选序列。审查者读取该文件并在本地重算：20 条的 `valid_first` 均为 true，`sum(frames-1)=8854`，与记录的每轮跟踪帧数一致。该采集报告给出的每轮有效 stride-5 标注状态为 1752；本地没有完整 manifest/valid_frames，未独立重算这 1752 个具体索引，也未访问远程数据。

默认两轮预期是 **17,708 次非初始化 track、3,504 次策略更新**，另有 40 次序列初始化。CSV 应有 3,504 个数据行，最后的 step 索引为 3503；最终 checkpoint/COMPLETE 的 step 应为 3504。它们是运行前预期，不是已完成结果。

`ab_moto2north0` 有 45 帧非正宽高 GT，其他所选序列记录为 0。`prepare_data.py` 的 valid_frames 定义就是宽高均正，因此 C 的有效标注过滤有实际数据证据。当前顺序只跳过这些帧的监督，仍逐帧执行 tracker。首帧有效性断言也已由这份实际数据采集报告支持；不需要为未出现的首帧缺失添加替代初始化逻辑。

## 解释边界与后续验收

**收益标签仍是现有裁剪定位代理损失。** `losses.py:9` 将 GT xyxy 截到 `[0,1]`，随后计算 `2×(1−GIoU)+5×L1`。C 复用这一 A/B 定义；它没有在完整图像坐标中对最终 `clip_box` 结果计算 OPE 损失。真实轨迹越界时，这一标签衡量裁剪后目标的定位代理，不能称为完整 GT 框或 PR/NPR/SR 的真实收益。`target_centre_inside` 如实记录中心越界情况；中心在内也不意味着整个 GT 框在内。该边界不破坏目前声明的代理训练方案，不据此修改 A/B 或增加越界分支。

此脚本完成的是 C 的训练轨迹刷新部分；dev 的预算/阈值校准、最终权重冻结、完整测试及系统延迟验收仍是后续工作。默认 `.05` 仍是 token 代理成本权重，不是已经校准的延迟预算。当前 CSV 只记录发生监督的帧，不能据这些行宣称全轨迹的动作比例。

C 正式运行前仍需用真实所选 train 序列执行到至少一个有效监督帧，完成实际反向与 AdamW 更新；核对 quality/gain 有有限非零梯度和参数变化，执行器与 preview 参数/BN 状态不变，保存后严格重载并对同一保存状态核对输出。A 的 M0、B 的准备审查或本报告均不能替代这条新 C 路径的原生验收。

本地六份 Python 源码经 AST 解析通过；相关已跟踪差异的 `git diff --check` 通过。最初使用 PATH 中的 `python.exe` 进行解析时因 `No pyvenv.cfg file` 失败，之后使用 `uv python find` 返回的现有 Python 3.13 解释器完成同一静态解析；没有安装依赖、修改环境或执行网络前向。实际部署环境的导入、CUDA 执行与重载仍待原生检查。
