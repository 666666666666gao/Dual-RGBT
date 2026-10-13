# 在线 Hann 窗口接入源码审查

- 日期：2026-10-13
- `review_independence: same-family`
- `acceptance_status: provisional`
- `review_context: continuing reviewer; separate local-only source diff`
- `requested_model: gpt-6-astra`
- `requested_reasoning_effort: max`
- `actual_model: unavailable to reviewer (runtime metadata not exposed)`
- `actual_reasoning_effort: unavailable to reviewer (runtime metadata not exposed)`
- `reviewed_base: bdf2fd0eff32ee94acf9a97187682b660ea01b6e`
- `reviewed_target: local uncommitted dual_rgbt/model.py and dual_rgbt/tracker.py`
- `source_verdict: PASS`
- `open_blocking_issues: 0`
- `open_nonblocking_issues: 0`
- `runtime_acceptance: PENDING`

本次只审查上述两个文件的 Hann 窗口差异，对照原 `lib/test/tracker/cadtrack.py`、`lib/test/utils/hann.py`、CENTER head 及现有 A/B/M0 调用。未修改实现，未执行 SSH、GPU、模型、训练或 baseline 精度评价。原 `SPATIAL_EXECUTOR_REVIEW.md` 保持不变。执行代理告知，当前 A 使用已部署的固定源码 `3fdf8ab`，本差异尚未部署或执行；本审查未访问服务器验证运行状态。

## 结论

没有发现本次局部接入的具体源码问题。窗口已经用于实际框解码、基于主响应的候选与收益状态，以及最终输出的置信度和模板更新条件。该 PASS 仅支持 C/D 准备代码的源码一致性，不代表闭环校准或运行验收已经完成。

## 核对依据

1. **使用原窗口定义。** `tracker.py:24` 调用原 `hann2d(..., centered=True)`，与原 tracker 的 `cadtrack.py:34–36` 一致。当前搜索尺寸 256、stride 16，窗口为 `[1,1,16,16]`，可与 `[B,1,16,16]` 响应直接广播。窗口在初始化时转到当前 CUDA 设备，创建与传输包含在已有初始化计时内。

2. **主框确实重新解码。** `model.py:92–103` 先取得原 score/size/offset，再令 `score=score*window`，调用同一个 CENTER `cal_bbox()` 重算 box。与原 `cadtrack.py:102–104` 的顺序相同。返回的 score 和 box 属于同一份加窗响应，不会出现只改变展示分数而沿用原始峰框的情况；没有增加 backbone 或 head 网络前向。

3. **候选和收益头读取加窗后的主状态。** `predict():186–188` 将 window 传给 primary 后，才执行 candidates 和 gain_predictions。PRIMARY 的峰、SECONDARY 的屏蔽邻域及分离峰均来自加窗响应，候选框从对应位置的 size/offset 得到。收益头的最大值、均值、熵、主框和候选位置特征因此与实际主定位一致。PREVIEW 仍按辅助预览图选择其独立粗候选，符合既定动作定义；它的最终框仍由后续加窗定位头决定。EXPAND 查询全部辅助网格。

4. **追加路径只加一次窗口。** `model.py:132–148,192–194` 复用主特征并编码所选辅助区域，取得新的融合响应后乘同一个窗口，再用该响应重算 refined box。它没有把已加窗的主 score 当作融合 score，也没有重复加窗。STOP 返回加窗主框，请求动作返回加窗融合框，两个分支的定位规则一致。

5. **置信度、模板和历史一致。** `tracker.py:35–61` 将窗口传给实际 predict 调用；STOP 的 confidence 取主 score 最大值，请求分支取融合 score 最大值，均对应输出框所用的加窗响应。这与原 `cadtrack.py:113–114` 用加窗 response 的最大值作为模板更新置信度的定义一致。模板按最终映射并裁剪的预测框更新；该 confidence 和最终位移/尺寸变化进入下一帧 history，不会混入窗口前的分数或框。现有更新模态范围保持其按需查询定义。

6. **现有熵计算不会因该窗口产生零值。** 原 `hann.py:6–16` 的 centered 形式在 `1..16` 上采样分母为 17 的余弦窗口，端点也严格为正。CENTER score 本身经 `[1e-4,1−1e-4]` 截断，当前 float32 路径的乘积保持正数。因此本次接入没有给 `probs.log()` 引入零概率，无需另加针对假设性零端点的分支。

7. **A/B/M0 调用保持原始响应语义。** 新参数默认 `window=None`。`forward_action()`、`train_policy.py` 的 primary/correct 调用和现有 M0 predict 调用均未传窗口；它们继续使用原 score 和原解码。差异不增删模型参数或 buffer，不改变 checkpoint 结构。当前 M0 没有覆盖新加窗分支，不能据旧 M0 的通过宣称新分支已经运行验证。

`git diff --check bdf2fd0 -- dual_rgbt/model.py dual_rgbt/tracker.py` 无输出、退出 0。检查仅涉及本地源码，未执行神经网络。

## 后续验收边界

B 的原始响应、零 history 裁剪监督不能直接证明加窗在线状态下的策略有效。C 仍须在训练目录内的真实闭环轨迹上刷新模态质量、候选动作的 signed 收益和历史状态，并在 dev 校准预算与模板阈值。审查任务提交时该流程被明确列为尚未实现或运行；审查结束时工作区出现的后续 C 源码及 train_policy.py 修改不在本次范围内，未获本报告验收。

本次只解决原 M06 中 Hann 接入的源码缺项。最终精度与完整在线延迟仍待相应运行验收；不重新测试用户提供的 baseline 精度，也不据这次源码修改宣称性能改善。
