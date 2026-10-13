# 完整 LasHeR 因果评价源码审查

日期：2026-10-13。当前结论：**SOURCE PASS，未关闭源码阻断项 0；运行与正式评价验收仍待完成。** 初读发现的 dev 完整性及计时帧数核对缺失，已在本次审查期间由执行代理修复并重新读取确认。本结论不代表 97 条 dev、245 条 test 或延迟实验已经执行。

```yaml
review_scope: source-only, complete causal LasHeR evaluation
review_context: separate reviewer reading current local sources and supplied toolkit
review_independence: same-family
acceptance_status: provisional
requested_model: gpt-6-astra
requested_reasoning_effort: max
actual_model_id: UNAVAILABLE
actual_reasoning_effort: UNAVAILABLE
actual_metadata_note: runtime does not expose independently attested model or effort identifiers
reviewed_base: 00d3e7732fa249d826ca84f26beabf0d6f058fef
reviewed_target: current uncommitted run_tracking.py and score_tracking.py with supporting sources
source_verdict: PASS
open_source_blocking_issues: 0
runtime_acceptance: PENDING
canonical_server_test_gt_consistency: PENDING
neural_execution: NOT_RUN_BY_THIS_REVIEW
benchmark_or_latency_measurement: NOT_RUN_BY_THIS_REVIEW
baseline_accuracy_evaluation: NOT_RUN
```

采用 `code-review-excellence` 的有证据正确性审查原则。直接读取项目 `AGENTS.md`、`docs/EXPERIMENT_PLAN.md`、`docs/EVALUATION_PROTOCOL.md`、`scripts/{run_tracking,score_tracking,prepare_data}.py`、`dual_rgbt/{tracker,model,data}.py`，以及原 `clip_box`、Hann、CENTER head 定义。原工具包实际读取自 `C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/`，包括 `dataset/basedataset.py`、`dataset/lasher_dataset.py`、`metrics/metrics.py`、`utils.py`、两个包入口及 canonical test 序列清单和 GT 文本。审查者只写本报告；没有执行 SSH、GPU 查询、神经网络前向、训练、指标计算或 baseline 精度评价，也没有修改工具包。

## 因果性与数据边界

`run_tracking.py:75–109` 每条序列仅将 `boxes[0]` 传入一次 `tracker.initialize`，第 1 帧起只向 `tracker.track(image)` 传图像。完整 GT 还用于确定帧数和写离线评分文件；不会传入在线决策。`read_pair` 只读取对应帧的 RGB/TIR 文件，未读取 `boxes`。

`tracker.py:28–64` 的下一帧搜索框来自上一步 `self.state`；当前预测解码并裁剪后写回 state，模板更新围绕当前预测裁剪，history 使用预测位移、尺度变化、置信度、模态和查询间隔。`initialize` 重新设置框、两个模板、history、窗口及计数器，因此不同序列不会继承上一序列状态。`model.predict` 的模态与动作分别由 quality 和 gain 减成本的 argmax 选择，没有 GT、oracle 或未来图像入口。模型 eval、两条在线入口 no_grad，runner 没有 optimizer 或更新参数操作。

`prepare_data.py` 从官方训练目录及固定 train/val 清单建立 882/97 划分，检查 train/dev/test 名称不相交。审查者在本地独立统计这两份清单，得到 882 和 97 个唯一名称，交集为 0；它们与工具包的 245 个 test 名称也无交集。已有 `reports/data_audit.json` 记录相同序列数且 issues 为空，但本地没有 `data/manifest.json`，不能据此声称重新核验了服务器完整图像或 dev 总帧数。

评价脚本没有用 test GT 生成标签、选择参数或选择 checkpoint 的逻辑。`--checkpoint` 和 `--cost-weight` 都是显式输入，所有合并分片必须具有相同 checkpoint SHA256 和成本。**最终权重及成本先由 dev 冻结、再执行一次正式 test** 仍属于执行流程要求；仅有这两个脚本不能证明操作人员从未根据 test 分数重选输入。正式记录应保留冻结决定及其 dev 依据。

## 完整性与分片

当前 `run_tracking.py:56–68` 在分片之前记录完整 split 的序列数与总帧数，使用 `entire_split[shard::shards]` 分配完整序列。默认正式路径遍历每个序列的全部帧；仅显式 `--smoke-frames` 才截断单序列，且生成 `SMOKE.json`。正式评分拒绝任何非零 smoke_frames 并要求 `COMPLETE.json`。

`score_tracking.py:22–37,40–56,66` 已验证：

- 所有分片的 split、checkpoint、cost、分片总数、完整序列数及完整帧数一致；分片编号完整且无重复，序列名称无重叠。
- 合并数量等于记录的完整数量，并强制 dev=97、test=245；test 名称集合还必须与 canonical toolkit 完全相等。
- 每个分片的预测文件集合等于配置名称集合，每条预测和随附 GT 的形状都等于评分 GT 的完整 `(N,4)`；预测有限且宽高为正。
- 每个分片计时数组严格为 `(所有预测帧数,)`，各值有限且为正，`COMPLETE.frames` 必须相等；合并计时帧数还必须等于完整 split_frames。

因此已关闭初读的两个具体问题：97 条 dev 的固定数量未受约束，以及最终 frames/FPS 可以脱离预测帧数。dev 的实际成员来源仍是 `prepare_data.py` 生成的固定清单；审查没有发现当前路径会更换该集合的代码，不要求额外兼容层。

审查者只用 Python 标准库读取本地 canonical 文本，核实 **245 个唯一序列、220,703 行 GT**；所有 GT 行都是有限的四列，每条首框宽高为正。完整 test 预期包含 245 次初始化及 220,458 次非首帧 track。这是本地标注清单的静态核验，尚非服务器图像/标注配对或已完成预测的证据。

## Canonical test GT 必要前提

`score_tracking.py:43–51` 对 test 直接使用原 `LasHeR()`，并检查每条随附 GT 以 float32 读取、`round(0)` 后等于 toolkit GT，长度也必须一致。因此原工具包内部截断预测的行为不能使短缺/多余帧静默通过评分。

**完整服务器 test/init.txt 与 canonical GT 的一致性仍待实际核验。** `EVALUATION_PROTOCOL.md:19–20` 只记录了 `10runone` 的前两框核对。本审查没有服务器数据，不能把当前校验代码或本地 220,703 行统计写成该全量校验已经通过。正式 test 前应取得 245 条的图像配对计数、init 行数、与 canonical 的逐行工具包取整等价结果，并单独核对实际用于初始化的 float32 首框是否与 canonical 原始首框一致。后者不能由“取整后相等”推出。任何实际不一致应先定位数据来源，不应通过截断、跳序列或更换评价 GT 隐藏。

这是数据验收前提，不是已发现某条 init.txt 错误；也不要求为尚未观察到的不一致加入 fallback。当前源码的最终评分断言会拒绝与 canonical 取整后不一致的结果包。

## 原工具包与汇总

评分直接构造原 `BaseRGBTDataet`、`TrackerResult`、`PR_LasHeR`、`NPR`、`SR_LasHeR`，没有本项目指标重实现或 monkey patch。`basedataset.initial_result_file` 对 GT/预测统一 float32 读取及 `round(0)`；`utils` 默认 strict 模式，保留原 ltwh/中心/IoU 的像素边界定义。

原指标先把首帧预测替为首帧 GT，首帧仍在分母内；对非正预测宽高按工具包规则使用前一框；对 GT 任一分量≤0的帧将误差/IoU设为 -1，因而计入 PR/NPR 成功、SR 失败。当前 tracker 的 `clip_box(..., margin=10)` 和评分正宽高断言与该路径兼容，未加入新预测 fallback。

PR 取中心距离≤20的第 20 列；NPR 取宽高归一化中心距离≤0.2的第 20 列，不是 NPR AUC。SR 使用 0 到 1 共 21 个阈值且比较严格大于阈值。评分器拼接各分片的**逐序列曲线**，再对序列平均；PR/NPR 取该列平均，SR 对所有序列及阈值平均。没有按分片等权、按序列帧数加权或用两模态 GT 取最大值。输出统一乘以 100，符合用户百分比表格单位。

原工具包文件本次仅被读取。这里的“沿用原工具包”不等同于已恢复用户 baseline 的历史命令；72.74/69.16/57.84仍按用户提供值引用，没有重新计算 baseline 精度。

## 计时与后续验收

runner 的每帧外层计时从 `read_pair` 之前开始，到 `initialize/track` 返回后结束，包含图像读取/解码、RGB转换与拼接、裁剪归一化、CPU→GPU传输、两路预览、路由、主/辅助编码、定位、预测状态及模板更新。tracker 结束前显式 CUDA 同步，所以没有只记录异步提交时间。每条序列初始化也进入同一 `frame_seconds.npy`，未遗漏或重复加回。内部 `route.seconds` 不含解码；正式均值、P95、FPS使用外层计时。

模型构造及两次权重装载位于序列计时之前。runner 和合并报告都明确 `model_loading_excluded=True`，不应称为含进程启动的冷启动耗时。逐帧计时也不含之后的 CSV/预测/进度文件写入；runner 的 `wall_seconds` 另包含序列循环内这些开销。合并 `fps` 是帧数除以各分片服务时间之和，源码已明确说明；它不是双卡同时工作的墙钟吞吐率。P95来自全部真实逐帧时长合并，不平均分片P95。

尚需真实 smoke/完整运行及完整输出验收来证明执行成功、动作计数和耗时，而不是仅靠源码断言。完整系统相对原去模块 CADTrack 的提速还需要相同设备、相同计时范围的 baseline **延迟** 测量；本次源码及报告不提供该结果，也不触发 baseline 精度重测。建议在 `EVALUATION_PROTOCOL.md` 后续更新中写明新的外层 runner 已实现解码和初始化合计，以免只读旧 Tracker 计时段时误解当前正式口径。

## 静态验证与审查快照

11 份项目/工具包 Python 文件通过 Python 3.13 的 AST 解析，未导入这些模块、未运行指标或网络。末次对 6 份项目 Python 文件重复解析通过，已有 tracked diff 的 `git diff --check` 通过。新文件的运行时导入、CUDA、真实视频执行、完整报告及 checkpoint 的实际冻结来源仍未由本审查证明。

以下为审查时实际字节 SHA256；后续修改应针对变更重新核查。

| 文件 | SHA256 |
|---|---|
| `scripts/run_tracking.py` | `076b40e386f00d1d1047e0c87c07696779259e604081b5cbaaa054257e567a29` |
| `scripts/score_tracking.py` | `22f0cb04de31714f70019f25c8b6b3d48b0384a7aff812d25be367394b3b4ac4` |
| `dual_rgbt/tracker.py` | `406927019ffdcdbc8546755d9f7bff0859aac20766116e4a70f8c89132694fff` |
| `dual_rgbt/model.py` | `434da9f818158c16bd7563e93ef44c65b2b726d881f35f94284fb6e8f9966e50` |
| `dual_rgbt/data.py` | `858e41c8e5218c7f5323a6e18cec00997a48d0323101384b71a73d61ea0280cb` |
| `scripts/prepare_data.py` | `2f2579a0a9a4e397412a93e847d352fa2337e2117ad874deaf3c92a897e9483e` |
| `docs/EVALUATION_PROTOCOL.md` | `3683b5ef47d64ad921f4fb661ad387bee5544fe9f7af3674a2e833fe537555b8` |
| toolkit `dataset/basedataset.py` | `96449e8090c4f1892fd9c8ea6ce34b12bf2b081e9400208870b1eee1d577b9d8` |
| toolkit `metrics/metrics.py` | `9ae628c82201a00002cbef8a62c78746eb26acbd8c226304a4cd78e9223d2382` |
| toolkit `utils.py` | `5a95cbafef8371222e35179da58abf461ed03482695efdaad5ff904a077ba994` |
| toolkit `__init__.py` | `73b199fb056e71ebe556775a73635bf2374be645deb5350c220bfe1bdd5bb0ee` |
| toolkit `dataset/lasher_dataset.py` | `2dddcbdc71e9d7562efe9677fe8fa724b6cbb720cacd6f6289b25b4205daf08a` |
