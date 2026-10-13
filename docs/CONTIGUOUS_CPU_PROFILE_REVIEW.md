# 连续 RGB/TIR 数组的 CPU 裁剪实验源码审查

日期：2026-10-13。结论：**PASS（仅源码），阻塞项 0。** 当前脚本可以用于所述限定 CPU 配对实验；本报告没有执行该实验，也没有产生实际一致性或速度结果。

```yaml
review_scope: source-only, isolated CPU crop experiment
review_independence: same-family
acceptance_status: provisional
requested_model: gpt-6-astra
requested_reasoning_effort: max
requested_metadata_source: review delegation
actual_model_id: UNAVAILABLE
actual_reasoning_effort: UNAVAILABLE
actual_metadata_note: no independently attested runtime model or effort identifier available
source_verdict: PASS
source_blocking_issues: 0
cpu_crop_experiment: NOT_RUN_BY_THIS_REVIEW
neural_execution: NOT_RUN
ssh_or_gpu_queries: NOT_RUN
local_pipeline_source_hashes_checked: 11
local_pipeline_source_hashes_equal_saved_manifest: true
remote_running_source_state: NOT_CHECKED
```

这是单独审查任务中的同家族、临时性源码结论。所请求的 Astra/max 不是已核实的实际模型路由。审查者只读取本地源码、既有报告并计算本地文件哈希；唯一写入是本报告。没有导入或运行实验脚本，没有 SSH、模型加载、GPU/训练查询、前向、反向、优化器更新或作业启动。

## 审查依据

主审查文件为 `scripts/profile_contiguous_crops.py`，参照 `dual_rgbt/data.py`、`reports/profile_tracking_default.json` 和 `docs/ISSUES.md` 的 M14。另检查了包初始化文件、清单生成器、已保存的数据审计、流水线入口及其 11 项源码清单。

| 文件 | 本次读取的 SHA256 |
|---|---|
| `scripts/profile_contiguous_crops.py` | `cbe7c7d6a7115d4ee383bec5ddac84b0de8179d4f2f085b628b8523e421e700f` |
| `dual_rgbt/data.py` | `858e41c8e5218c7f5323a6e18cec00997a48d0323101384b71a73d61ea0280cb` |
| `reports/profile_tracking_default.json` | `6356735a99423b03654cecf6ed3929454cd64003a461fb9b635e430fc8ef22f8` |
| `docs/ISSUES.md` | `c4bc4132cffb7bd93bd702c2f337a31fdef825cdf4db01b4ae0bbe3649713981` |
| `reports/full_pipeline_source_manifest.json` | `9984345d9f0a4377ee45cbc8747b83866a30c91fcb6652f2054166214deca50b` |

既有默认线程报告记录 `crop_pair` 73 次调用累计 4.824751 秒，NumPy mean 累计 3.478360 秒，暖机后解码/跟踪均值为 55.112810/109.410392 ms；线程数为 PyTorch 32、OpenCV 64。这支持调查真实 CPU 裁剪开销，但该报告没有把开销因果归于非连续数组。M14 已明确不同进程线程对照没有隔离缓存和系统负载，不能直接称为加速。本次审查没有重新运行旧报告。

## 核对结果

| 检查 | 结论与源码依据 |
|---|---|
| 唯一候选差异 | 新脚本 53 行从同一 `read_pair` 六通道图像复制两张连续三通道数组；候选 33–35 行在这些数组上计算均值和执行 `warpAffine`，原 `data.py:31–34` 在六通道图像的三通道视图上执行同样操作。两路通道顺序均为 RGB 后 TIR。没有换解码器、插值、填充或缩放规则。 |
| 几何与目标 | 候选 25–32、36–39 行与原裁剪器保留相同的中心、面积平方根边长、仿射矩阵、float32 转换、xywh 归一化、MEAN/STD 及 CHW 拷贝顺序。候选返回的 tensor、target、`[left, top, side]` 都有原函数的逐项比较。没有修改 GT 或为负宽高另加处理分支。 |
| RNG | 128 模板无扰动；256 搜索的两路调用均新建 `RandomState(2027 + identity)`，先抽 normal 尺度扰动，再抽两维 uniform 中心扰动。每次暖机与计时重建相同状态，没有先运行旧路而推进候选 RNG 的问题。这里抽取的是固定配对扰动，不声称复现训练数据加载器已推进的完整 RNG 历史。 |
| 闭包 | `old` 和 `candidate` 虽引用循环变量，但所有调用在当前 case 的内层循环结束前完成；函数没有存入 `records` 延后执行。下一帧、尺寸和 seed 赋值前已完成该 case，因此不存在晚绑定导致全部测到最后一帧的缺陷。 |
| 实验样本 | 源码只读 `manifest['dev'][:10]`，每序列取首个和中间有效帧，每帧各测 `(2,128)` 与 `(4,256)`，计划为 40 cases。已保存审计有 97 条 dev；清单生成器要求每条至少 3 个正宽高有效帧，故两次选帧不同。当前本地没有 `data/manifest.json` 或原始图片，本审查未核实实际服务器清单内容或解码成功。 |
| 一致性门槛 | 每个 case 的第一对调用同时暖机并检查 `torch.equal`、两次 `np.array_equal`；检查位于计时和最终报告写入之前，失败会中止正常运行。预计 40 对一致性检查，实际是否全通过仍待运行。后续重复使用同一输入与固定 RNG，未发现输入被原地修改。 |
| 配对计时 | 每 case 先完成一对不计时调用，再做三对计时；`(identity + repeat) % 2` 反转调用顺序。计划产生每路 120 次、共 240 次计时调用，`paired_timed_calls=120` 表示配对数。两路采用同样的 `perf_counter` 边界、同一进程和线程设置。 |
| CPU 边界 | 入口要求 `CUDA_VISIBLE_DEVICES == ''`。新脚本没有 `.cuda()`、CUDA API、模型/Tracker 导入或权重读取；`dual_rgbt.__init__` 只有文档字符串，`data.py` 只定义数据工具。OpenCV 接受 NumPy 图像，`torch.from_numpy` 构造 CPU tensor，`torch.equal` 比较这些 CPU tensor。未发现意外神经网络或 GPU 执行路径。 |
| 活动流水线隔离 | 本地源码搜索只在新脚本中找到候选函数/脚本引用，流水线入口没有调用它。逐项计算既有清单的 11 个本地源文件 SHA256，全部仍与保存值相同；包括 B、C、Tracker、model、data、loss 和完整流水线控制器。没有修改原裁剪器或正在使用的脚本。远端当前状态未由本审查查询。 |

## 测量解释边界

计时包含完整裁剪函数内的边界均值、仿射裁剪、拼接、归一化、目标/几何生成与返回对象处理。256 搜索还包含两路相同的 `RandomState` 构造开销，因此是匹配的裁剪调用计时，不能称为只测 `warpAffine` 内核。图像读取/解码、原六通道拼接以及 `ascontiguousarray` 的准备均在计时外；报告的 `image_decoding_and_separate_array_setup_excluded=true` 如实披露这一点。候选当前通过复制既有六通道图像构造连续数组，尚未实现生产解码流程直接保留两张原始数组。

三次计时在单个 case 中无法各有相同的先后次数，但 40 cases 的总体顺序为旧路先 60 次、候选先 60 次。由于尺寸顺序固定，128 组每 case 旧路先两次、256 组候选先两次；分尺寸结果只能作为有限的微基准描述。报告的总均值/P95 混合两种尺寸、不同图像和三次重复，原始 records 保留可追溯数据；这不是完整延迟分布或统计显著性证明。

源码不会测图片准备、CPU/GPU 传输、preview、定位头、策略、模板更新或在线轨迹，也不计算 PR/NPR/SR。通过本次源码审查只允许将后续成功运行描述为这 40 个真实 dev 裁剪的输出一致性与隔离 CPU 调用计时；不能据此声称完整系统提速、训练数学全域等价、精度保持或三个创新点有效。无需为该无神经网络的 CPU helper 添加 GPU sanity；未来若接入生产路径，应按实际接入差异另行验证。

没有要求修复的源码阻塞项；未增加 fallback、防御分支或任务外重构。实际 CPU 结果仍待执行代理运行后核验。
