# 固定权重因果前缀延迟对照源码审查

2026-10-13。结论：**SOURCE PASS，未关闭源码阻断项 0；native inference/latency NOT RUN。** 当前脚本可作为后续固定 checkpoint、固定成本的独立 dev 前缀诊断入口。此结论不等于已部署、已通过服务器原生运行，也不证明最终速度、精度或任何模块的有效性。

```yaml
review_scope: source-only
review_context: separate reviewer directly reading local source artifacts
review_independence: same-family
acceptance_status: provisional
policy_requested_model: gpt-6-astra
policy_requested_reasoning_effort: max
actual_model_id: UNATTESTED
actual_reasoning_effort: UNATTESTED
actual_routing_note: actual model and effort routing were not independently proved
source_verdict: PASS
open_source_blocking_issues: 0
native_model_import_or_checkpoint_load: NOT_RUN
neural_forward_backward_optimizer: NOT_RUN
latency_measurement: NOT_RUN
baseline_accuracy_evaluation: NOT_RUN
ssh_gpu_query_or_deployment: NOT_RUN
active_pipeline_source_hashes: 11/11 MATCH
```

本次审查遵循项目 `AGENTS.md` 和 `C:/Users/gb/.codex/skills/shared-references/local-codex-policy.md`。只写本报告；没有修改源码、权重或活动管线，没有运行脚本入口、SSH、GPU 查询、模型前向或训练。读取到的 Git HEAD 为 `0cd2a148751cfee86b5d0abc20f7c7669b8d0066`；目标脚本仍是新增未提交文件。

审查目标 `scripts/compare_tracking_latency.py` 的 SHA256：
`4885422a5865942474fea78e5e007a05d29d6fe535b096e58a97ef71e03711e2`。

## 真实接口、模型与严格装载

脚本第 59–69 行直接建立原 `CADTrack(params)`，没有用另一个简化 forward 冒充 baseline。显式参数与原 `lib/test/parameter/cadtrack.py:17–31` 一致：template factor/size 为 2/128，search factor/size 为 4/256，`save_all_boxes=False`；另设 `debug=0`。实际 YAML 指定两个模板、16 像素 patch、CENTER head、每 5 帧检查模板更新、阈值 0.65；六通道 mean/std 与 Dual-RGBT 的常量一致。

`lib/test/tracker/cadtrack.py:23–38` 以 `training=False` 构造网络，对原 `/home/gaob/CADTrack_ep0014.pth.tar` 的 `net` 执行 `strict=True`，随后 `.cuda().eval()`。`build_cadtrack(..., training=False)` 不进入训练期的预训练路径或 `strict=False` 分支，YAML 中旧的训练预训练路径不参与本次加载。原 tracker 所需的 cfg、checkpoint、四个裁剪参数、debug 和 save_all_boxes 均已提供；未发现缺失参数或返回值接口错误。

实际 `vit_care.py:57–63,98–146` 对 RGB/TIR 分别运行同一组 12 层 Block；本地去模块实现没有执行 MFI/CAM/DAM。原 model 从两路最后 256 个 search token 拼接，经 `decode_fuse_search` 和 CENTER head 得到响应；`track_query_before` 明确输出 `None`，因此这个 baseline 不是带有效跨帧 query 的完整论文版本。不能由仍存在的模块文件或 YAML `CROSS_LOC` 推断对应模块已执行。

`DualRGBT()` 自身先按相同 YAML 构造去模块 CADTrack，并严格装载原 checkpoint，再保留 backbone/head/decode 和新增 preview/quality/gain；比较脚本第 71–73 行随后对指定 Dual-RGBT checkpoint 的整个 `net` 再做 `strict=True`。两侧都是 eval、no_grad、requires_grad=False，无 optimizer。严格装载是已核实的代码要求，**实际 checkpoint 键、形状和可加载性未在本审查执行验证**。

两个 checkpoint 在模型建立前计算 SHA256，全部序列结束后再核验相同 SHA256；脚本没有保存或更新 checkpoint。运行必须传入已固定的 checkpoint 路径和成本；本脚本既不搜索权重/成本，也不证明外部输入已经按 dev 决策冻结。`DualRGBT` 再次更新同一份 YAML 配置，不改变本次 baseline 已采用的配置值。

## 因果时序、GT、解码和各自状态

第 54–58 行只取 `manifest['dev']` 的固定前缀，要求所选序列的 RGB/TIR/boxes 长度相同且足够、首帧有效；没有读取 test split。每条序列只将 `boxes[0]` 用于初始化，第 1 帧起顺序处理 `range(args.frames)`，两侧 `track` 都只收当前图像。其他 GT 只参与长度检查，不参与选模态、选动作、重定位、模板更新、计时筛选或准确率评分。

`read_pair` 实际执行两次 `cv2.imread`、两次 BGR→RGB 和六通道拼接；每个 arm 都在自己的计时区间内调用一次，没有将首个 arm 的已解码数组或 crop 复用于另一个 arm。它仅访问当前帧的 RGB/TIR 路径。两侧共享原始帧与首框，后续 crop 来自各自预测，因此这是完整因果轨迹的前缀比较，不是相同 GT crop 的离线 forward 比较。

原 tracker 在 initialize 重置 state、frame_id、template list 和 track_query_before；后续从自己的 state 裁剪，Hann 响应解码并 `clip_box(..., margin=10)` 后写回。其预测置信度超过 0.65 且帧号整除 5 时，加入完整六通道模板，保留初始模板并替换动态模板。

Dual-RGBT 在 initialize 重置 state、两模板、八维 history、Hann 窗口、frame 和 since_query。后续裁剪、框映射、模板和历史都只依赖自身预测。每 5 帧且置信度大于等于 0.65 时更新选中主模态的动态模板；只有执行辅助动作才更新另一模态。主模态由 quality argmax 决定，动作由 gain−cost×token_cost 的 argmax 决定，STOP 不执行辅助编码。没有把原 baseline 的预测或模板喂给 ours，也没有在第 9 帧预热分界重置状态。

本地没有 `data/manifest.json` 或服务器图像，故未重新核验实际所选序列名称、长度、首框、图像可读性；已有 `reports/data_audit.json` 的 882/97/245 计数只是已存记录。断言会在不满足默认前缀长度时停止，并不会静默跳帧或换序列。这里没有证据要求新增 fallback。

## 计时与统计

第 80–105 行按 `(sequence_index + frame) % 2` 交替先后顺序；两个模型处于同一进程、单个可见 CUDA 设备，并串行运行。`CUDA_VISIBLE_DEVICES` 只接受 `0` 或 `1`，两侧没有另选设备的代码。每个 arm 在启动计时前同步，随后用 `perf_counter` 包住自己的解码、initialize/track 和结果提取，并在结束计时前再次同步。因此测量覆盖实际 CUDA 完成时间，而不是仅异步提交时间。

计时包含读取/解码、颜色转换、裁剪归一化、CPU/GPU 传输、预览、路由、主/辅助编码、检测头、CPU 框映射、模板及历史更新。ours 的 tracker 内部还保留自己的同步和计时，这是实际实现的一部分；报告使用外层时长，没有把内部 `route['seconds']` 再加一遍。框的 finite/正宽高检查、行 append、末尾哈希与 JSON 写入在计时外。

模型构造和加载在所有帧计时之前；同时排除构造时的 baseline PreprocessorMM 均值/方差及 Hann 等静态对象建立。ours 的 Hann 在逐序列 initialize 内建立，会被计入。这忠实保留了各自当前生命周期，但 `full_service` 不能解读为包含进程启动、导入和全部构造开销的冷启动延迟。

`statistics` 对每个 arm 的原始逐帧秒数计算 `1000*mean`、`1000*np.quantile(..., .95)` 和 `帧数/sum(seconds)`。没有平均各序列的 P95，也没有把两个 arm 的时间混成一个 FPS。相对均值降低量 `100*(1−ours/baseline)` 的方向正确；它不是 P95 降低量。记录包含每帧顺序、两侧时长和 ours 路由，能够事后按顺序或动作检查分布。

默认 10 条×64 帧时，每侧 full_service 有 640 个样本，其中 10 次初始化；warm_service 仅用每条序列 frame≥9 的 55 个样本，共 550 个，排除初始化及前 8 次 track。动作和模态计数只使用 630 个非首帧样本。小于等于 9 帧被参数断言拒绝，warm 数组不会为空。该 warm 定义是固定截取规则，不是已验证的稳态判据；没有丢弃异常慢帧或使用未来帧预热。

源码中的重型 token 数为：主路径 385，局部辅助 193，完整辅助 385；因此 STOP/局部/扩大总数为 385/578/770。它们是路由诊断量，不是 GPU 实测 FLOPs、延迟比例或独立模块贡献。

## 对照归因与后续运行的边界

以下是结果解释限制，**不是当前发现的源码阻断，也不要求修改活动 tracker**：

- **两套原生实现的系统比较。** 原 CADTrack `sample_target` 使用 ceil/round 整数裁剪、零边界、resize 和 attention mask 生成，再由 PreprocessorMM 在 GPU 转换/归一化；ours `crop_pair` 使用连续 affine、每路全图均值边界、两次 warpAffine 和 CPU float32 归一化。ours 每次 crop 还计算全图均值，即使目标未越界。这些真实差异连同搜索轨迹、模板更新、head 调用次数、preview/policy、内部同步都进入时长；不能将总差值单独归因为 token 裁减、共享权重或某个模块。两侧 backbone 本来就共享两模态权重。
- **双模型常驻及交替解码。** 这能在同一设备上取得配对前缀样本，但改变显存、缓存与调度环境。先读 arm 可能承担更多文件缓存冷读；交替顺序降低固定顺序偏置，并不证明消除了全部缓存/顺序效应。结果不能替代单模型独占部署吞吐率、双卡墙钟吞吐率或完整 LasHeR 速度结论。
- **前缀与单次测量。** 首 10 个 dev 序列的 64 帧不是完整 97 条 dev，更不是 245 条 test。短前缀可能低估长期漂移、模板更新和追加查询频率；本脚本不输出重测方差或置信区间。固定 checkpoint 的测量也不等于最终锁定 C 权重的速度。
- **物理设备与环境尚未验收。** 源码保证两侧使用同一个可见 CUDA 设备；`physical_gpu` 字段记录的只是 `CUDA_VISIBLE_DEVICES` 字符串。此次未验证它与服务器实体卡/UUID 的映射，也未核验显存余量、其他进程干扰、目标环境依赖或运行期线程设置。脚本记录 GPU 名称和 PyTorch/OpenCV 线程数，但不是完整环境封存。独立运行应等活动任务协议允许且使用固定输入；本审查没有接入或启动任何控制器。
- **出处记录有范围。** 脚本保存 10 个主源码的运行结束时哈希，未覆盖 config 默认值、head、clip_box、Hann 等全部传递依赖，也未保存 manifest 的字节哈希。因而单份运行 JSON 不能充当完整依赖封存；复现时应同时保留相应源码版本、固定 manifest 和启动环境记录。未发现本次实际源码漂移。

没有 baseline PR/NPR/SR 重测入口，也没有评分调用；用户给定 baseline 72.74/69.16/57.84 不由这份脚本重算或验证。即使将来得到正的前缀均值降低量，也不能据此宣称 PR≥78/NPR≥72/SR≥62、完整系统提速 30% 或三个模块有效。

## 实际完成的静态核验

使用 `E:/anaconda/python.exe`，实际版本 `3.7.6 (default, Jan 8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)]`，执行标准库 `ast.parse` 读取新脚本文本，**PASS，exit 0**。没有导入目标脚本或 torch。首次 PATH 中 `E:/Scripts/python.exe` 调用因 `No pyvenv.cfg file` 在脚本执行前退出 1；随后从已安装解释器清单选择上述实际 Python 3.7 完成核验。这是本机语法核验，不是 Linux/CUDA 原生验收。

另直接读取 `cadtrack_source.tar.gz`，未解包或写入源码；归档 SHA256 为 `359e8f7717d741c29159d1dd36da89000d3c6b41b1a91215db12dc7bf18d40e0`。以下 7 个关键 baseline 文件与归档对应成员**逐字节相等**，无换行归一化才相等的情况。这验证本地保存副本的一致性；不冒充本次已核验远程历史环境或用户 baseline 的原始运行命令。

| 本地文件 | 归档及本地相同 SHA256 |
| --- | --- |
| `lib/test/tracker/cadtrack.py` | `6e0863824a890d6a16b9da5a7b5acd3e9f7bad3c4dd37f0a818c63aa38e433f2` |
| `lib/test/tracker/data_utils.py` | `873337d0419ecfdd08b64123cdc22a4936ccdd6af98a092bc2106c32cf401fe5` |
| `lib/train/data/processing_utils.py` | `7aca916f5e5f62e1865fbd322ffb63dc08197c544a241938b6c822d4bfb89bf6` |
| `lib/models/cadtrack/cadtrack.py` | `9cddf1ee0f30acb0aa8d3117af9a5ecabdce6cb97f7f5130270e404e6d934dfd` |
| `lib/models/cadtrack/vit_care.py` | `ae74beb13742793524bf8c69c2ddc2bcdc175c983ce30d37d32d71be6872e13c` |
| `experiments/cadtrack/cadtrack.yaml` | `462a56bdba611aa60f1602c22f7d2794781caea55d9e17ba01294da0873aca1a` |
| `lib/config/cadtrack/config.py` | `8a1dfa9bf4f500be0bf64b205c0772605ac8e2617dbb13705215423fc5884e56` |

`reports/full_pipeline_source_manifest.json` 自身 SHA256 为 `9984345d9f0a4377ee45cbc8747b83866a30c91fcb6652f2054166214deca50b`。重新计算其 11 个条目，**全部等于清单记录**；脚本未出现在 `complete_pipeline.py` 的调用路径中。审查期间这些活动源码没有修改。

| 活动源码 | 本地实际 SHA256＝清单 SHA256 |
| --- | --- |
| `scripts/train_policy.py` | `2fa070434680704c984d9a1b5b992973103971c47d692a6fcbea734571922913` |
| `scripts/check_policy_checkpoint.py` | `eaab05ae60827070a4e9d4a8013a407508acc14e699d7da1749ee8dd9b9100c1` |
| `scripts/train_trajectory_policy.py` | `d45ba30d89c4400f82d6900cd23155f23568c8b03075be5855e898b00b462740` |
| `scripts/run_task.py` | `90220bfc86bdb8c50bf4d8d45cf58ec10a3c54b2cacfae1f8bf09cac6be66e7f` |
| `scripts/run_tracking.py` | `3c43f74a53d824e98cbeb2e5c0affd7945ea434885a9ea92d9466266133b60b8` |
| `scripts/score_tracking.py` | `22f0cb04de31714f70019f25c8b6b3d48b0384a7aff812d25be367394b3b4ac4` |
| `scripts/complete_pipeline.py` | `56ed5936b1d3c59f37741963c88a7b66201772f1c04912034c5e086cc2bbb58f` |
| `dual_rgbt/model.py` | `434da9f818158c16bd7563e93ef44c65b2b726d881f35f94284fb6e8f9966e50` |
| `dual_rgbt/data.py` | `858e41c8e5218c7f5323a6e18cec00997a48d0323101384b71a73d61ea0280cb` |
| `dual_rgbt/tracker.py` | `406927019ffdcdbc8546755d9f7bff0859aac20766116e4a70f8c89132694fff` |
| `dual_rgbt/losses.py` | `d3b941610e666a236036fd67079af15d5003e366a56adf8acab8b34d8ac584b1` |

后续真实输入装载、原生因果跟踪、完整时长记录及结果解释仍需实际证据；本报告只完成上述本地源码审查与确定性静态核验。
