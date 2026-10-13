# Full-dev coverage source review

2026-10-13。总体结论 **WARN，阻断项 0**；指定范围内的源码正确性 **PASS**。这是独立上下文的源码审查，不构成完整 dev 数据执行或正式评测验收。

```yaml
scope: SOURCE_ONLY
requested_model: gpt-6-astra
requested_reasoning_effort: max
actual_model_effort_route: UNATTESTED
review_independence: same-family
acceptance_status: provisional
verdict: WARN
source_correctness: PASS
blocking_findings: 0
native_full_data: NOT_RUN
neural_execution: NOT_RUN
```

读取范围为新分析脚本、固定 `data.py` / `tracker.py` / `model.py` / `run_tracking.py` / `score_tracking.py`、相关裁剪与 token 源码、原 toolkit 源码、两份获准的历史报告及方法协议。只做源码读取和哈希核对；未查询 live controller/status/log，未读未完成 dev、图像或模型权重，未 SSH、启动 GPU/NN、导入或执行分析入口。仅新增本文档。

| 检查 | 结论 | 源码依据与边界 |
|---|---|---|
| 完整 dev 门槛 | PASS | `analyze_full_dev_coverage.py:98–112` 先读完成标志，要求 `dev`、`smoke_frames=0`、`shards=1/shard=0`、97 个唯一名称、配置与完成记录均为 54,646 帧；三类文件名称集合必须等于配置集合，读后再次累加帧数。固定 `run_tracking.py:133` 在全部序列写完后才写 COMPLETE。此结论针对固定导出链路生成的归档；本次没有核验任何真实归档内容。 |
| 帧与 trace 对齐 | PASS | `:30–49` 核对预测/GT 的同形四列、全部有限值、正预测宽高及初始框等于 GT。trace 帧号必须完整等于 `1..N-1`，动作限 0–4、模态限 0–1，置信度有限，含解码用时有限且为正。每序列排除初始化帧，因此通过全部数据断言后应有 `54,646−97=54,549` 个 tracked frames。 |
| 动作和 token | PASS | 主编码为 `1 CLS + 2×64 template + 256 search = 385`；STOP 辅助为 0，三个局部动作辅助为 `1+128+64=193`，EXPAND 为 385。与 `model.py:68–90,122–148,182–195` 及 `tracker.py:64–67` 一致，分析逐行断言这些值。重型 token 是计数，不是端到端延迟或 FLOPs。 |
| 因果裁剪几何 | PASS / WARN | `:54–65` 用前一帧预测，复现 `data.py:23–24,28,38` 的 float32 输入、`sqrt(float(w*h))*4` 和 float32 返回几何；对应 `tracker.py:34–43` 的前态裁剪与最终 clipped state。保存格式是 `run_tracking.py:112–113` 的六位小数，不能据此声称逐位重放原始状态。 |
| GT 有效性和误差 | PASS / WARN | `:62–72` 只在正宽高 GT 上统计连续 raw IoU/中心误差，且所有 GT 先通过有限值断言；x/y 可为 0 或负值。汇总是帧加权、排除初始化帧，不是 toolkit 阈值曲线或序列等权 OPE。具体 toolkit 差异见下文。 |
| 动作条件关联 | PASS | `:115,132–139` 明确不推断反事实收益；归档只保存选中最终框，没有每帧主框和全部替代动作框。路由频率、置信度、token 与定位误差的条件统计不能证明辅助模态收益。`model.py:124–136` 的 EXPAND 读取同一 search crop 的 256 个辅助 patch，没有扩大原图搜索视野。 |
| ROI 中心重入 | PASS | `:76–88` 在每条序列内要求前后相邻非初始化帧均有效、前帧 outside、当前帧 inside；不跨序列、不跨无效 GT。这里比较的是每帧各自的移动 ROI，统计当前重入帧的路由/误差，不能作为准确跟踪恢复、完整越界 episode 或恢复耗时。 |
| 空分组与分母 | PASS | `summarize():19–26` 对空分组返回 0 帧、全零动作/模态计数和 `None` 均值，序列及全局均使用此函数；JSON 写成 `null`。所有用于 raw IoU 均值的 mask 均含 valid，不把初始化为 NaN 的无效 GT IoU 纳入均值。`all_tracked_routes` 单独包含全部非初始化帧，避免把有效 GT 路由频率冒充全部路由频率。 |
| 方法协议 | PASS | `FULL_DEV_COVERAGE_PROTOCOL.md:28–59` 与实现一致，明确完整运行尚未执行、边界量化、帧加权、空分组、重入含义及权重只引用配置 SHA。其历史训练计数段未在本审查中重算。 |

保留以下非阻断 WARN：

1. **几何是名义裁剪正方形的近似中心归属。** `relative` 两轴采用闭区间 `[0,1]`。六位小数保存与 float32 运算可改变边缘帧判定；本审查没有真实帧证据来界定受影响数量。此统计也不证明完整目标框、实际采样像素或未填充图像内容被覆盖。现有脚本/协议已明确近似性质，无需为尚未观察到的边缘情况添加分支。
2. **原 toolkit 的有效性不是“丢弃无效帧”。** `basedataset.py:19–24,79–80` 对本 dev 链路的预测和 GT 取整。`metrics.py:266–272,309–315,351–357` 对任一 GT 分量 `<=0` 的帧设 `res=-1`，但保留 `len(res)` 分母；这会被 PR/NPR 的非负 `<=` 阈值计入，而不会通过 SR 的 `>` 阈值。另有首帧替换和非正预测宽高的前帧替换。分析脚本有意不复刻这些规则，不能将其 raw 均值换名为 PR/NPR/SR，也不能据此改写原正式工具包结果。
3. **完整数据实际验收仍为 NOT_RUN。** 本次未确认某个 run 已完成、未执行任何真实输入断言、未形成覆盖比例或定位数值。输入文件 SHA 绑定所读字节，配置中的 checkpoint SHA 只是引用，并未加载验证权重。须等原控制器成功回收归档后，由主流程执行并记录真实 exit/数据断言结果；源码审查与 CLI-help 均不能替代这一步。

没有发现需要修改分析源码的阻断项；不建议为上述已披露统计边界加入 fallback、容错读取或额外兼容层。

证据固定如下：

| 文件 | SHA256 / 核对结果 |
|---|---|
| `scripts/analyze_full_dev_coverage.py` | `2fb756272dfacb2868091b6db3cb2098f2898e3d47f5cff60a781a219e3b4dc0` |
| `reports/full_dev_coverage_cli_check.json` | `ecde9977821332a278577eed2b74138d39373524aa25b3349b2c33357b5aed9f`；历史记录为 Python 3.7 AST PASS、CLI exit 0、neural false、full-dev NOT_RUN，所载源码 SHA 与当前一致。 |
| `reports/full_pipeline_source_manifest.json` | `9984345d9f0a4377ee45cbc8747b83866a30c91fcb6652f2054166214deca50b`；独立重新计算的全部 11 项源码 SHA 均匹配。 |
| `docs/FULL_DEV_COVERAGE_PROTOCOL.md` | `f11df33120d4dac0c96f8096127bab516860b92cc6fcb69e43956f7057a43f39` |
| 原 toolkit `rgbt/metrics/metrics.py` | `9ae628c82201a00002cbef8a62c78746eb26acbd8c226304a4cd78e9223d2382` |
| 原 toolkit `rgbt/dataset/basedataset.py` | `96449e8090c4f1892fd9c8ea6ce34b12bf2b081e9400208870b1eee1d577b9d8` |
| 原 toolkit `rgbt/utils.py` | `5a95cbafef8371222e35179da58abf461ed03482695efdaad5ff904a077ba994` |

原 toolkit 读取根目录为 `C:/Users/gb/Desktop/document/cadtrack/toolkit`。路由元数据依据 `C:/Users/gb/.codex/skills/shared-references/local-codex-policy.md`；当前 reviewer 的实际后端/effort 没有可核验运行证明，故如实标为 UNATTESTED，不声称跨模型家族验收。

工具记录：本轮尝试用本地 `python` 做独立标准库 AST 解析时，命中的 `E:/Scripts/python.exe` 在解析前以 `No pyvenv.cfg file` / exit 1 终止。没有修改环境或转而执行分析脚本；因此本文只引用 SHA 匹配的既有 CLI/AST 记录，不声称本轮新 AST 检查通过。
