# B/C 策略与在线状态源码审查

日期：2026-10-13。审查结论：**WARN；B 当前源码未发现阻塞性正确性缺陷，C 尚未实现完备，不能据此进入最终 D 验收。**

```yaml
review_scope: source-only, future B/C preparation
review_independence: same-family
acceptance_status: provisional
requested_model: gpt-6-astra
requested_reasoning_effort: max
requested_metadata_source: experiment-bridge / local-codex-policy skill contract
actual_model_id: UNAVAILABLE
actual_reasoning_effort: UNAVAILABLE
actual_metadata_note: reviewer runtime does not expose an independently attested model or effort identifier
source_verdict: WARN
stage_b_source_blocking_issues: 0
stage_b_native_sanity: NOT_RUN_BY_THIS_REVIEW
stage_c_trajectory_refresh: NOT_IMPLEMENTED
benchmark_or_latency_measurement: NOT_RUN
```

本次是独立上下文的 Codex 同家族源码审查，不是跨家族验收。所请求的模型/推理强度不能冒充已核实的实际运行元数据。只读取本地文件并写本报告；没有 SSH、GPU 查询、神经网络运行、作业启动或执行代码修改。

按执行代理提供的状态，A 的 PID 3035566 正在 server2025 GPU 0 从原始预训练权重继续执行；本审查未远程核实该状态，也未修改其已导入的 model/data/loss 文件。A/M0 已有审查与验证不在本次重审范围内。B 和 Tracker 均是准备代码，没有被本报告描述为已训练、已部署或已评测。

审查了 `scripts/train_policy.py`、`dual_rgbt/{tracker,model,losses,data}.py`、`docs/EXPERIMENT_PLAN.md`、`AGENTS.md`；为核对实际接口，又读取了原 CADTrack tracker、CENTER head、Hann 函数、裁剪器、配置、数据清单生成器，以及 A 中冻结策略模块的声明。

## BLOCKING：阶段边界

**B：没有发现当前源码必须修复的阻塞性缺陷。** 这只通过源码层面，不能替代 AGENTS.md 要求的真实训练数据前向、反向、AdamW 更新和重载检查。此审查没有执行新的 B 原生 sanity。

**C/最终 D：真实轨迹监督刷新仍缺失。** `train_policy.py:25,84` 将 8 维 history 固定为零；`data.py:60–64` 使用历史 GT 模板与当前 GT 周围的抖动搜索裁剪。相比之下，`tracker.py:33,44–58` 使用预测搜索框、选择性模板更新与实际运动/尺度/查询间隔历史。这是计划明确允许的 B 起点，但不能充当 `EXPERIMENT_PLAN.md:36` 的 C 实现。质量头和收益头中的 history 输入列在 B 中没有由数据产生的梯度信号，只有可能的 AdamW 衰减；直接给它们输入真实非零 history 不能称为已学习历史策略。

最小后续工作：C 在训练序列上从初始框开始，保存每帧决策前的预测框、模板、history；在同一状态上枚举动作，GT 仅用于计算监督损失，再继续实际选择产生的轨迹。用这些样本训练策略/预览并刷新标签，在 dev 上确定预算，之后才能冻结进入 D。不要将后续 GT 送入 Tracker 的搜索框或模板更新。无需为此改动正在运行的 A。此项是已知的未完成阶段，不是新发现的 B 标签漏洞。

## NONBLOCKING：具体差异与最小处理

### N1：新路径明确省略了原 CADTrack 的 Hann 运动约束

证据：原 `lib/test/tracker/cadtrack.py:102–114` 对 score map 乘 `output_window`，再解码 bbox，并用加窗响应的最大值决定模板更新。新 `model.py:101–111,184–188` 使用原始 score 的 bbox/候选；`tracker.py:42–44` 的 STOP confidence 也是原始最大值。原 head 的 `cal_bbox` 只对输入 score 求 argmax（`lib/models/layers/head.py:141–155`），不会隐式补上 Hann。

因此，即使相同 score/size/offset，边缘峰比中心峰略高时，两种 tracker 的最终框与模板更新决策也可能不同。这是具体推理协议差异，不能把准确率差值完全归因于模态或查询策略；本审查没有证据证明它已经造成性能下降。当前 B 与新 `predict()` 都不加窗，二者内部一致，故不列为 B 正确性阻塞。

最小处理：在 B 前明确保留当前无窗协议，或决定恢复原 Hann。若恢复，必须让 STOP bbox、候选选择和 B 的实际动作标签使用同一解码约定；不能只在最终 Tracker 输出端补窗、继续沿用无窗标签。不要修改活动中的 A 文件来做这项决定，也不要求重跑 baseline 精度。

### N2：模板更新的两个 confidence 不同义，不能假定 .65 已校准

证据：`tracker.py:42–44` 将 STOP 的主热图最大值和查询后的 `sigmoid(quality_logit)` 共用 `.65`。后者在 `losses.py:31–32` 以最终框的 GT IoU 为 BCE 软目标；前者在 `losses.py:27–30` 受目标中心热图 focal loss 监督。原 CADTrack 的 `.65` 还对应加窗响应（配置 `cadtrack.yaml:87–88`），不是 IoU 预测。

两种分数都可用作启发式 confidence，但相同数值不代表相同模板可靠性。这没有混入 signed utility，也不是 GT 泄漏；它是 C 中实际模板轨迹的校准问题。最小处理是在 dev 上校准 STOP 与查询分支的模板更新阈值，并记录所用定义；校准前只能将现值称为初始启发式，不能称为继承了 baseline 的同等更新门槛。无需增加新的置信度网络。

### N3：后续实测成本不能直接覆盖收益头正在使用的 token 特征

证据：`model.py:50–51` 的 `token_cost` 既在 `gain_predictions` 的输入中出现（`:160`），也用于 `predict` 的收益减成本决策（`:187`）。计划要求先用 token 代理、随后用实测延迟校准（计划 `:35`）。当前固定 token 代理训练没有问题；但若 C 直接把这一个 buffer 换成秒数，已经训练的收益回归器输入也会改变，改变的不只是动作惩罚。

最小处理：保持收益头的既有 token 输入定义；将实测增量成本仅用于选择时的惩罚，或在明确变更特征定义后重新训练对应策略。无需现在加入未使用的兼容接口。193/385 是 token 数代理，不是已测得的延迟比例。

### N4：B 的目标中心越界分支没有当前数据依据

`train_policy.py:93–101` 的注释声称几何抖动可能使目标中心离开搜索框，并用 `inside` 分支过滤。实际上 `data.py:23–28` 先从同一个 GT 框求中心，再平移 `u * side`，其中 `u ∈ [-.375,.375]`；归一化后的目标中心恒为 `.5 - u ∈ [.125,.875]`，与尺度扰动无关。因此 B 的当前数据路径不会触发中心越界，且 8×8 标签索引有效。

最小处理可直接在 B 的所有样本上计算空间交叉熵，删除这段不准确的注释/防御分支。真实 C 的预测搜索裁剪可能出现中心越界，应在实现那个实际路径时按其数据定义处理；不要据此给冻结中的 data.py 增加新分支。

## 源码确认正确的部分

| 检查 | 证据与结论 |
|---|---|
| 冻结和优化器 | `train_policy.py:49–54,67` 重新构造模型、严格载入 A 的 state_dict，再冻结 backbone/decode/box_head/corrector；参数列表只剩 preview/quality/gain。state_dict 不携带 A 的 requires_grad 标志，因此 A 曾冻结策略模块不影响 B 的默认可训练状态。全模型 eval 固定 BN/dropout；preview 本身没有 BN/dropout，eval 不会关闭 autograd。 |
| 同状态 signed 收益 | `train_policy.py:69–83` 对每个 modality 只生成一个 primary，再复用完全相同的 templates/search/primary 枚举四个动作，目标为 `main_loss - action_loss`，允许负值。STOP 的预测收益和追加成本均为零（`model.py:163,50`）。质量目标是负定位损失，不是分类置信度。 |
| 标签新鲜度 | 每一步用更新前的当前 preview 选候选并实际执行，随后在相同权重下重新求可微 preview，最后才 optimizer.step。没有长期缓存旧 preview 的动作标签；两次 preview 之间没有随机层或参数更新。 |
| 梯度路径 | no_grad 只覆盖冻结执行器及标签生成（`:69–83`）；`:84–105` 的 quality、gain 和空间交叉熵能反传到各自策略层及 preview。argmax 候选是离散的，空间交叉熵提供预览图监督；没有声称梯度穿过 argmax。是否实际非零、是否发生参数更新仍需原生 sanity。 |
| 张量维度 | quality 输入 `32+32+8+2=74`；gain 的共享上下文为 `128+8+768+3+4+2=913`，加候选 `768+4+1+4` 后为1690；corrector 为 `768+768+4+4=1544`。监督的质量为 `[B,2]`，每个 modality 的四项收益为 `[B,4]`，与损失输入一致。 |
| 数值约定 | GT xywh 与预测 cxcywh 在同一搜索坐标中转为 xyxy，使用同一 GIoU/L1 定位损失。score 的熵计算没有 log(0) 的现存路径：原 CENTER head 将 score 钳在 `[1e-4,1-1e-4]`（`head.py:176–178`）。不需要再添加 epsilon/fallback。 |
| 训练/测试边界 | B 只请求 manifest 的 train（`:64–65`）；清单构建器只从 trainingset 读取 train/dev 标注，test 只列名称做分离校验。没有源码路径把 test GT 送进 B。未对实际服务器数据重新审计。 |
| 在线因果状态 | Tracker 只在 initialize 接受初始框，后续 track(image) 从上一预测框裁剪；框从 crop 坐标正确映射回图像，clip 后成为下一状态。8维历史长度与网络一致，包含上一帧模态、confidence、中心运动、尺度变化和查询间隔。 |
| 模板写入 | 首模板保持固定；每次允许更新时写第二模板的 primary 通道，只有 action 非零才写 auxiliary 通道（`tracker.py:44–51`）。两个模态可持有不同更新时刻，这是当前代码的实际语义。裁剪器仍计算双路裁剪，不能把通道选择写入误称为全部双路预处理都被省略。 |
| 真正的选择性执行 | predict 只跑一个完整主路；STOP 不调用 correct；非STOP只编码所选 auxiliary 区域，未重跑主路。局部仍读取原图 patch 并保留原位置编码。每路 token 含两张模板128和CLS 1：主路385、局部193、扩大385；不能把搜索token的64/256当作整个执行成本。 |

## 原生验证与计时验收边界

B 正式运行前，最小 sanity 是一个真实 train 批次的标签生成、反向和 AdamW 更新：检查 preview/quality/gain 的有限非零梯度及实际参数更新，冻结执行器参数与 BN 状态不变；保存后严格重载，比较同输入的预测。A 的已通过 sanity 不能证明这条新的策略训练路径已执行成功。无需重复 A 全套实验或 baseline 精度。

`tracker.py:17–26,30–60` 在开始前与结束时同步 CUDA，以 wall clock 计入裁剪、归一化、CPU/GPU传输、双路preview、模态/动作选择、主路、实际追加查询、映射与模板更新。初始化时间单独返回，后续评测必须显式计入序列总时间；`track()['seconds']` 单独取平均不会自动包含初始化。此接口接收已解码的 image，图片读取/解码在计时边界之外，应如实标注。

当前没有动作实测增量成本、完整序列聚合器或 PR/NPR/SR/P95/FPS 结果。本报告不能证明任何实际提速；也不能由 token 数推导30%延迟降低。最终评测需记录总时间（包含初始化）、平均/P95、FPS、动作比例及token量，并在相同计时边界下给出实测延迟对照；这不要求重测用户禁止重跑的 baseline 精度。

LasHeR PR≥78、NPR≥72、SR≥62 与实际延迟目标均仍未被本次审查验证。保留用户提供的 baseline 72.74/69.16/57.84 的外部来源归属，不将其描述为本项目复测结果。

## 同轮最小修复复核

报告写入后，执行代理修改了 `scripts/train_policy.py:93–99`：删除 `inside` 过滤与分支，将注释改为当前有界抖动保证中心位于搜索区域内，并对全部样本计算空间交叉熵。审查者读取了这份实际 diff；它准确关闭 N4，未改变其余标签、梯度、优化器或执行器逻辑。改动不是本审查者实施。目标中心在 `[.125,.875]` 内，乘8后取整得到有效的行列索引，`row * 8 + column` 与 `[B,64]` logits 对应；去掉批内过滤不会改变原本所有样本均有效时的损失归约。

本次限定差异的复核结论为 `followup_source_verdict: PASS`、`N4: CLOSED`；仍属 same-family/provisional，未执行 GPU 验证。N4 的原始发现保留；当前源码阻塞项仍为0。N1–N3 作为 **C/D 前必须明确并完成相应校准的设计项** 保留，C 和原生 sanity 的未完成状态不变。

按执行代理后续通知，当前先等待 A 终态并依据真实 utility/覆盖诊断执行器，B 暂不启动；不能用扩大收益头掩盖执行动作本身缺乏收益。本审查未读取远程训练指标，不把该通知写成自己核验过的训练结果。A 的 model/data/loss/训练源码未在本次复核中改动。
