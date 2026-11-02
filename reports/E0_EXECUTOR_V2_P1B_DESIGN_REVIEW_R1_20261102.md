# P1B Direct-E0 Design Review R1

日期：2026-11-02（America/New_York）。审查根：`C:/Users/gb/projects/Dual-RGBT`。

**结论：DESIGN_REVIEW / IMPLEMENTATION_REQUIRED；无 GPU 准入，无执行授权。** 现有 direct E0 原语可以支持保持架构的 P1B，但指定源码中尚无满足下述完整合同的 P1B 实现。本报告不生成 SOURCE_PASS，不证明 P1A 完成、P1B 已实现、训练有效或科学接受。

模型归属：本会话原生 Codex 助手；精确后端模型身份未经独立证明，`actual_model=UNATTESTED`。未调用其他模型或聊天，未调用 gpt-6-astra；请求的 Astra 审计路由在当前工具接口不可用。按同族、暂定工程意见处理；`scientific_result_audit_accepted=false`，不替代正式 experiment audit。

仅阅读用户指定的七份仓库文件与两份用户计划。没有 SSH、GPU 查询、测试、数据采集、训练、评估、部署或源码修改。接口审计中的 P1A 进度是该文档的历史陈述，本次没有核实当前进度或完成状态。两份计划的文件名/正文日期为 2026-11-03，本报告保留其标识，不把其日期或运行陈述当作本次已发生的见证。

## 1. Direct E0 与 Candidate E1 的边界

| 证据位置（仓库相对路径，行号为本次读取版本） | 实际接口或行为 | P1B 含义 |
|---|---|---|
| `dual_rgbt/model.py:34-48` | `DualRGBT` 的直接命名空间是 `backbone/decode/box_head/preview/quality/gain`；quality 输入 74，gain 输入 1690；STOP 成本为零 | P1A 成品权重必须以同一 direct E0 factory 严格加载，保留全部键和形状 |
| `dual_rgbt/model.py:105-130` | 四个位置是 first、远离 first 的 second、辅助 preview 峰 third、first；动作 1/2/3 读同一搜索区内 8×8 token，动作 4 读该区全部 16×16 token | 没有 K8 池、独立全图查询坐标或新的 broad crop |
| `dual_rgbt/model.py:150-165` | gain 拼接固定四种 action one-hot，输出 `[0,g1,g2,g3,g4]` | STOP 加四个查询动作，不能把“四动作”误读成总共四个输出 |
| `dual_rgbt/model.py:181-195` | 先 quality argmax 决定主模态，再对 `gains-cost*token_cost` argmax，只执行所选查询 | 保留主模态调度、查询和 STOP 的原决策语义 |
| `scripts/train_executor_v2.py:32-54,157-161` | P1A 是 direct `DualRGBT`，保存整个 `net`；policy 模块冻结 | P1B 是在成品执行器上刷新三策略模块，不是换执行网络 |
| `reports/E0_EXECUTOR_V2_P1B_INTERFACE_AUDIT_20261102.md:19-24` | 审计文档描述 candidate trainer 使用 `executor.preview/executor.quality/query_gain`，并依赖候选几何/rank、缓存等合同 | 不能将 candidate trainer 或旧缓存改个 checkpoint 路径即当成 direct E0 P1B |

用户计划第一部分 `Dual-RGBT_plan_20261103.md:139-171` 自身区分 direct E0 与全图 K 候选；第二部分 `Dual-RGBT_plan_part2_架构模块与执行提示词_20261103.md:40-63,80` 同时记录 direct gain 1690 与候选 gain 955。输入维度、动作定义、命名空间和输出合同不同，严格 state_dict/数据合同不能视为同一接口。

证据边界：本次未读 candidate E1 的实现文件；其命名空间事实来自指定接口审计，候选结构来自两份指定计划。直接从源码验证的是 E0 合同，不声称完成了 E1 实现的独立源码审计，也未实际尝试跨架构加载。

## 2. 现有两个策略入口的可复用部分和缺口

`scripts/train_policy.py:24-32,68-91` 已有“两主模态 + 每路四动作”反事实监督，以及重新计算 preview 的可微训练路径；`103-104` 检查三个模块梯度。它不能原样充当 P1B：

- `48-53` 使用隐式 `DualRGBT()` factory、一个统一 LR，且只冻结参数，没有完整执行器 buffer 不变见证。
- `63-70,84` 用旧随机 GT crop 样本和全零历史；`dual_rgbt/data.py:64-79` 明确由搜索帧 GT 生成 crop，不是实际轨迹状态。
- `69-80` 未传 Hann；`92-98` 的预览位置损失假定中心位于 crop 内；真实漂移轨迹不满足这个假定。
- `117-119` 只存 latest，没有最后五轮、输入链和 native 完成绑定。

`scripts/train_trajectory_policy.py:69-87` 已有正确的因果骨架：先保存旧 state/templates/history，执行真实 `track()`，之后才生成当前帧标签；`94-109` 枚举四动作并传同一个 Hann。它也不能原样充当 P1B：

- `45-50` 隐式 factory，冻结 preview，只训练 quality/gain；不满足 preview 1e-5。
- `31-34,42-44` 默认 20 条、stride 5、两轮，不能表示 882 条完整声明覆盖；其参数必须显式冻结并记录。
- `112-120` 在轨迹中直接更新 policy；不是“冻结行为策略采集，再离线训练”的流程。
- `159-163` 仅存 latest，未绑定 P1A terminal/native/source/manifest 证据。

应复用这些原语和公式，新增版本化入口；不要修改这两个历史入口或活动 P1A 及其依赖。

## 3. 最小 P1B 初始化和冻结合同

1. DropMoE/SOT 各自只接收对应已完成 P1A 的一个明确 checkpoint。最小默认采用预先声明的最终 epoch 20，不根据本次没有执行的 dev/TEST 选权重；保留 P1A epoch 16-20。若以后要用其他 epoch，另行精确绑定。
2. 显式构造 `DualRGBT(factory_checkpoint)`，再对完整 P1A `net` 做 `strict=True` 加载。绑定 factory 初始化权重及其 SHA，即使随后被完整成品权重覆盖，也不能隐式依赖机器上的默认 ep14。
3. **不能用 P1A 的 `initialize(cadtrack,parent)` 载入 P1B 成品。** `scripts/train_executor_v2.py:34-46` 先载 parent 再覆盖 backbone/decode/head，这会把成品执行器换回初始 CADTrack 执行器。P1B 只做 factory 构建和成品严格加载，不再覆盖其执行器张量。
4. 整个 model 保持 `eval()`，冻结 backbone/decode/box_head 参数及其 buffers；保留 token_cost 不变。只有 preview、quality、gain 参数显式设为可训练，不能依赖 P1A 保存过的 requires_grad 标志。eval 不会切断 autograd；现有 preview 没有 BN/dropout（`dual_rgbt/model.py:14-23`）。
5. 两组 AdamW：preview LR `1e-5`；quality+gain LR `1e-4`；weight decay `1e-4`。最小沿用历史 policy 梯度裁剪 `1.0`（`scripts/train_policy.py:105`），先声明并冻结，不借机改执行器配方或增加训练调度。
6. 保存冻结前的所有执行器参数/buffer和 token_cost，采集后、每轮训练后、smoke 更新后逐项 `torch.equal`。不仅检查“没有梯度”，也检查数值不变；这正对应本项目 decode/head 的运行状态风险，不添加无证据的兼容或 fallback。

P1A factory 与完整依赖快照仍需未来 implementation source review 核验。`model.py:37-40` 实际读取配置并构造 CADTrack，本次仅查看指定文件，未宣称验证这些依赖的闭包。

## 4. 因果 TRAIN 采集合同

行为模型是冻结的完整 P1A direct E0 snapshot（后续刷新轮使用冻结的上一轮 P1B snapshot）；采集过程中没有 optimizer 更新。cost 固定 `0.05`，seed `2027`，与计划第二部分 `251-252` 对齐；精度/TF32、stride、每序列范围、轮数和 dtype 写入新草案，不能沿用隐式默认。

先声明 882 个唯一 TRAIN 序列及其顺序/分片，全序列前向跟踪；最小首版对每个初始化后有效 TRAIN 帧生成标签，即 `label_stride=1`。任何降采样都必须在未来草案中预声明，不能称作逐状态全覆盖。无效标注帧仍按真实预测推进 tracker，只不产生监督；不得在未来有效帧上重新 GT 初始化以修复漂移。初始化采用声明的首帧 GT，若数据不符合首帧可用合同则报告具体序列并阻断，不添加悄悄跳序列的分支。

每个帧 t 按以下顺序：

1. 读取当前 RGB/TIR 图像；复制实际 pre-state、两个模板、真实 history、frame/since_query 和 Hann。用 pre-state 构造 search，不读取当前帧 GT 来决定 crop。采用 `data.py:35-51` 的原 crop/归一化，搜索边长是 `4*sqrt(w*h)`，不要替换成 4w、4h 或 max(w,h)。
2. 调用原 `Tracker.track(image)`，只让行为模型执行一次真实决策与状态更新。保存 actual route、输出框和更新后的 tracker 状态证据。当前 GT 不进入 predict，也不参与状态纠偏、动作选择、模板接受或重置。
3. 真实决策提交以后，才访问当前帧有效 GT，按已保存 search geometry 转成归一化 xywh 标签；不读取未来帧标签。第一帧 GT 初始化是明确例外。
4. 在保存的 pre-templates/search/history 上，以 eval/no_grad 枚举两路 primary，以及每路 1..4 的 correct，得到两个 primary 和八个 query 的完整标签，不仅对 actual route 打标签。枚举不能调用 `Tracker.track()`，不能将 oracle 框写进 tracker。STOP 的输出就是该模态 primary，raw gain 恒为零。
5. 枚举前后比较 tracker 的 state/templates/history/frame/since_query，确认完全不变；执行器 buffers 同样不变。日志中的实际路由只来自步骤 2，不能被反事实结果或 oracle 替换。

真实历史的源码证据：`tracker.py:34-61` 从预测框生成位移、尺度、confidence 和 since_query；`47-54` 动态模板按原预测置信度/帧计数写入。P1B 不增加 GT 写入、接受头、时序 token 或恢复机制。历史 snapshot 必须 `.clone()`，模板也必须 `.clone()`；tracker 模板有原地切片赋值，保留引用会把过去样本改成未来模板。

最小 shard 保存精确的 pre-templates/search 数值（CPU tensor）、history、geometry、序列/帧、post-decision GT 标签、actual route/state摘要、两路四动作的 candidate indices/boxes/outcomes、行为模型 SHA、执行器 SHA、Hann 配置和 schema/source 绑定。这样离线拟合不依赖旧 observer 缓存，也不需要修改 tracker 增加模板重建系统。该方案可能占较多磁盘，实施前按 tensor 字节数估算容量；本审查未证明 882 条逐帧缓存的容量或用时。不能未经审查改成低精度缓存或静默抽样。

## 5. 标签、Hann 和离线梯度

最小建议明确命名为 **direct-E0 signed-localization-loss policy refresh**，沿用 `train_policy.py:73-99` / `train_trajectory_policy.py:90-110` 已有目标，而不是宣称移植了未审阅的 candidate Joint-Quality：

```text
L_m       = localization_loss(primary_box_m, target)
Q*_m      = -L_m
G*_(m,a)  = L_m - localization_loss(correct_box_(m,a), target), a=1..4
G*_(m,0)  = 0
L_policy  = SmoothL1(Q,Q*) + mean_m SmoothL1(G_m[1:5],G*_m)
            + 0.1 * mean_modality spatial_CE(preview_map, target_cell)
```

完整保留负 gain，不做截零、仅正样本或仅 chosen-action 监督。标签是现有 localization loss 的有符号改善，不能在报告中称为纯 IoU 增益；网络接口不要求标签一定是 IoU。cost 只在原决策式扣一次，raw gain 标签不减 cost，不增加 learnable STOP。

Hann 必须统一：行为 `Tracker.track` 传 `output_window`（`tracker.py:24,35-37`）；所有 primary 标签和 correct 标签传同一个 Hann（`model.py:99-101,144-146`）；因此 gain 的 score统计、候选索引和训练输入都按同样窗口定义。不能只对最终框补乘窗，或把 candidate readout 的无窗路径套进来。

离线训练必须从原始 decision tensors 重新运行 `preview_state()`，且这次在启用 autograd 的上下文中，不能直接训练 detached 的历史 pool/descriptor/quality/gain：

- `model.py:55-61` 提供 preview→pool/descriptor→quality 的连续梯度。
- `model.py:157-165` 提供 preview→pool/descriptor→gain 的连续梯度；冻结/detach 的 primary features 不妨碍这条梯度。
- `model.py:62` 的相似图可接受空间 CE；当前真实漂移状态的 GT 中心可能在 crop 外。只对有限、有效且中心严格位于 `[0,1)` 的样本计算空间 CE；其余样本保留 quality/gain 标签但不造边缘格标签、不为此重裁 crop。这是 `train_policy.py:93` 的旧假定遇上真实轨迹的具体缺口。
- `predict()` 自带 no_grad（`model.py:181`）；不能用它产生训练图。candidate argmax/gather 不可微，也不能声称通过 argmax 把 counterfactual box loss反传给 preview。

**必须处理 preview 改变导致的第三动作标签漂移。** `model.py:112-114` 的 third 取决于辅助 preview map。离线更新 preview 后，旧第三动作标签可能对应不同区域。首版采用最直接方案：每个训练 batch 用该步更新前的当前 policy，在 no_grad 下重新枚举两模态全部四动作并重算标签，然后用相同参数、相同固定输入在 grad 上下文重算 preview 及 quality/gain loss，最后执行一次 optimizer.step。模型始终 eval；这一轮目标、candidate indices/boxes 与预测分支使用同一份候选定义。冻结执行器不等于禁止额外离线前向。

采集的旧标签保留为采集 snapshot 的可核验产物，不能自动当成当前 preview 的标签。上述重算只发生在已保存的因果状态上，不回放或修改真实轨迹；current GT 只用于离线标签，不能选择 candidate 或输入 crop。首版不增加陈旧标签兼容器或可微 argmax。

每个 smoke 更新检查 preview/quality/gain 均有有限且非零梯度，并在若干真实更新后各有张量数值变化；所有 executor 参数/buffers仍逐项相等，且 optimizer只包含三策略模块。不能用“optimizer存在”或 loss有限代替梯度可达见证。

## 6. 刷新轮、最后五轮和完成绑定

最小首版流程：冻结行为 snapshot采集完整 TRAIN → 离线拟合 → 保留末五轮并写完成证据。不要边收边改行为模型；该流程是基于冻结行为轨迹的监督学习，不宣称是最终新策略自己的轨迹。

为覆盖计划中的 C 类闭环刷新，预声明第二轮：冻结第一轮最后一个 policy checkpoint，重新采集 882 条 TRAIN因果轨迹，再按同一公式离线拟合。两轮均用固定执行器和独立 collection/run IDs；这复用旧 trajectory trainer默认两轮的意图（`scripts/train_trajectory_policy.py:32,64`），不在本次执行。每轮 epoch 数及优化步数必须在新草案中给出；建议每轮至少 5 epoch，保留各轮最后五个独立 `epoch_NNN.pth`，不只是 latest。若只实施第一轮，就明确 P1B 闭环刷新尚未完成。

P1B checkpoint仍保存完整 direct E0 `net`，另存 optimizer、epoch/step、collection摘要、factory/source/input hashes；末五轮权重逐一登记 SHA。严格重载比较同一保存状态上的 quality/gain、route和box；当前报告没有这种运行见证。每轮、每 shard都要有预期序列/帧/标签数量、actual数量、缺失/重复清单；只有 declared shards全量完成才能输出 collection COMPLETE，不能把小 smoke或旧缓存充当完整 TRAIN。

启动前最小证据链：

| 绑定 | 必须验证的关系 |
|---|---|
| P1A completion | 对应新 P1A task的 native COMPLETE/exit0，config 与 COMPLETE均存在；20 epoch、50,000 updates、TRAIN-only及 retained16-20 与已冻结计划一致。仅 native exit0不足；旧 R3 noop tasks不能算 smoke或训练完成 |
| model | 完整 epoch20 checkpoint SHA、保存的 epoch/step/config 与 P1A run一致；不接受别的分支的同名 latest |
| factory/source | 显式 factory checkpoint SHA、构建配置、direct E0 model/tracker/data和新 P1B依赖的完整 source map；启动快照绑定到 run，活动 P1A源码不动 |
| data | 精确 manifest bytes SHA、882 TRAIN名称集合和顺序、数据/标注来源与读入记录；不得借计划中最终979条方案把97 dev混进本轮 |
| collection | 每 shard绑定同一 executor、行为 policy、manifest、cost/Hann、seed/precision和完整 schema；不同分支/刷新轮的数据不得混为同一来源 |
| offline fit | 消费的 shard清单与每个 SHA、候选重算方案、更新数、冻结见证、末五轮及terminal receipt均绑定 |
| execution | 新计划、source review、绝对 argv、host/task/run和native command相符；未来单独 CPU检查和真实 TRAIN smoke，当前设计审查不提供这些准入 |

P1A源码现只记录若干输入hash（`train_executor_v2.py:90-97`），COMPLETE也没有全量 artifacts map（`177-181`）。P1B新入口应读取现有实际字段并绑定其文件SHA、native命令和checkpoint内部config；不得要求receipt里不存在的plan_sha256字段，也不得为满足新合同热改P1A。若实际证据不足，报告该缺口并等待既有任务终结后的补充证据。

## 7. 两份计划中的语义冲突及建议处理

1. **K8/Joint-Quality：** 第二部分 `251-252` 写 K8，但第一部分 `151-161,171` 明确 direct E0是四个固定查询。建议未来P1B精确计划写“direct E0 STOP+4，K不适用；只借用preview/head双LR和两模态监督”，标签采用第5节现有公式。candidate ranking/Joint-Quality若另有特定损失要求，先单列公式和源证据再审；不能把它当作已实现。不是把K8悄悄改成K4，也不替换架构。
2. **旧paired collector：** 第二部分 `251` 只说复用 `collect_e0_current_full.py` 流程；指定接口审计 `22-24` 说明其旧缓存/父权重约束。可借“按序列收集”的组织方式，不能直接复用缓存作为新执行器标签。第一部分 `193` 的历史统计亦不能当成本轮训练数据来源。
3. **Hann归属：** 第二部分 `30,86` 的“只在推理/训练无窗”描述针对旧实现；当前P1A `103,135-136` 已传Hann。第一部分 `239,291` 关于无窗query指向candidate evidence路径，不能据此断言direct E0 `correct()`无窗。
4. **候选几何：** 第二部分 `51` 将second描述成“曼哈顿>2”，源码 `model.py:109-111` 排除的是横纵距离均≤2的方形邻域。动作4是原search的全部256 token，不是另裁2倍全图区域；不把后续C3 broad-recovery设计塞进P1B。
5. **后续架构：** 第二部分 `93-127` 的EvidenceHead、LoRA、missing填充、temporal tokens、AcceptHead属于后续v2阶段，均不在本P1B最小方案中；尤其STOP保持0、gain保持1690、preview/quality/gain形状保持不变。
6. **dev/TEST和覆盖范围：** 两份计划有后续dev选型、TEST和979条最终重训步骤；本次范围只做882 TRAIN的设计，不执行或授权这些阶段，也不根据计划中历史test数字选择本轮epoch/cost。

## 8. 下一步最小可实现文件集合

只建议新增以下四个源码文件；名称为实施建议，当前未创建，现有P1A/model/tracker/data及历史policy入口不改：

| 新文件 | 最小职责 |
|---|---|
| `dual_rgbt/p1b_direct_e0.py` | 显式factory+P1A严格加载、冻结合同、纯函数两模态四动作label/训练loss、shard schema与绑定校验。只调用既有E0原语，不扩展网络 |
| `scripts/collect_p1b_direct_e0_train.py` | 主入口guard；读取精确计划/输入；冻结行为模型；保存pre-state；原tracker推进；post-decision完整反事实标签；不可变TRAIN shards和完成回执 |
| `scripts/train_p1b_direct_e0_policy.py` | 主入口guard；读取全部已绑定shards；重算当前候选标签；preview/head双LR；梯度与冻结见证；末五轮、strict reload、terminal evidence |
| `tests/test_p1b_direct_e0.py` | CPU可用的接口stub与小张量检查：两模态×四动作及STOP顺序、pre-state克隆/GT时序、Hann逐路传递、反事实不改tracker、第三动作变化触发标签重算、三个policy梯度连通、executor/buffers不变、来源不符拒绝、末五轮索引和CLI入口 |

另生成新的精确计划及source/input maps属于未来实施产物，不是本报告写入项。最小任务序列是：实现上述文件 → CPU检查及入口见证 → 新源码/绑定独立审查 → 单独授权的真实TRAIN smoke → 完整采集/离线拟合。不得跳过中间环节用本DESIGN_REVIEW发起GPU。

这些检查针对已有具体缺口，避免引入通用兼容层、隐式factory fallback、静默过滤数据或吞掉绑定错误的try/except。若实现暴露新的具体问题，再以源码证据处理。

## 9. 本次读取文件的 SHA-256

以下仅固定本次设计证据，不是未来P1B启动source map；P1B尚无源码hash。

| 文件 | SHA-256 |
|---|---|
| `reports/E0_EXECUTOR_V2_P1B_INTERFACE_AUDIT_20261102.md` | `ed3b970d90b64a57b7f710b6a53c7b32675f914724f03d3552e878d94f308b0a` |
| `scripts/train_executor_v2.py` | `71d36cfcac98973b37a8dab22ea24abeaffbecef10f334beff4a1bddbd8cab06` |
| `scripts/train_policy.py` | `2fa070434680704c984d9a1b5b992973103971c47d692a6fcbea734571922913` |
| `scripts/train_trajectory_policy.py` | `d45ba30d89c4400f82d6900cd23155f23568c8b03075be5855e898b00b462740` |
| `dual_rgbt/model.py` | `e5895b115436838304c5b165ca32b6ec641bd982de5066abe3a503b49cdb79f4` |
| `dual_rgbt/tracker.py` | `406927019ffdcdbc8546755d9f7bff0859aac20766116e4a70f8c89132694fff` |
| `dual_rgbt/data.py` | `133d00bc4f3a6693809789a21b33cb81aaab1634c1a200d42684273f09995ba0` |
| `C:/Users/gb/Desktop/document/Dual-RGBT plan/Dual-RGBT_plan_20261103.md` | `02c3a00e9190a59af7b6a46d99e715d059637814803de02a613ba6ce75c311d3` |
| `C:/Users/gb/Desktop/document/Dual-RGBT plan/Dual-RGBT_plan_part2_架构模块与执行提示词_20261103.md` | `410fbd381f0fd6c80167d6f50625be124979421904fc0ebf68a2e66154f5d661` |
