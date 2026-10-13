# Dual-RGBT 冻结 train20 动作收益审计

日期：2026-10-14（Asia/Shanghai）。审计角色：fresh Codex reviewer `/root/audit_train_reward_diagnostic`。

**总体判定：WARN；已保存结果的确定性核验 PASS；代码语义审查 PASS / provisional；0 个计算正确性阻断项。**

`review_independence: same-family`，`acceptance_status: provisional`。技能要求的路由为 `gpt-6-astra` / `max`；实际 model / effort routing 均为 **UNATTESTED**。本报告不把技能期望值当作宿主路由证明。

本审计使用 `experiment-audit`，读取原代码与保存的结果，独立编写标准库核验器；没有执行项目分析脚本、导入 torch/numpy、访问服务器、读图像、训练、运行 NN、加载/重载权重、重测 baseline 或更改项目文件。输出只写指定私有报告目录。可取得的原工具调用完整导出为 **UNAVAILABLE**，没有伪造 transcript。

## 实际核验与证据层级

`run_verification.ps1` 调用现有本地 Python 3.12，以 `-B` 运行 `verify_saved_rewards.py`。实际 process exit **0**，宿主 exec chunk **63c021**，UTC `2026-10-13T21:12:28.6114969` 至 `21:12:30.4505286`，stderr 空。命令、stdout/stderr、SHA 与真实退出码保存在 `verification_command_receipt.json`，核验器没有导入项目代码。

- 核对 **40 个输入**的 SHA/字节长度，读取前后没有变化；计划封存的 **19 项源 SHA 全部相等**，计划、正式/ sanity config、startup 及完成回执互相一致。
- 从原 CSV 独立重算完整报告的 **968 个数值字段**：962 个完全相同，其余 6 个仅因独立实现的 `math.fsum/n` 与原 `statistics.mean` 舍入不同，最大绝对差 **1.3877787807814457e-17**。整数、键集合、动作集合和行内 gain/loss 差全部精确一致。比较容差 1e-12；实际差远低于容差，所有非零差路径均列入机器结果。
- 新收到的 manifest 字节快照为 **102,528,114 bytes**，SHA **389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc**，与已封存计划相同。本审计自行哈希并解析该本地快照，而非仅复述父线程哈希。
- 对 **14,016 行执行动作框**，使用 manifest 中的训练 GT（先按原程序转 float32）独立重算原图 IoU，**14,016/14,016 完全一致**。CSV 不含完整 crop 几何、每帧跟踪状态及主框坐标，故没有声称仅凭 CSV 重算全部原始/执行 GIoU-L1 或逐行 raw→pixel 映射。
- 本地没有 B 权重 payload，权重 fresh byte verification **UNAVAILABLE**。原程序前后 SHA/265 项状态比较，以及父线程保存的远端权重 SHA 是可追溯的原执行证据，不是本审计的本地权重重哈希。

## A. GT 来源与输入隔离：PASS

GT 来源是 manifest 的数据集 `init.txt` 标注，不是模型框生成的参考。`scripts/prepare_data.py:17–25` 读取 init.txt，记录 boxes，并以正宽/正高生成 valid_frames；`:27–30` 检查拆分名字不交叉。本次从封存快照重新验证所选 20 序列的图像/标注长度一致、GT 数值有限、正宽高判定恰等于 valid_frames。

manifest 中 train **882**、dev **97**、test 名字 **245**；三组名字两两交集为零。诊断所选 20 条全部属于 train，与 dev/test 均无交集，并与 `trajectory_data_plan.json` 顺序及保存的 all882 config 前 20 条一致。没有读取 test 标注或据此提出调参建议。NumPy RandomState 的排列没有在本次标准库进程重放；固定 seed2027 的源码与保存的选择列表已核对。

`scripts/diagnose_policy_rewards.py:105–127` 中，GT 只用于首帧初始化、目标有效性/监督帧筛选及损失标签。`:111–117` 在 `track` 前复制 previous/templates/history，然后用 previous 重新裁搜索图。`:127–137` 的 `labels`、`gain_predictions` 和 `correct` 没有 target 参数；`scripts/train_policy.py:24–32` 的 labels 函数只从图像、模板和历史产生预览、主输出和候选。

“模型输入无 GT”的准确含义是**没有当前帧 GT 作为特征、动作候选或后续搜索框**；首帧 GT 初始化仍然存在，模板和后续历史自然依赖初始化。不能宣称绝对没有任何 GT 依赖。`dual_rgbt/tracker.py:34–61` 的后续框、模板写入和历史都由预测及置信度驱动，未见用当前帧 GT 拉回搜索或重置失败轨迹。

`dual_rgbt/data.py:51–66` 中用于 Stage B 的 `LasHeRPairs` 有 GT 裁剪路径，但当前诊断不实例化该 Dataset。不能因该未调用路径而误判本诊断为 GT 维护搜索。

## B. 三种收益、映射与归一化：PASS

`dual_rgbt/losses.py:7–21` 的 clipped loss：预测 cxcywh→xyxy，不裁预测；GT xywh→xyxy 后把四坐标 clamp 到 [0,1]。损失为 `2*(1-GIoU) + 5*mean(abs(predicted_xyxy-truth_xyxy))`。

`scripts/diagnose_policy_rewards.py:29–42` 的 uncensored loss 保留完整 GT，GIoU/L1 公式及权重相同。`:45–49` 的 execute_box 与 `dual_rgbt/tracker.py:38–43` 使用相同 normalized cxcywh→像素 xywh 算式、相同 `clip_box(H,W,margin=10)` 和 float32 状态写入。`lib/utils/box_ops.py:97–106` 是这两个调用共同引用的实际 clip_box。

执行 loss 在 `diagnose_policy_rewards.py:141–146` 把执行后的像素 xywh 映回**同一个搜索坐标系**，再转 cxcywh 并按原 tensor dtype 计算 uncensored loss。因此它包含原像素边界约束及 float32/坐标往返效应；仅移除 GT clamp 并不等价于执行误差。该 L1 是搜索坐标中的 corner L1，不是原图像素 L1；另存的原图 IoU是独立几何量。

每状态两模态各有一个主输出和四个动作，共 10 个输出、8 个 gain 对。`:147–151` 的 null control 对 GT 四角都在 crop 内的状态比较 clipped/uncensored loss；本次 **1445×10=14450** 个原输出检查与 CSV 的对应相等关系一致。它不是 14450 个独立状态或训练更新。

`:175–176` 将选中反事实动作执行框与 tracker 实际返回状态做逐位比较；原正式回执记录 **1752 次**，覆盖有监督状态，不是对全部 8854 次 track 的逐帧冗余复验。其源码映射一致性可审，原运行断言可追溯；本次未重跑 NN 重现这些 tensor 比较。

未见把精度除以模型自身最大输出的可疑归一化。坐标除以 crop side、IoU 的几何 union、token cost 除以固定 385、均值按状态/动作数计算，都与声明定义一致。`model.py:151–154` 的 score/sum 与 entropy 只是策略输入特征，不是性能指标的自归一化。

## C. 结果存在、冻结和执行回执：PASS（证据限定）

正式 `COMPLETE.json`、sanity `SANITY.json`、两份 config/CSV 和完整 analysis JSON 均存在且 SHA 已绑定。正式结果还与 `reward_diagnostic_status.json.result` 逐字段相同；sanity 与 startup 中记录的结果/原 SHA 一致，sanity 的 16 行也恰等于正式 CSV 的前 16 行。

`diagnose_policy_rewards.py:59,88–91,186–192` 使用 eval/no_grad、全模型 requires_grad(False)，在 rollout 前后逐项比较 model.state_dict()，并复查 B checkpoint SHA；代码没有创建 optimizer、backward 或保存新 checkpoint。**265 指同一模型 state_dict 中全部参数/持久 buffer 项的计数，覆盖每个诊断进程开始与结束的冻结状态；不包括会正常变化的 Tracker 框/模板/history，也不是 265 次更新。** 本结论只针对诊断，不能外推到正在训练的 Stage C。

sanity 第二次 strict load 的检查在 `:201–207`，比较的是最后一个 sanity pre-state 上 predict 的模态、动作、框、gains；没有宣称重载检查覆盖所有正式状态。正式阶段另起进程按同一原 B strict load，不复用 sanity tracker 状态（`queue_reward_diagnosis.py:69–92`）。

原执行身份核对如下：

| 证据 | 内容 | 可支持的结论 |
|---|---|---|
| startup formal record（`c882_reward_formal_startup.json:972` 起） | wrapper1072641、child1072642、CUDA_VISIBLE_DEVICES=1、原 argv、RUNNING | 原正式任务身份及启动，不证明完成 |
| `reward_diagnostic_status.json:23–38`；`reward_diagnostic_native_completion.json:3–18` | 同 task/argv/start/pid，COMPLETE，exit_code0 | 保存的原诊断子命令终态；耗时1216.372606秒 |
| startup sanity terminal（`:1209–1226`） | task check_reward_diag_input_sanity_20261014、pid1070684、exit_code0 | 原 sanity 完成；耗时26.878684秒 |
| 本地 owner 等待回执 | owner47240 的实际 handle；owner_exit_code_observed=0 | 原本地队列 owner 退出码0 |
| 同一回执 waiter30620 | scope 明确 waiter exit0 不等于 owner exit0 | waiter 的原工具 native exit码本审计未取得，UNAVAILABLE |
| 本审计 CPU exec | actual exit0 / chunk63c021 | 本次标准库核验命令通过，与上述 NN native exit 分属不同命令 |

`scripts/run_task.py:27–32` 明确由 subprocess.run 的真实 returncode 写终态并退出；原保存终态不是从进度百分比或 owner 消失推断。`queue_reward_diagnosis.py:23–30,79–94` 要求原 task exit0、结果断言和文件收集成功后才写 COMPLETE_DIAGNOSTIC。

`reward_diagnostic_native_completion.json:20–30` 的远端文件 SHA 与本地三文件独立哈希相符；父线程本次采集不等于本审计访问了服务器。原工具 trace 未提供，不把转存 JSON 冒充原始宿主工具 transcript。19 项 source seal 只证明列表中的 19 文件；本次额外记录了 box_ops 等已读取文件 SHA，没有把它扩称整个依赖环境的完整封存。

## D. 指标调用与完整算术：PASS

clipped/uncensored/executed loss、execute_box、full_image_iou 都在诊断 main 的实际输出循环中被调用；三种损失和其差进入 CSV。原分析脚本的 pair_summary/state_summary 在 main 中用于所有分区、8 个模态-动作子集及20条序列，结果均存在且本次完整独立重算（`scripts/analyze_policy_rewards.py:27–82,144–164`）。没有把未调用的诊断函数输出当作结果。

| 比较 | 严格符号变化数 | 翻转两边绝对收益较小值：最小 / 中位 / 最大 | 平均绝对 gain 差 |
|---|---:|---|---:|
| clipped→uncensored | 191 | 0.0000348091 / 0.0028104782 / 0.1085443497 | 0.0036883914 |
| uncensored→executed | 6 | 0.0005722046 / 0.0033118129 / 0.0117547512 | 0.0001999355 |
| clipped→executed | 197 | 0.0000348091 / 0.0028104782 / 0.1085443497 | 0.0038841961 |

这里严格要求两 gain 乘积<0，零不计，未加近零阈值。两阶段 flip 集合在本数据中交集为零，并集恰为197；不能把这种相加关系当成任意数据上的普遍恒等式。分位数的含义是 `min(abs(g1),abs(g2))` 的幅度，不是绝对差、相对百分比或置信区间。

预算 oracle 固定 CSV 的实际 chosen_modality，仅比较 STOP 与该模态4个动作，成本0.05、STOP0、local193/385、EXPAND1；loss 改善为正，utility=gain−cost。Python float64 标量同值按 STOP→action 顺序选择；实际 route 直接读 CSV，没有声称逐位重演 GPU float32 argmax。

- clipped/executed oracle 动作分歧 **56/1752**：10 STOP→query，6 query→STOP，40 query→query。
- 执行效用下 clipped oracle 平均 regret **0.00016439238816459066**，实际策略平均 regret **0.0683987663848361**。
- 实际动作计数 STOP/PRIMARY/SECONDARY/PREVIEW/EXPAND = **104/690/200/123/635**；平均辅助 tokens **251.1324200913242**。
- 实际动作平均执行 loss 改善 **0.10359177744381776**，平均原图 IoU 改善 **0.03747898034471106**（STOP改善为0，按1752状态均值）。后者不是 PR/NPR/SR，也不是整条轨迹换策略后的提高。
- 当前主模态的执行主 loss 比另一模态更高的状态 **935/1752**；平均相对较优主模态 loss 差 **0.15654770337623547**。这是同一 pre-state 下两个主输出的离线对比，不证明切换模态后的长期改善或调度修复效果。

## E. 范围与口径：WARN（声明须保持限定）

完整20序列的 manifest 总帧数 **8874**，包括20次初始化；非首帧 track 是 **8854**。stride5候选为 **1761**；`ab_moto2north0` 的10,15,20,25,30,35,40,45,50帧 GT 无效，9个候选不产生监督，剩 **1752** 个有效状态，恰 **14016** 个收益对。GT无效帧仍在源码正常 track 循环中，只跳过监督；原 COMPLETE/native exit0 与该完整循环一致。CSV 本身只记录监督状态，不能单独充当8854行逐帧日志。

互斥分区：GT四角在crop内 **1445**，中心在内但角点超出 **49**，中心在外 **258**。对应总 flip 为 **6/7/184**，预算动作分歧为 **0/2/54**。外围分区内保存的原图IoU改善为0，不应把搜索坐标loss降低直接等同于重新找回目标。

这是**一个 seed、一份冻结 B、实际访问过的训练状态上的20完整序列诊断**。不是882完整训练/诊断，不是97dev或245test完整精度，不是重放原 C 的更新轨迹。每个状态的8个动作共用前状态，不能作为8个独立样本做置信区间；报告按状态或动作对加权，不是toolkit序列等权SR。oracle 使用 GT，不能部署，其 regret 也不是轨迹性能上界。没有真实延迟测量或三模块有效性消融。

`reference_test_selection_summary.json:2–7,31–43` 的参考C20 checkpoint SHA为94be…、cost0.05来自完整dev选择；诊断使用B SHA ba49…。本报告没有将二者混成同一模型，也没有用该dev表或test作本次奖励改动的效益证据。

## F. 评价类型：real_gt（train-only paired counterfactual diagnosis）

GT 是数据集标注，执行框由冻结模型产生。标签真实不自动扩大指标的适用范围：支持“已访问训练状态上 clipped 与实际执行收益存在可测差异”，不支持“修复奖励后跟踪精度已提高”“可部署 oracle”“完整训练或最终测试已完成”。

## 可操作问题与声明影响

1. **仍有一处文档历史状态，WARN。** `docs/REWARD_DIAGNOSTIC_PROTOCOL.md:48` 仍说实际 sanity 尚待执行。该表述只适合作为预执行历史，不能代表当前状态。另一个 `docs/POLICY_REWARD_ANALYSIS_PROTOCOL.md` 已在本次算术核验后由父线程添加第3–5行，明确实际结果已完成并把以下 NOT_RUN 限定为历史；本审计重新读取并核验新SHA后，关闭该文档旧时态警告。原失败检查与旧报告完整保留，详见文末封存补充。
2. **证据不可扩大。** 保留权重 payload 与原工具 trace UNAVAILABLE、实际路由 UNATTESTED；保留265状态项、1752执行框检查及sanity单一输入重载的准确覆盖范围。
3. **结果解释。** 197严格flip的中位幅度约0.00281，预算oracle有56分歧但平均执行utility差0.0001644；这些是诊断事实，不能直接宣称有效策略翻转、训练修复成功或长期精度提升。没有从test标注提出阈值/训练建议。

| 声明 | 判定 |
|---|---|
| 正式20完整train序列，拆分不交叉，8854track/1752标签/14016对 | 支持（manifest+源码+完成回执；CSV只含标签帧） |
| 当前帧GT未输入策略/候选或维护搜索 | 支持；须保留GT初始化限定 |
| 0optimizer/0backward、模型265项冻结 | 支持原代码/运行断言与回执；本次没有重跑tensor检查 |
| 保存分析统计、预算oracle及模态诊断算术正确 | 支持；968字段独立核验通过 |
| 同状态收益定义有差异 | 支持 |
| 由此证明新方法有效、长期轨迹提高、全量最终指标达标 | 不支持 |

## 输入封存、执行记录与路径纠正

全部40输入的绝对路径、字节长度、SHA见 `independent_verification.json.input_sha256_bytes` 和 `audited_inputs.json`；机器审计见 `EXPERIMENT_AUDIT.json`。核心结果文件：

| 文件 | SHA256 |
|---|---|
| reward_diagnostic_plan.json | f0edc89d98077c49a031dec0f86c4091e5a6255187ff6de0f881a9cb8d85374b |
| formal config.json | 464da732994f85bcb2604fcdaa3268e80fd65851bc5a17000c687be4ebe2e53e |
| formal COMPLETE.json | 27a541f2f2c5d9e0154bb1b75dea4b44fd3ad2965e3b3f8f41b2d95834178a57 |
| formal action_rewards.csv | 9d3b905e732afbdef200343bb6262c4a8105ad28c87c3a4e16a936d708d98ffa |
| independent_verification.json | dd5055827ea0e3f8de7e7e9c5fc1a74515f49625e5bdb952acec3157bbe876b1 |
| verify_saved_rewards.py | 121cd8e9143d0fb6c05b1541e9fbe2ed9595309c4cb0aa2294f886c7a99c1ab2 |

最初父线程提供的 `dual_rgbt/training.py` 不存在，清单读取实际 exit1（chunk201c41）；收到明确更正后读取 `dual_rgbt/losses.py`，没有构造替代文件。另一次审计员定位命令猜测不存在的 build_manifest.py/prepare_manifest.py，实际 exit1（chunkefde6e），随后从 rg 搜索结果定位真实 prepare_data.py；没有运行或更改项目。两次仅为只读定位失败，完整保留在命令结果记录中，不掩盖为通过，也不混入原诊断失败。

`001-audit.request.json` 保存收到的任务及后续材料原文；`001-audit.response.md` 与本完整报告相同；`run.meta.json`、`001-audit.meta.json` 明确路由/trace可用性。这些是事后审计记录，**不是宿主原始 tool-call transcript**。


## 最终封存补充：仅一份文档在核验后更新

本完整最终报告对应 `FINAL_AUDIT.json`。原第一次完整报告 `EXPERIMENT_AUDIT.md/json`、原请求/回答及其SHA均原样保留；它们代表第一次审计快照。

第一次最终输入封存检查实际 **exit1 / chunkc04690**，正确识别父线程在算术核验后更新了 `docs/POLICY_REWARD_ANALYSIS_PROTOCOL.md`。该文件从SHA `441caf7946046b0a9d0692f0387d3cce5931e430ec7e0d64e2bd744de85ee885` 变为 `fc18ac4122b9f795e3336f7152ed0e8c0547b3fe18be195c30bbb90b30e4ec49`，新字节数3514；其余39输入保持不变。父线程已确认这是它添加正式结果/历史状态说明，随后停止改动本次审计输入。

展示这份新中文文档的Python命令因默认GBK stdout无法编码箭头，实际 **exit1 / chunk75ee96**；读取/计算哈希没有写项目。后续用 `Get-Content -Encoding utf8` 完整重读，实际 **exit0 / chunk28e60f**。没有把这两次失败隐藏，也没有把展示错误混作实验失败。

新第3–5行准确区分已完成结果与原NOT_RUN历史，解决该文档警告。这里新增的原CPU分析93a7b3退出码仍属于文档的原执行声明，其原宿主trace未取得；本审计的独立CPU证据依然是实际exit0的63c021。

增量封存仅重新哈希输入/既有审计输出并审阅该文档，没有重新计算CSV、执行NN或连接服务器。全部39个其余输入（包括19源、计划、manifest、config/COMPLETE/CSV和结果数值）与原独立核验一致，故无需重复已通过的968字段算术核验。当前40输入SHA见 `FINAL_AUDIT.json`；变化、真实失败码、父线程确认原文及旧新SHA见 `POST_AUDIT_INPUT_UPDATE.json`。
