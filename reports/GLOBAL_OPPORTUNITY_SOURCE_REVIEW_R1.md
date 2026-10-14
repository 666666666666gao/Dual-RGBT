# GLOBAL_OPPORTUNITY_SOURCE_REVIEW — R1

结论：**SOURCE_FAIL，1 个阻断项**。仅完成 SOURCE_ONLY 审查；未作新实验运行准入。

请求配置为 Astra/max；实际模型路由 **UNATTESTED**，同 family/provisional。本报告不宣称已经取得独立异构模型审查认证。

## 阻断项 GO-SRC-001（P2，冻结契约）

`scripts/diagnose_global_opportunity.py:220-222` 新建 strict-reload clone 后只 `.cuda().eval()`、strict load、执行 probe；没有像主模型 `203-204` 那样关闭全部参数 `requires_grad`。factory 会创建可训练的参数，clone 没有继承主模型的冻结循环。它不满足本任务“全部参数冻结”的约束。

这**不是发生权重更新的证据**：`scripts/train_global_evidence_policy.py:61-71` 的 `probe_prediction` 有 `@torch.no_grad()`；clone 已处于 eval；新诊断源码没有 optimizer、backward、step 或 torch.save 调用。最小修正是在新 driver 中为 clone 显式关闭梯度标志，然后重新封存 driver/plan 审查。不要改旧 source30，也不要用一次权重更新作为这个只读诊断的准入。

## 输入与约束

- 初始 plan SHA：`e59b79d011cc8f1d93af6512f74514958c6c266ce47b66a1a6a89d0c6538b845`；与指定 e59b79… 完全一致。
- 新 driver SHA：`ea1230cb5987769f5925a0c353be5a7e08182c8cbc14c628d55b754257ce5c3a`。
- source31 全部与计划相符，前 30 个与原训练计划逐项相符。训练计划 SHA b5cf7d… 相符。初始 seal 已保存源文件字节副本。
- 部署 helper SHA：`df39ba61b0f7578ed9038193a9b2c68d12cae7b530ca6bdb5a39ba229771c5d5`。
- 启动 helper SHA：`b896344d92d00a9e491d9520e235469501497312ce997eac2756c4fd26f739f4`。
- 审查结束再哈希：所有 source31、计划、协议、AGENTS 与两个 helper 均未改变。
- 未导入项目或 Torch；未构建模型/NN、未使用 GPU/SSH、未读图片、manifest 或 test GT；未安装环境，未改项目/helper，未 spawn。
- 计数 nativea99863/exit0 和 A10/B2/C2 已完成梯度/Adam准入来自父任务陈述，本次没有重新读取其数据或冒充重验。

## 已核对的执行路径

1. **实际参考轨迹与反事实**：driver:30-57 复制现有廉价主模态调度，强制只执行 STOP 主模态完整编码；初帧 GT 初始化，后续 track 只接收当前 image。模板每五帧且主预测置信度 ≥ .65 时只更新实际选中主模态，未读辅助不更新辅助模板。driver:103 在 track 前 clone templates/history 并 copy box，115-117 的反事实使用保存状态，并用 `np.array_equal` 对照实际所选 STOP 框。
2. **GT边界**：`boxes[frame]` 在实际 track 完成后用于当前反事实 IoU与诊断 controls；GT中心不会写入 tracker、templates、history 或下一帧路由。训练 counterfactual 中 target 只参与 IoU/gain标签，候选位置来自预测和廉价预览。加载整个 train 标注数组不等于把未来标签输入模型；源码没有未来索引用于行为。
3. **动作与 controls**：两主模态各 STOP/LOCAL_MAIN/LOCAL_ALTERNATE/GLOBAL_LOCAL/GLOBAL_WIDE 共10；只有 `modality == selected` 才追加 dense/random/GT-local/GT-wide 共4。dense为同main源/256；random和GT-local为main源边长的一半/128；GT-wide为main边长/256。重型 token 分别385/193/193/385。没有把控制称作可部署动作、完整动作oracle或轨迹上界。
4. **接口、dtype、model head**：counterfactual 13 参数、query 6 参数均与当前实现一致。缓存 NumPy 1.26.4 实测所有 control geometry 为 shape(3,) float32；search/global geometry及 crop归一化常量为 float32。128 query 采用8×8自定义解码；256 query采用16×16。共享读出经过真实 `decode` 和 CENTER `get_score_map`；没有把8×8送入固定16格 `cal_bbox`。gain输入913+42=955，quality输入74。
5. **factory与checkpoints**：factory输入是已钉住的旧executor checkpoint，先strict载入并去除旧1690输入gain head，再构造955输入head、strict载入新A或B。诊断路径固定到训练计划的正式A/B run，要求实际SHA、原生 COMPLETE/exit0、6250 updates/50000 pairs、sanity=false；checkpoint内部还要求stage、step和训练计划SHA。M0不能通过该入口。protected CADTrack始末SHA检查存在，没有权重输出。
6. **冻结与重载**：主模型 `.eval()`，全参数 `requires_grad_(False)`，全state clone至CPU；诊断后逐项 torch.equal，所有主模型grad=None。probe七字段是 quality/gains/queries/primary_box/box/pixel_box/score，strict reload逐项torch.equal。clone标志未冻结是上述唯一阻断；实际265张量、bit equality仍须GPU sanity证明。
7. **IoU与数值含义**：实际与反事实均用同 `execute_boxes` → 原 `clip_box(margin=10)` → float32；IoU使用原图uncensored GT。CSV delta是两项float32 IoU取 `.item()` 后Python相减；检查用0.1f/0.9f展示了它可与float32直接相减不同，未宣称训练gainbitEqual。
8. **helper门禁和唯一启动**：两个helper先检SOURCE_PASS/零阻断、source31和plan SHA。部署只复制driver/plan，先检查远端旧30未变、新driver不存在；hidden --help在空CUDA可见集执行，只验证CLI，不能被写成GPU/NN通过。启动取实际正式checkpoint SHA；formal另要求同checkpoint/plan sanity的native COMPLETE/exit0和冻结/重载PASS。`launch(...,1,...)`沿用GPU1显存空闲检查，`run_task`在启动子进程前以exist_ok=False创建唯一task目录；driver独占run目录。没有杀进程或修改训练文件。
9. **隐私和科学范围**：完整CSV在run目录且两helper均无CSV收集/上传；协议明确PRIVATE_ONLY，公开只聚合报告。当前聚合按action合并两个主模态，CSV STOP行可提供同状态main源内/外分组依据；不要误称已输出完整分组统计。A阶段global preview未训练，当前候选差不能否定可学习性。这个诊断检验执行器查询潜力，不替代完整882训练、dev97与最终test245指标。

## 非阻断的证据范围限制

- **GO-EVID-001**：driver sanity只设置首序列最多10track、states>0且14比较/state，未显式要求10/2/28；SANITY写len(before)，formal helper未核对其恰为265，也未读取sanity COMPLETE计数。源码的七probe字段确定，但helper也未要求字段数量为7。依照父任务已验证的实际数据/固定manifest，这不是已有数据出错证据；实际admission仍必须核对10track、2states、28比较、265张量和七字段。
- **GO-EVID-002**：factory executor checkpoint在入口校验，出口只重验所选A/B、protected CADTrack和source31。源码没有任何checkpoint写入；若要声称全部输入始末不变，应保存factory input及plan末次哈希。这是证据覆盖不足，不是推断发生了修改。
- **GO-EVID-003**：helper没有自动消费review JSON的helper_sha256；本报告已封存准确helper字节，运行前应由控制者核对，不能将本次review延伸到修改后的helper。

## 实际检查与失败留存

- stdlib AST parse：31 source + 2 helper 全通过；4段helper生成的远端代码只格式化/compile通过，未执行。
- `source_audit_checks.py`：**28 PASS / 1 FAIL，真实exit1**；唯一FAIL为reload_clone_all_parameters_frozen。完整命令/输出与JSON保存在本目录，没有抹去失败。
- NumPy CPU检查仅执行AST提取的纯数值函数和controls构造，使用合成box与shape元数据；不读/创建像素图片，不导入Torch。
- 首次review自有日志tee在解释器退出时flush已关闭文件，造成exit1；seal在此前完整生成。原stderr与修复记录保存在`initial_review_runner_failure.txt`，初始seal没有重跑覆盖。这是review记录器错误，非项目源码问题。
- `initial_inputs_seal.json`、`inputs/`保留最初输入；每次reader/parse均保存`.command.txt/.output.txt`；`source_audit_checks.*`保留真实checker输出；`GLOBAL_OPPORTUNITY_SOURCE_REVIEW.json`供机器门禁读取。

本R1的SOURCE_FAIL不能用于部署/启动授权。修正新driver后必须以新driver/plan SHA作R2；旧30始终保持本seal值。
