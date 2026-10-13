# 本轮 C 完成与真实轨迹状态

原任务 `train_c_complete` 在2026-10-13 18:55:53 BJT正常结束，wrapper真实exit0。
正式输入为完整B latest，固定seed2027的20条train完整序列、两轮、stride5。
两轮17,708次track、40次初始化、3504次监督更新；不称全882条train逐帧遍历。
唯一原本地控制器在19:08节点确认终态，随后自行启动完整dev成本0/0.05两组。

终态后独立CPU审计实际工具exit0，固定真实manifest的20序列顺序、全部3504个
round/sequence/frame三元组和step0–3503精确相等；全部loss有限，记录公式与C源码一致。
最终checkpoint config与本run config（去sequence_names）相同。
与原完整B比较，所有257个冻结state张量逐项torch.equal；quality/gain各4个参数Tensor
均发生净变化。保存的8个Adam参数step均3504。该CPU审计没有构造模型、前向或重载预测。

末轮权重SHA256：`94be6239f7f6864cc193fff7c21d67c9479924c9432039c6ff16cd0d1223f15f`。
当前dev各成本使用此同一末轮权重；成本选择和测试指标尚未完成。

| 轮次 | 监督更新 | mean total/quality/gain loss | 中心越界 | STOP | RGB/TIR |
|---|---:|---|---:|---:|---|
| 1 | 1752 | 0.797038 / 0.751421 / 0.045616 | 278 (15.87%) | 517 | 621 / 1131 |
| 2 | 1752 | 0.750061 / 0.714237 / 0.035824 | 289 (16.50%) | 584 | 541 / 1211 |

两轮负收益标签比例均约39%。监督帧动作按STOP/PRIMARY/SECONDARY/PREVIEW/EXPAND分别为
517/392/392/370/81和584/437/403/275/53。它们来自不断更新中的策略和不同历史状态，
不是最终冻结模型的完整在线调用分布，不能推广到全部17,708帧或dev/test。
相同20序列并不意味着两轮具有相同搜索状态，不能把损失下降写成精度提升。

## 已观测到的覆盖问题

567/3504个监督帧中心在实际搜索crop外。事后按此记录分组：

| 轮次/中心状态 | 帧数 | mean quality loss | mean gain loss | STOP |
|---|---:|---:|---:|---:|
| 1 / 图内 | 1474 | 0.305878 | 0.045548 | 388 |
| 1 / 图外 | 278 | 3.113760 | 0.045982 | 129 |
| 2 / 图内 | 1463 | 0.257913 | 0.035611 | 436 |
| 2 / 图外 | 289 | 3.024280 | 0.036902 | 148 |

这些是条件统计，不证明越界造成全部quality误差，也不证明任何动作已恢复或失效。
losses.py先把GT xywh转换为xyxy，再将GT角点clamp到[0,1]；越界或部分越界的损失
因此不是原始全图定位误差。当前EXPAND只读取相同主搜索crop内的全部辅助token，
扩大的是辅助编码区域，**没有扩大原图搜索视野**。本轮不存在源图外部区域读取路径。

这为M11提供了真实闭环证据：后续应在全部dev上检查搜索覆盖与恢复，用原全图框误差
核对裁剪代理收益，决定是否需要更大的辅助视野或修改收益监督。不将567帧全称为失败，
不根据单个训练统计热改当前评测或提前承诺新机制。参数/预算选择继续只用dev，不用test。

监督帧按序列分解后，20条中8条出现中心越界，12条没有。三条序列贡献485/567（85.54%），
详见FULL_DEV_COVERAGE_PROTOCOL.md及completed_c_coverage_concentration.json。
已准备独立CPU离线分析器，但完整dev数据执行尚未发生；区域中心重入与准确跟踪恢复分开解释。

## 完整评价启动证据

19:10:20只读启动检查确认原dev wrappers4038334/4038394及子4038335/4038395存活，
命令分别cost0/0.05、同一C latest、完整dev；nvidia-smi只查授权0/1的UUID/PCI/显存及
compute PID，实际子PID分别位于实体GPU0和1。没有温度/功率查询。
尚未读取dev进度、评分或更改预处理。原控制器首观察节点约21:23:10，其后必要240秒。
第三组cost0.15及完整245条test继续由同一控制器依次执行。

原始：`reports/full_pipeline/stage_c_complete_20261013/`、
`reports/completed_trajectory_audit.json`、`completed_trajectory_audit_native.json`、
`completed_trajectory_state_analysis.json`、`c_complete_dev_startup.json`。
源码审查与真实CPU执行分开保存；这不是完整PR/NPR/SR、速度或三个模块的独立有效性验收。
