# 已结束固定 C 轨迹的 CPU 完整性审计：源码审查

日期：2026-10-13。结论：**SOURCE-ONLY PASS，源码阻断项 0。** 当前实现与固定 20 条 train 序列、2 rounds、stride=5 的产物格式一致，可在真实终态后用于所声明的 CPU 完整性核对。**本次没有执行该审计入口；native completed-C audit 为 NOT RUN。** 这份报告不宣称 C 当前已经结束或任何 C 权重检查已经通过。

```yaml
review_scope: source-only, completed fixed stage-C integrity audit
review_context: independent reviewer reading source and saved completed-B artifacts
review_independence: same-family
acceptance_status: provisional
requested_model: gpt-6-astra
requested_reasoning_effort: max
requested_metadata_source: local-codex-policy.md
actual_model_id: UNAVAILABLE
actual_reasoning_effort: UNAVAILABLE
actual_model_route: UNATTESTED
source_verdict: PASS
source_blocking_issues: 0
native_completed_trajectory_audit: NOT_RUN
actual_C_checkpoint_or_C_CSV_read: NOT_RUN
actual_full_manifest_reconstruction: NOT_RUN_LOCAL_MANIFEST_ABSENT
neural_forward_backward_optimizer_or_reload: NOT_RUN
ssh_or_gpu_queries: NOT_RUN
local_pipeline_source_hashes_checked: 11
local_pipeline_source_hashes_equal_saved_manifest: true
remote_running_source_state: NOT_CHECKED
```

遵循项目 `AGENTS.md` 与 [local-codex-policy.md](C:/Users/gb/.codex/skills/shared-references/local-codex-policy.md)。上述请求模型不是已证实的运行路由。本次只写本报告；没有修改代码、导入审计入口或项目模型、读取活动 C 日志/权重、调用 SSH/GPU、启动实验或改变原控制器。

主审查文件为 `scripts/audit_completed_trajectory.py`，逐项对照 `train_trajectory_policy.py`、`train_policy.py`、`complete_pipeline.py`、`run_task.py`、`dual_rgbt/model.py`、`reports/trajectory_data_plan.json`、11 项源码清单和已回收完成 B 的 config/COMPLETE/CSV。既有 B checkpoint 检查及 C 输入 sanity 报告只作历史格式依据，没有重新执行其中的神经网络。

## 读取门槛与 CPU 边界

`audit_completed_trajectory.py:15–16` 首先要求固定任务 `train_c_complete/status.json` 同时为 `state=COMPLETE`、`exit_code=0`，之后才读 C 的 COMPLETE/config，`:31` 才读 CSV，`:41–42` 才加载 B/C checkpoint。没有绕过此门槛的分支。`run_task.py:28–32` 等子进程结束后，以真实 returncode 写终态，再以同码退出；控制器 `complete_pipeline.py:46–48,124–134` 同样先等终态成功再收取 C 完成文件。这是对 wrapper 持久化退出凭据的读取，不能把本次源码核对写成已观测到 wrapper 的实际退出。

`:9` 在导入 Torch 前把 `CUDA_VISIBLE_DEVICES` 设为空，两次 `torch.load` 均显式 `map_location='cpu'`。入口没有模型/Tracker 导入、构造模型、CUDA 调用、forward、backward、optimizer 构造/step 或 subprocess/SSH 路径。`torch.equal` 和保存 Tensor 的 `.item()` 是 CPU 产物检查，不执行训练。Torch 在终态门槛前被导入，但权重与 CSV 均在门槛后才读。入口是顶层代码，因此本次只做 AST 解析，没有 import 或运行它。

## 固定协议与记录顺序

审计 `:19–25` 固定 sequences=20、rounds=2、seed=2027、label_stride=5 和正式完成 B 的路径；`RandomState(seed).permutation(len(manifest['train']))[:20]` 与训练入口 `:42–44` 完全相同，并要求 JSON 的 sequence_names 等于重建顺序。

审计 `:26–33` 按 round、所选 sequence、原始 frame 逐项重建全部监督三元组，使用与训练 `:64–79` 相同的 `range(1,len(boxes))`、stride 整除条件和 valid_frames 过滤；CSV 三元组列表必须完整且顺序完全相等，step 列必须为 0 至 3503。该列表比较会检查所有监督行，不能只靠总行数或末行掩盖缺行、重复或顺序错误。

训练在监督过滤前已经执行 `track` 并累加 frames，所以 `2 * sum(len(boxes)-1)` 正确覆盖所有非初始化 track 调用，包括没有 GT 监督的帧。审计同时要求 COMPLETE 的两轮/每轮20条、steps=3504、tracked_frames=17708。另有40次序列初始化，不计入17708。

本次用本地标准库重算既存 plan：20 条、每轮 `sum(frames-1)=8854`、两轮17708；不考虑 valid_frames 时每轮共有1761个 stride-5位置。plan记录有效监督1752/轮，故对应3504；唯一含无效 GT 的所选序列为 `ab_moto2north0`，记录45个无效帧。**本地没有 data/manifest.json，不能独立重建具体1752个有效索引或实际 NumPy 选序。** 这些是未来入口使用真实 manifest 执行的检查；本报告没有把 plan 中的1752当作本次已逐索引验证的结果。

## B 到 C、保存配置及参数检查

本次直接重算已回收 B CSV，得到6250行、step 0–6249连续、每个epoch 1250行；全部 loss/quality_loss/gain_loss/preview_loss/negative_fraction 有限。B config 的10000样本、batch_size=8、5 epochs对应6250次更新，COMPLETE也记录5/6250；B公式 `loss=quality_loss+gain_loss+0.1*preview_loss` 的最大残差为 `1.3113021846145045e-7`。

审计 `:41–44` 从固定正式 B 文件读取 `step=6250`，要求 C checkpoint 为 round=2、step=3504。C训练 `:46–50` 只加载 B 的 net 并新建 AdamW，故 C计数从0开始；3504不是B+C的累计计数。审计没有读取 C sanity 的两步权重。

C JSON是 `dict(vars(args), sequence_names=...)`，checkpoint保存 `vars(args)`；`:44` 仅剔除 JSON 的 sequence_names 后做字典全相等，正好匹配训练 `:55–56,159–160`。它包含 run、policy、cost_weight、sanity等其余字段。这里验证的是 **C checkpoint config 与 C JSON一致**，不是B checkpoint config的另一次独立核对；B的配置证据来自上述既存JSON/CSV。固定控制器命令还显式给出cost_weight=.05，不带sanity；本审计没有把JSON其余每个字段重新逐项对照命令行。

`:45–48` 先要求B/C net键集合相同，再排除quality/gain前缀，对其余257项逐一 `torch.equal`。范围包括已记录250项执行器状态、preview六项卷积参数和token_cost一项buffer；与既存C输入sanity的257一致。由`:49–52`要求quality和gain各至少一项净变化。当前这两个头各含两个带bias的Linear、无持久buffer，因此检查的是参数变化，不能扩大为每个参数元素均变化或获得正精度收益。本次未加载这些实际checkpoint；257项相等和两头变化仍待终态CPU审计。

质量头和收益头各有4个参数Tensor，AdamW只接收这8项。PyTorch版本记录为1.12.1+cu113；官方v1.12.1 AdamW实现将step初始化为单元素Tensor，每步加1，默认非capturable计算也使用`.item()`。因此审计`:53–54`读取8个保存state的`state['step'].item()`并要求均为3504，符合此项目的新建AdamW保存格式，无需添加旧版本整数step兼容分支。只读取state字典，不构造或运行optimizer。[PyTorch v1.12.1 AdamW源码](https://raw.githubusercontent.com/pytorch/pytorch/v1.12.1/torch/optim/adamw.py)

## 统计口径与结论边界

`:37–40` 检查C CSV四个loss/比例字段有限，并检查C特有的 `loss=quality_loss+gain_loss`；没有误加B的preview CE。`:56–62`的各轮损失、中心越界次数、action/modality counts均从监督CSV分组；`:69–71`明确限定于3504个监督帧，不能外推为17708次track的全程路由分布。chosen route记录于该帧更新前，训练中策略持续更新，也不是最终固定checkpoint的完整评测轨迹。

未来JSON中的PASS只表示该固定训练产物检查通过：不证明full882遍历、checkpoint实际forward重载、完整dev/test PR/NPR/SR、完整系统速度或三个模块的独立贡献。最终checkpoint SHA只绑定届时读取的文件，本次没有生成该SHA或任何实际C审计JSON。SOURCE-ONLY PASS不能替代未来真实的冻结比较、优化器状态/完整CSV检查和完整指标验收。

## 已执行的静态核验与文件身份

本次使用现有CPython3.13.12的标准库 `ast.parse(..., feature_version=(3,7))`，审计入口及11份活动源码均通过Python3.7语法模式解析；同时人工核对该审计入口未使用3.8+语法/API。这是语法检查，**不是目标Python3.7环境的native运行**。没有导入NumPy/Torch。首个PATH中的 `E:/Scripts/python.exe` 因缺少pyvenv.cfg直接退出，未执行检查；随后通过 `uv python find --no-python-downloads` 定位现有解释器完成核验，没有安装环境。

11项本地源码SHA256逐项等于 `reports/full_pipeline_source_manifest.json`；远端状态未查询。审查时Git HEAD为 `9938b291733b6e72d937abb8f3a5fbae7e05748e`，新审计脚本为未跟踪文件；以下SHA绑定实际读到的文件，不能只用HEAD代替。

| 文件 | SHA256 |
|---|---|
| scripts/audit_completed_trajectory.py | 868a653d89bf2faaba619c3294b4dad29fd7c15af6b54bb68f5eb9d2f7aa2c2b |
| reports/full_pipeline_source_manifest.json | 9984345d9f0a4377ee45cbc8747b83866a30c91fcb6652f2054166214deca50b |
| reports/trajectory_data_plan.json | d97cb6f79219e8a9c760cdac5c9f5d1d141ec427742383c71881bad38f50f583 |
| reports/full_pipeline/stage_b_refreshed_20261013/config.json | 4317f9cbfc10c40cc60efe9b947fbeedf6abfa2a55dea21837fe7fd33e534dd3 |
| reports/full_pipeline/stage_b_refreshed_20261013/COMPLETE.json | f7a1db8148769bd7e63944e719802675eddde7a388c1512e3668153cf76259da |
| reports/full_pipeline/stage_b_refreshed_20261013/policy_steps.csv | 62c296440b35c6ff19fad5e3f30b5b1853dbf576afa7158c8aaa944ca7582792 |

11项活动清单的逐项核对结果：

| 源码 | 本地SHA等于已保存manifest |
|---|---|
| scripts/train_policy.py | true |
| scripts/check_policy_checkpoint.py | true |
| scripts/train_trajectory_policy.py | true |
| scripts/run_task.py | true |
| scripts/run_tracking.py | true |
| scripts/score_tracking.py | true |
| scripts/complete_pipeline.py | true |
| dual_rgbt/model.py | true |
| dual_rgbt/data.py | true |
| dual_rgbt/tracker.py | true |
| dual_rgbt/losses.py | true |
