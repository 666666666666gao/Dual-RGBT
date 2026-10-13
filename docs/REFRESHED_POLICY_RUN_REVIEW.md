# A7 执行器上的完整 B 训练源码审查

日期：2026-10-13。结论：**PASS；本次完整 B 启动方案的源码阻断项 0。** 该结论只覆盖源码与命令的一致性，属于 same-family/provisional，不是实际运行、精度或延迟验收。

```yaml
review_scope: source-only, fresh Stage-B training from selected A7 executor
review_context: separate delegated reviewer
review_independence: same-family
acceptance_status: provisional
requested_model: gpt-6-astra
requested_reasoning_effort: max
requested_metadata_source: root agent confirmed explicit invocation parameters in follow-up
actual_model_id: UNAVAILABLE
actual_reasoning_effort: UNAVAILABLE
actual_metadata_note: no independently attested runtime model or effort identifier exposed
reviewed_head: 00d3e7732fa249d826ca84f26beabf0d6f058fef
source_verdict: PASS
wrapper_source_verdict: PASS
open_blocking_issues: 0
new_source_defects: 0
reviewed_epochs: 5
expected_policy_updates: 6250
runtime_acceptance: NOT_VERIFIED_BY_THIS_REVIEW
ssh_gpu_model_training_execution: NOT_RUN
```

审阅期间执行代理将原拟议的三轮改为用户要求的完整五轮。本报告最终只覆盖下列五轮命令。只读取本地源码、既有文档与已收回的 JSON 记录，唯一写入文件为本报告；未连接服务器、读取 GPU、载入模型或 checkpoint tensor、启动训练。

## 精确启动对象

执行目录须为服务器 `/home/gaob/Dual-RGBT`，SSH 别名为 `2025`：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 conda-envs/dual_rgbt/bin/python scripts/train_policy.py --executor runs/stage_a_spatial_full_20261013/best.pth --run stage_b_refreshed_20261013 --samples 10000 --epochs 5 --batch-size 8 --workers 4 --seed 2027
```

这是从选中 A7 状态开始的一项全新 B 训练。`10000 / 8 = 1250` 次更新/轮，五轮合计 50000 个抽样训练样本、6250 次更新；样本是确定性抽样，不表示 50000 个唯一帧。完成时预期 checkpoint 为 epoch=5、step=6250，`COMPLETE.json` 为 epochs=5、steps=6250，CSV 有6250行，行内 step 为0至6249。以上均为源码推导的预期，尚不是该新 run 的运行结果。

本地 `reports/stage_a_full/COMPLETE.json` 记录 first_epoch=3、last_epoch=7、steps=7250、当前 run 新增6250更新；`best.json` 记录 best epoch=7、step=7250、dev_loss=0.9196168765202165。它们支持执行代理提供的 A 已完成状态，但本审查没有核验服务器当前进程、远端文件哈希或 best.pth 内部 tensor/元数据。历史 `HANDOFF.md` 仍记载扩大 A/B 启动时刻，不能把其中当时的运行状态当成本次最新观测。

旧 `reports/stage_b_diagnostic/config.json` 指向 `runs/stage_a_spatial_20261013/best.pth`；其对应本地 best 记录为 A2/1000。旧 B 两轮1000更新仅属旧执行器诊断。新命令没有旧 B checkpoint 或 M0 checkpoint 输入，不能继承它们的策略权重、优化器或收益标签。短 sanity 产生的权重也不作为这项正式训练的初始化。

## 源码核对

| 项目 | 当前源码与结论 |
|---|---|
| 初始化与优化器 | `train_policy.py:48–53` 先构造模型，再严格载入所指定 A 文件的完整 `net`，随后新建 AdamW(lr=1e-4, weight_decay=1e-4)。没有优化器恢复或旧 B 加载路径。`train_executor.py:44–47` 在 A 中冻结 preview/quality/gain，因此 B 初始化继承 A 保存的这些策略状态；state_dict 不携带 requires_grad，B 新模型的策略参数正常可训练。 |
| 执行器冻结 | `train_policy.py:50–53,66` 冻结 backbone/decode/box_head，只把仍可训练的参数交给 AdamW，并每批调用 eval 固定执行器 dropout/BN 状态。当前 model 没有旧报告中的独立 corrector；空间融合完全属于已冻结的 decode/box_head。eval 不关闭策略 autograd。 |
| 新鲜、同状态标签 | `train_policy.py:68–83` 每批在 no_grad 中使用当前 A7 执行器与更新前 preview，先算两个 primary，再各执行四种修正；目标为同一输入/primary 下的 main_loss−action_loss，保留负值。`84–107` 重新求可微 preview 后才更新策略。两次 preview 之间没有参数更新，其网络也没有 dropout/BN；没有加载缓存收益标签。 |
| 决策方向和动作顺序 | quality 目标为负定位损失，在线 argmax 方向正确。`model.py:105–120,150–165` 的四项候选顺序与训练 action=1..4 一致，STOP 收益固定0。两模态收益目标各为 `[B,4]`，对应预测去掉 STOP 后的四列。 |
| 梯度与数值 | `train_policy.py:84–107` 的 quality/gain 回归和 preview 空间交叉熵位于 no_grad 外；每次 backward 后检查三个模块各有非零梯度，并检查裁剪前总范数有限，才执行 optimizer.step。候选 argmax 本身不可微，preview 的空间损失提供显式监督。CENTER head 将 score 限于 `[1e-4,1−1e-4]`，当前无窗 B 熵路径没有已知 log(0) 问题。 |
| 空间目标与定位损失 | `data.py:23–28` 使目标中心为 `.5−u`，u∈[-.375,.375]，因此8×8标签有效。`losses.py:7–21` 将预测 cxcywh 和目标 xywh 转成相同坐标的 xyxy，所有动作使用同一 GIoU/L1 定位损失，没有用分类置信度替代真实定位收益。 |
| 数据与随机性 | `train_policy.py:63–64` 只取 manifest 的 train；`data.py:52–64` 使用 seed+index 的局部 RandomState，按真实先前帧取模板。五轮各自 seed 基数为2027、12027、22027、32027、42027，worker数不会改变 index 对应的抽样定义。`prepare_data.py:11–30` 从官方训练目录建立 train/dev，test 只取名称校验分离，不读取其 GT 生成训练标签。服务器实际 manifest 未由本审查重验。 |
| 输出隔离 | `train_policy.py:54–55` 新建唯一 run 且 exist_ok=False；已存在的同名 run 会直接报错，不会覆盖或自动续训。`117–119` 每轮保存 latest，最后写 COMPLETE。只有参数实际更新并记录当前行后 step 才递增。 |
| 设备与启动依赖 | 指定 PCI_BUS_ID 与可见设备1，源码接受单设备字符串1，`.cuda()` 使用该进程唯一可见设备。命令与项目解释器路径一致。`model.py:34–40` 仍先读取保留的原始 `/home/gaob/CADTrack_ep0014.pth.tar` 再加载 A7，故该原文件也是启动依赖；当前任务已要求保留它。源码不证明物理卡实时空闲或单卡任务数。 |

## 已知阶段边界与运行验收

**没有需先修复的 B 源码阻断项。** B 当前仍是 GT 周围抖动裁剪、零 history、无 Hann 的监督起点（`train_policy.py:69–70,78–84`）。在线 Tracker 使用预测框、非零 history 和 Hann；这项五轮 B 不能代替 C 的真实闭环监督刷新。已有 `ONLINE_WINDOW_REVIEW.md` 与当前计划明确记录了该边界，不能把早期 `POLICY_CODE_REVIEW.md` 对旧架构的 corrector/置信度描述套用到当前空间版实现。

执行代理给出的后续范围为 C 固定20条训练序列两轮、完整97条 dev 校准、完整245条 test 评价。本报告没有审阅或验收这些后续任务的具体新启动命令，也没有把它们描述为完成。不得使用 test 标注选择权重、预算或阈值；正式 PR/NPR/SR、完整在线平均/P95延迟与 FPS 仍须来自后续实际评价。

本地历史 M0 记录 `reports/resume_and_policy_m0_execution.json` 与 `reports/policy_m0_check.json` 报告两次 B 更新、250个执行器 state tensor 不变、32帧闭环和两次严格加载副本的预测逐位一致；其 config 明确使用 A2、batch4、workers2。它证明的是先前执行记录，不是本次 A7、batch8 的运行结果。依据 AGENTS.md，新实验仍需实际训练数据前向、反向、优化器更新与重载检查；本报告未执行、未替代该步骤。核验新 B 时，应将冻结 tensor 比对对象明确设为本命令的 A7 executor，确认三个策略模块发生实际变化，并如实保留执行结果。sanity 权重不得成为正式 B 输入。

实际启动时须由执行代理确认指定源文件已部署、A7文件身份正确、目标 run 尚不存在、GPU1满足每卡最多一个训练任务；这些是当前明确的运行约束，不是要求增加新的 fallback、兼容层或训练代码分支。

## 五轮启动脚本补充复核

执行代理后续明确确认审阅任务的请求路由参数为 `gpt-6-astra` / `max`，本报告据此更正 requested 元数据；actual 仍为 UNAVAILABLE，不将请求参数视为独立核实的实际运行路由。

逐行读取新增 `scripts/run_stage_b_full.sh` 后，**wrapper 源码结论为 PASS，阻断项0**。该复核只涉及本地脚本，不查询已启动任务。

- `:3–6` 进入 `/home/gaob/Dual-RGBT`，显式导出 PCI_BUS_ID、GPU1 和无缓冲输出，符合上述命令及项目限制。
- `:7–9` 前台调用项目 conda Python，executor、run、10000样本、五轮、batch8、workers4、seed2027 与本报告精确命令一致；单次脚本执行只含一次训练调用。
- `:10` 紧接训练命令保存 `$?`，中间没有其他命令、管道或后台进程覆盖训练退出状态。脚本使用 `set -u`，没有让非零训练退出码在记录前触发 `set -e` 中断。
- `:11–12` 将保存的训练退出码写入 `runs/stage_b_refreshed_20261013.exit_code.txt`，随后以同一个状态退出；不把启动成功、日志出现或 COMPLETE 文件存在冒充原生进程 exit0。该退出码文件只在前台训练命令返回后产生，训练尚未结束时不能期待最终码。
- 本地实际脚本为458字节、12个LF、0个CR，开头为无BOM的 `#!/`，末尾为LF；没有 Windows 换行或BOM破坏 shebang/续行的现存问题。未执行脚本或 `bash -n`。

执行代理在此次补充任务中报告：A7/batch8 的两次实际更新及32帧检查已 PASS，正式 B 已单次启动，PID3616703。本审查未读取对应的新运行产物或远程状态，以上仅保留明确来源的后续通知；不追溯改变初次源码审查的未运行范围，也不声明五轮训练已完成。后续以这项已启动 run 的真实终态产物核验6250次更新与退出码。

## 审查证据固定

审查开始的 HEAD 为上列提交；四个核心文件当时无本地 diff：

| 文件 | SHA-256 |
|---|---|
| `scripts/train_policy.py` | `2fa070434680704c984d9a1b5b992973103971c47d692a6fcbea734571922913` |
| `dual_rgbt/model.py` | `434da9f818158c16bd7563e93ef44c65b2b726d881f35f94284fb6e8f9966e50` |
| `dual_rgbt/data.py` | `858e41c8e5218c7f5323a6e18cec00997a48d0323101384b71a73d61ea0280cb` |
| `dual_rgbt/losses.py` | `d3b941610e666a236036fd67079af15d5003e366a56adf8acab8b34d8ac584b1` |
| `scripts/run_stage_b_full.sh`（补充复核） | `0ceb33877311359894474b8bca1c4990fdb145104036a4009d3d2626a09f9536` |

逐行审阅了任务指定的四个源码文件、两份历史 policy review 与 HANDOFF，并核对 AGENTS、EXPERIMENT_PLAN、ONLINE_WINDOW_REVIEW、A训练器、数据清单生成器、Tracker 和 CENTER head 的有关调用。仓库 `git diff --check` 退出0。本地仅尝试以 Python 标准库解析上述四文件 AST，但 Python 启动器以 `No pyvenv.cfg file` 退出1，未进行解析；因此不声明 AST 检查通过。未运行任何神经网络或训练验证。
