# B→C→完整 dev→完整 test 串联源码审查

日期：2026-10-13。结论：**SOURCE PASS；未关闭源码阻断项 0。** 当前串联实现满足本次限定的完整训练续接和评价流程。该结论属于 same-family/provisional 的源码审查；本审查没有启动控制器、查询远端 B、运行网络或取得正式指标，完整研究目标仍未验收。

末次定向复核覆盖 `collect_training`、其完成 B/C 后的两处调用，以及 C/dev/test 首次观察改为60/135/280分钟。该增量 **PASS，新增阻断项0**；当前完整控制器 SHA256 为 `56ed5936b1d3c59f37741963c88a7b66201772f1c04912034c5e086cc2bbb58f`。

```yaml
review_scope: source-only, full pipeline continuation and frozen test inputs
review_context: separate reviewer reading current local sources and returned evidence files
review_independence: same-family
acceptance_status: provisional
requested_model: gpt-6-astra
requested_reasoning_effort: max
actual_model_id: UNAVAILABLE
actual_reasoning_effort: UNAVAILABLE
actual_metadata_note: no independently attested runtime model or effort identifiers exposed
reviewed_base: 00d3e7732fa249d826ca84f26beabf0d6f058fef
source_verdict: PASS
open_source_blocking_issues: 0
controller_execution: NOT_RUN_BY_THIS_REVIEW
ssh_gpu_nn_or_metric_execution: NOT_RUN_BY_THIS_REVIEW
full_pipeline_runtime_acceptance: PENDING
goal_acceptance: UNMET
```

继续采用 `code-review-excellence` 的具体正确性审查方式。直接读取新增 `scripts/complete_pipeline.py`、`scripts/run_task.py`、当前 `run_tracking.py` 和 `score_tracking.py`，并核对实际被调用的 `check_policy_checkpoint.py`、`train_trajectory_policy.py`、`train_policy.py`、`run_stage_b_full.sh`。同时读取已收回的 GT、C sanity、B 启动报告和既有完整评价/C sanity 审查；末次定向复核还读取 `reports/profile_tracking_default.json` 并重算已有逐帧时间。仅维护本报告；未修改源代码、旧审查或任何结果文件。

## 顺序、来源与权重隔离

控制器不重新启动 B。它只等待固定 `stage_b_refreshed_20261013` 的 exit 文件和 COMPLETE，要求实际 exit=0、COMPLETE恰为 `epochs=5, steps=6250`，然后使用该 run 的 `latest.pth`。原 B shell 先完成训练子进程，再保存实际退出码；B 训练每轮先保存 checkpoint，全部循环后才写 COMPLETE。

随后按顺序在 GPU0 启动：B checkpoint 检查、以完成 B 为输入的新 C sanity、以同一完成 B 为输入的正式 C。每一步均等待前一步任务状态终结且 exit=0；没有把已有短 sanity 的 PASS 直接替代这次完成 B 后的 sanity。

`check_policy_checkpoint.py` 检查 B 确实有更新、执行器 state 与 A7 best 逐项相同、preview/quality/gain 相对 A7 各有变化，严格载入两实例并比较输出，还运行真实训练序列的32帧因果轨迹与 token trace。它没有自行验证 checkpoint 的 epoch=5/step=6250；该阶段完成约束由前述同一 B run 的 COMPLETE/退出记录提供。当前新目录与顺序保存路径没有证据显示会混入另一 checkpoint，不要求增加无依据的恢复兼容层。

C sanity 写入 `stage_c_complete_input_sanity_20261013`；正式 C 写入不同的 `stage_c_complete_20261013`，正式命令没有 `--sanity`，并重新严格载入 B、创建新 AdamW。正式起点没有指向 sanity 的两步权重。显式参数为20条 train、2轮、stride=5、cost=.05、seed=2027；源码仍逐帧推进真实预测，仅 train GT 用于初始化及监督。控制器在 C 实际 exit=0 后还要求 COMPLETE 的3504次更新及17708次非初始化 track，与既有数据计划一致。C 的 `latest.pth` 之后只读，dev/test 阶段没有 optimizer、续训或再次保存它的路径。

原 CADTrack 权重、A7 best、B latest、C sanity latest、C formal latest 均使用各自既有或新建路径。`run_task` 的任务目录、训练 run 和跟踪 run 都使用 `exist_ok=False`；重复执行会停止，不会覆盖已存在任务或结果。原 B 的活跃状态仅由本地启动报告及委托说明提供，审查者没有远端核验，不能把当前源码审查写成 B 已完成。

## 完整 dev、冻结决定与完整 test

正式 C 完成后，预先声明的三个成本为 **0、0.05、0.15**。前两个各占一张 GPU 同时做完整97条 dev；该组完成、回收并评分后才在 GPU0运行第三个。三者都读取同一正式 C latest，均无 smoke 或序列截断参数。评分器强制97条、完整分片、每条 GT/预测长度以及完整计时帧数。

选择严格依据 dev 结果：先最大 SR，再 PR、NPR，最后较低平均延迟。这个排序在读取 test 结果之前已经写在源码中。三个 dev 结果和选择标准写入本地 `reports/frozen_test_selection.json`，同时写入所选 checkpoint 的真实 SHA256 和成本，再上传到远端相同相对路径；上传失败会抛错，后面的 test 不会启动。

当前 `run_tracking.py:69–72` 对正式 test 在模型构造之前读取冻结文件，断言实际 checkpoint SHA256 和 cost 完全相等。两张卡分别使用 shard=0/1、shards=2，所选成本相同，不进行 test 预算搜索。`score_tracking.py` 还要求所有 test 分片同一 SHA/cost、完整且不重叠的245个 canonical 名称、预测和随附 GT 与 canonical 行数一致、随附 GT 取整后逐行一致，以及每分片与合并计时帧数准确。

因此不会把半个 test、两个不同权重的分片或 smoke 当完整结果。指标仍直接调用原 toolkit，保持 float32+round(0)、PR/NPR第20列、SR21阈值严格大于比较及逐序列等权；本次没有改动 `score_tracking.py` 或原 toolkit。冻结文件在 test 之前生成，控制器没有据 test 分数回到 dev、改成本、选权重或重新训练的循环。

## 已收回证据的边界

审查者实际读取并核对 `reports/test_gt_identity.json`：245条、220703帧、245条 raw-float32 首框相等、全部帧按 toolkit 取整后相等、issues为空；其中还记录了服务器 GT 归档 SHA256。它补充了上一份完整评价审查时尚未提供的全量 GT 一致性凭据。**本审查依赖已收回报告，没有重新读取服务器 GT 或独立重放归档审计。** 图像配对和图像/GT长度仍由实际 runner 的逐序列断言检查；该 GT 报告本身不等于已完成跟踪。

`reports/c_a7_sanity.json` 明确记录 PASS、2次更新、10次 track、257个冻结 state 项及保存重载输出相等。这是此前短检查的已收回证据；报告没有证明正式 C 完成，也不证明它的输入已经是最终五轮 B。控制器会在最终 B 完成后对该输入重新执行 sanity，因此没有错误地复用旧短检查权重或跳过最终输入检查。

原 `COMPLETE_EVALUATION_REVIEW.md` 的计时边界继续成立：外层逐帧时间含读取/解码、初始化、预览/路由、裁剪传输、主/辅助网络、定位和模板更新；模型装载明确排除。合并 FPS 为总帧数除以分片服务时间之和，不是双卡墙钟吞吐率；P95来自全部逐帧时间合并。

## 回收目录与失败处理

新增 `collect_training(run)` 仅在 B 的 exit=0及五轮6250步 COMPLETE断言通过之后，以及正式 C 的 exit=0及3504更新/17708 track断言通过之后调用。它先以 `exist_ok=False` 创建本地 `reports/full_pipeline/<训练run>`，再将同一远端训练目录的 `*.json` 和 `*.csv` 两组文件复制进该已存在目录。B 源码实际产生 config/COMPLETE及policy_steps.csv；C 源码实际产生 config/COMPLETE及trajectory_steps.csv，成功完成路径上两组通配符都有对应文件。没有 `-r` 或权重匹配，不会回收 `latest.pth`，也不会修改远端文件。SCP失败仍立即终止控制器。

训练回收使用 `stage_b_refreshed_20261013` 和 `stage_c_complete_20261013`；后续跟踪使用 `dev_complete_*`/`test_complete_*`，目录名无重合。审查者末次只读本地检查发现 `reports/full_pipeline` 和这7个目标目录均尚不存在。首个训练回收会创建共享父目录；后续训练目标仍是新目录，跟踪目标也仍满足不存在断言。

`collect(run)` 保障本地 `reports/full_pipeline` 父目录存在，要求具体跟踪目标 `<run>` 不存在，然后将远端 `/home/gaob/Dual-RGBT/runs/<run>` 整目录 SCP 到这个新目标。因此每份本地跟踪 run 根目录直接包含 config、COMPLETE、predictions、groundtruth、traces、progress和frame_seconds；没有预先创建同名跟踪目标目录而造成重复嵌套。`score` 收到的就是该目标根目录。与训练文件复制到已存在目录的方式相容，最终 test 也先回收两个分片再评分。

预期本地结果位置为：

- `reports/full_pipeline/stage_b_refreshed_20261013/` 和 `stage_c_complete_20261013/`，仅含完成训练的JSON/CSV。
- `reports/full_pipeline/dev_complete_c_cost0p0/`、`dev_complete_c_cost0p05/`、`dev_complete_c_cost0p15/` 及相应 `_metrics.json`/`.npz`。
- `reports/full_pipeline/test_complete_c_shard0/` 和 `test_complete_c_shard1/`。
- `reports/full_pipeline/lasher_full_metrics.json` 及对应 `.npz`，包含完整 PR/NPR/SR、均值/P95/FPS及逐序列指标。

这些路径目前是源码规定的未来输出，不表示文件已经生成。回收范围是 B/C训练JSON/CSV及完整跟踪结果目录，不会复制 B/C权重或删除服务器原件。任务的 `status.json`、`run.log`保留在远端 `runs/pipeline_tasks/<task>`；启动器日志位于远端 reports，控制器本地状态保存最后观测和启动 PID。这些位于训练run以外的任务凭据不会由当前两个回收函数自动下载，不应把本地目录描述成包含全部训练任务日志的归档。

`run_task` 使用参数列表运行子进程，固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 和单卡 `CUDA_VISIBLE_DEVICES`，只允许0或1；正常子进程终止后保存实际 returncode，映射 COMPLETE/FAILED，再以同一码退出。控制器读取 FAILED 后先保存该观测再断言退出；SSH/SCP/评分调用也使用 `check=True`。没有重试、自动恢复、重新启动 B、覆盖目录或调度其他 GPU 的分支。控制器发生回收/评分等自身错误时，本地状态可能仍停在最后阶段，不能只凭状态字符串判断活跃；实际退出和 stderr 仍需由启动端收取。这是明确的失败即停流程，不建议为假设性故障增加 fallback。

## 观察节点与运行前提

B 首次观察时间由启动参数 `--b-not-before`提供，源码不会自动从 B 回执重算。实际本地 B 启动记录为 epoch **1791882298.7479835**，所以“启动45分钟后首次查询”要求传入值至少为 **1791884998.7479835**。之后每次未完成观察都延后240秒。

末次变更后，每个短检查的首次观察仍为60秒，正式 C为60分钟，每组首个完整 dev任务为135分钟，test首个任务为280分钟；未完成后的观察间隔仍为240秒。同组第二张卡的首次查询发生在第一张卡等待/回收之后，不早于同组既定首个节点。`wait_until` 的最多60秒睡眠仅为本地睡眠，不会每60秒查询 SSH/GPU。启动前仅查询所指定卡的显存用量并要求<500MiB，没有查询温度或功率。

节点修订的已收回依据为 `reports/profile_tracking_default.json`，它记录一条 dev序列的64帧执行剖析，线程数为 Torch32/OpenCV64。审查者从实际JSON重新计算 frame9–64共56个 warm帧，得到平均解码55.1128103055ms、平均 track109.4103916403ms，合计 **0.164523201946秒/帧**，与报告摘要一致。该数值来自64帧剖析中的56个 warm帧，不是64个 warm帧，也没有采用先前16帧冷启动均值。新节点只是较保守的观察时间估计：短序列剖析不保证所有序列、成本和训练监督具有相同速度，C还包含反事实监督及反向开销，不能把节点本身写成实际完成时刻。

这次增量没有设置线程数，没有改变模型、追踪/评分源码或已运行 B。读取现有剖析只为核对预计观察时间；它没有 matched baseline、正式完整延迟或加速结论，不能由0.1645秒/帧推导“已提速”。

实际启动端仍须使用委托指定的现有本地 Python3.12环境（含 NumPy1.26.4/Matplotlib）和原 toolkit路径，部署本次固定源码，并只启动一个控制器实例。源码中评分使用 `sys.executable`，因此会继承控制器解释器；这个环境是否实际被采用不由本次静态审查证明。源内固定远端项目和 conda路径与 AGENTS 一致。

## 验收范围与快照

`COMPLETE_PIPELINE` 只在两个完整 test 分片回收及原 toolkit评分成功之后写入。它表示这一固定串联已结束，不表示 PR≥78、NPR≥72、SR≥62、完整系统提速或三个独立创新贡献已经证明。控制器没有 baseline精度调用，也没有匹配 baseline延迟测量或完整独立消融；研究目标仍须依据真实结果另行判断。当前没有完整 dev/test精度或匹配baseline的加速结果，不能以源码 PASS、两步 sanity、单序列剖析或任务启动代替目标达成。

7份 Python脚本通过现有 Python3.13的 AST解析；控制器4段动态远端代码字符串经占位值格式化后的 AST语法检查通过。末次对新增回收/节点版本重新进行 AST解析，确认 run_task/run_tracking/score_tracking哈希保持前次值，并仅用标准库重算已收回剖析及检查本地目标路径。未导入项目或执行远端代码；没有实跑SCP、GPU或NN。`git diff --check`通过。实际 subprocess、SCP目录回收、CUDA执行和最终评分仍待运行验收。

| 文件 | 审查时 SHA256 |
|---|---|
| `scripts/complete_pipeline.py` | `56ed5936b1d3c59f37741963c88a7b66201772f1c04912034c5e086cc2bbb58f` |
| `scripts/run_task.py` | `90220bfc86bdb8c50bf4d8d45cf58ec10a3c54b2cacfae1f8bf09cac6be66e7f` |
| `scripts/run_tracking.py` | `3c43f74a53d824e98cbeb2e5c0affd7945ea434885a9ea92d9466266133b60b8` |
| `scripts/score_tracking.py` | `22f0cb04de31714f70019f25c8b6b3d48b0384a7aff812d25be367394b3b4ac4` |
| `scripts/check_policy_checkpoint.py` | `eaab05ae60827070a4e9d4a8013a407508acc14e699d7da1749ee8dd9b9100c1` |
| `scripts/train_trajectory_policy.py` | `d45ba30d89c4400f82d6900cd23155f23568c8b03075be5855e898b00b462740` |
| `scripts/train_policy.py` | `2fa070434680704c984d9a1b5b992973103971c47d692a6fcbea734571922913` |
| `reports/test_gt_identity.json` | `79742d78a975eab5ff254560922fccde43bc0836cd69caf8c5d6d2210d02f63b` |
| `reports/c_a7_sanity.json` | `d3ed91e1de78b603f34f03b8cb7c8877c6e6e2d8d29590a5683422e6cea713ba` |
| `reports/launch_stage_b_refreshed.json` | `fb7d6c914124b119ff7db92afe6870eef6d7e3aac988756fd8e0060e44def089` |
| `reports/profile_tracking_default.json` | `6356735a99423b03654cecf6ed3929454cd64003a461fb9b635e430fc8ef22f8` |
