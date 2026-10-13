# Dual-RGBT 完整参考 Test 实验审计

- 日期：2026-10-14（Asia/Shanghai）
- 审计对象：C20 参考模型的完整 LasHeR test
- 审计方式：fresh native Codex reviewer；只读；same-family；provisional
- 请求路由：gpt-6-astra / max；实际模型与 effort 无运行时证明，routing=UNATTESTED
- Overall verdict：WARN
- Deterministic evidence verdict：PASS
- Integrity status：WARN（范围与比较主张限制；未发现本次指标造假或归一化操纵）
- 服务器连接、NN forward、优化器更新、baseline 精度重测：均为 0
- 审计期间未安装环境、未写入报告或修改被审计文件。

## 结果

| 项目 | 独立核验值 |
|---|---:|
| Test 序列 | 245 |
| 总帧数 | 220,703 |
| 初始化帧 | 245 |
| 连续在线 trace 行 | 220,458 |
| PR | 73.91623094612851 |
| NPR | 70.39888941394052 |
| SR | 58.587111241795334 |
| 平均服务时间 | 105.94228367596992 ms |
| P95 服务时间 | 163.01994650275444 ms |
| FPS | 9.439101794884401 |

原始报告位置：`reports/full_pipeline/lasher_full_metrics.json:2–17`。

原工具包重新计算的 PR、NPR、SR 曲线分别为 245×51、245×51、245×21，与 `lasher_full_metrics.npz` 逐元素完全一致，最大绝对差均为 0。JSON 全局指标以及全部 245 条逐序列指标也完全一致。此次重算仅评分已有预测，没有运行模型。

## A. Ground Truth Provenance：PASS

1. `scripts/run_tracking.py:30–37` 从数据集各 test 序列的 `init.txt` 读取 GT，并检查 RGB、TIR 与 GT 帧数一致。
2. `scripts/score_tracking.py:35–51` 对 test 使用原工具包 `LasHeR()`，校验全部 245 序列集合、每个预测/保存 GT/工具包 GT 的完整长度及取整后的 GT 相等。
3. 工具包 `rgbt/dataset/lasher_dataset.py:13–16` 从 `rgbt/gt_file/LasHeR/lashertest.txt` 和 `lasher_gt/` 加载数据集标注，不从模型输出生成参考。
4. 本次直接读取 GT 归档 `reports/test_init_gt.tar.gz`。实际 SHA256 为 `8850bda8d7da83052435f513b533f3ced61fa418e601442793412a088ec0e39b`，与 `reports/test_gt_identity.json:7` 一致。
5. 全部 245 个保存 GT 文件在 float32 下与该原始归档逐元素相等；全部 245 个取整 GT 与工具包 GT 相等。全部 245 个预测首帧与原始 GT 首帧相等。
6. Test GT 只用于初始化与离线评分。`run_tracking.py:83–101` 仅把 `boxes[0]` 传入 initialize，后续为 `tracker.track(image)`。`dual_rgbt/data.py:15–19` 的 read_pair 只读取图像。`dual_rgbt/tracker.py:31–61` 从前一预测生成搜索区域、更新模板与历史；`dual_rgbt/model.py:182–195` 的 predict 不接收 GT。
7. C20 训练名单及全882计划名单与 test 名单均无交集；二者与三个候选共同使用的 97 条 dev 名单也均无交集。

限制：本次审计没有重新连接数据服务器。数据来源验证依赖本地保存的原始 GT 归档、其既有 SHA 及原工具包标注的一致性。

## B. Score Normalization：PASS，附工具包口径限定

- `scripts/score_tracking.py:58–74` 调用原 PR_LasHeR、NPR、SR_LasHeR，按序列等权汇总；没有使用预测输出的最大值、均值或最小值作为报告指标的归一化分母。
- `rgbt/utils.py:118–159` 的 NPR 使用 GT 宽高归一化中心距离。PR 取 20 像素阈值，NPR 取 0.20 阈值，SR 为 21 个成功率阈值的平均值，再乘 100。
- `rgbt/dataset/basedataset.py:19–24` 对 GT 与预测读取结果进行 round(0)。
- `rgbt/metrics/metrics.py:253、296、338` 将首帧预测设成 GT；首帧仍存在于各序列分母中，并非真正删除首帧。
- 必须保留并披露原工具包的特殊处理：任意 GT 坐标≤0时，PR/NPR 将该帧误差设为 -1，随后计入非负阈值的成功数量；SR 将该帧设为失败，分母仍包含它。对应代码：`metrics.py:266–272、309–315、351–357`。
- 本次完整 test 中，52 条序列共 1,260 帧触发上述 GT 条件。本次严格保持工具包实现，没有替换过滤规则，也没有另造“修正分数”。

因此，结果可以称“按用户提供的原 RGBT 工具包口径所得指标”；不能改称“仅在全部有效 GT 帧上计算”或“剔除无效帧后的指标”。

## C. Result Existence 与一致性：PASS

- 两分片全部 config、COMPLETE、245 个 predictions、245 个 groundtruth、245 个 traces、frame_seconds.npy 与 progress.json 实际存在并读取。
- shard0：123 序列，107,898 帧，107,775 条在线 trace。
- shard1：122 序列，112,805 帧，112,683 条在线 trace。
- 两分片名单互斥，其并集精确等于工具包 245 条 test 名单；分别为完整排序名单的偶数/奇数位置。
- 两 config 均为 `split=test`、`smoke_frames=0`、`shards=2`，COMPLETE 均为 `smoke_only=false`。
- 每个预测/GT 文件均为完整 T×4；预测有限且宽高为正。每条 trace 均覆盖连续 frame=1…T−1，无缺行、重号或前缀冒充完整序列。
- 每条 trace 的耗时、动作、模态及 token 汇总与 progress.json 相符；所有时间数组逐序列精确对应初始化时间加该序列全部在线时间。
- 完整曲线、逐序列指标、全局指标、平均耗时、P95 与 FPS 重算完全相同。

### 冻结与权重绑定

- 当前 `frozen_test_selection.json` 与 `reference_frozen_test_selection.json` 的实际 SHA 均为 `84b67409e62593ac5865425c640466f11045a94feb04a1e623e5c259ef410480`，与 startup 和 selection summary 记录一致。
- 两分片 config、指标报告、冻结文件与 summary 中的 checkpoint SHA 均为 `94be6239f7f6864cc193fff7c21d67c9479924c9432039c6ff16cd0d1223f15f`，cost 均为 0.05。
- 冻结文件包含同一 checkpoint 的完整 97 条/54,646 帧 dev 候选，cost=0、0.05、0.15。按照冻结规则 SR→PR→NPR→较低平均耗时重新选择，赢家确为 0.05。
- `complete_pipeline.py:154–167` 在 test 启动前写入 dev 选择；`run_tracking.py:69–72` 对非 smoke test 强制检查 checkpoint SHA 和 cost。
- 全882计划记录的 17 份源码 SHA 与本地实际源码全部相符。

限制：本地不存在 `runs/stage_c_complete_20261013/latest.pth`；本次没有重新读取权重本体。上述 checkpoint SHA 是对历史启动、配置与冻结记录的一致性验证，不应表述为“本次重新哈希实际权重文件”。

### 真实退出码

- `reports/reference_full_test_native.json:27–29、58–60` 保存两个 test task 的 COMPLETE / exit_code=0。
- 实际 config/COMPLETE 文件 SHA 与 native 凭据中记录的四个 SHA 全部相等；PID、启动时间、命令与 startup 凭据对应。
- `scripts/run_task.py:27–32` 将实际 subprocess.returncode 写入状态并以它退出。
- 原本地 controller 的 ExitCode 仍为 null；`reference_full_test_native.json:70–72` 明确写了“do not infer exit0”。所审计证据未把该 null 冒写成 0。
- `full_pipeline_status.json:2` 的 COMPLETE_PIPELINE 表明流程已写出完成状态；这不能替代未知的 controller 原生退出码。

## D. Dead-code Detection：PASS（限本次报告指标）

- `complete_pipeline.py:86–89、176–179` 实际计分路径调用 score_tracking 并读取结果。
- `score_tracking.py:58–78` 实际调用 PR_LasHeR、NPR、SR_LasHeR，并把曲线及报告写出。
- 本次独立重算完整输出完全一致，证明报告数值可由实际保存预测与相同 metric 路径得到。
- 工具包其他数据集用 MPR/MSR 等类不构成本次未执行结果的依据。
- `dual_rgbt/tracker.py:1` 的 “not yet benchmarked” 注释已落后于当前保存结果；它不推翻实际存在的完整结果。

## E. Scope：WARN

### 可支持的范围

这是一个固定 checkpoint、固定 seed2027、选定 cost0.05 的 C20 参考模型，在完整 LasHeR test 245 序列/220,703 帧上的一次评测。

C20 配置实际为 20 条训练序列×2轮、label_stride5、cost0.05。证据：
- `complete_pipeline.py:122–132`；
- `stage_c_complete_20261013/config.json:4–9`；
- `stage_c_complete_20261013/COMPLETE.json:1`；
- 本次读取 trajectory_steps.csv，核实 3,504 行、20 个序列、轮次1/2、step连续0…3503，监督帧均为5的倍数；
- 保存序列长度计算得到 17,708 次 track，与 COMPLETE 一致。

### 不能扩大的范围

- 不能称这份结果来自“全882条完成闭环训练的最终模型”。全882计划的 917,106 次 track、182,182 次更新、1,764 次初始化属于另一个预先固定的训练预算。
- C20 名单恰为全882计划名单的前20条，但这不使 C20 结果成为全882结果。
- A/B 曾涉及882条随机样本训练，也不能替代 C 阶段完整882条闭环训练的证据。`docs/ALL_TRAIN_PROTOCOL.md:9–12、27–33` 明确区分了这些范围。
- 本地无 train manifest，因此本次未独立重新计算全882预算；没有对当前全882队列或训练状态作服务器查询。
- 只有一个种子及一次选定配置，不能据此声称跨种子稳健或多个数据集普适。

### 速度范围

平均 105.9423ms、P95 163.0199ms、9.4391fps 为保存的本方法服务时间，含图像解码和序列初始化，排除模型加载；计时边界也排除后续结果/trace写盘及离线计分。

FPS 定义为总帧数除以两个分片服务时间之和，不是两卡并行墙钟吞吐。证据：`run_tracking.py:84–109、124–132` 与 `score_tracking.py:72–74`。

所审计材料没有同设备、同边界的 baseline 配对延迟，故不支持“比 baseline 更快”“已证明真实效率优势”或“已满足完整在线系统延迟优势”的主张。

## F. Evaluation Type：real_gt

基于真实数据集 GT 的完整保存预测评测；不是 synthetic_proxy、自监督一致性或仿真评测。本次审计重算属于已有真实预测的 CPU 离线计分。

## Claim Impact

| 主张 | 结论 |
|---|---|
| C20 参考模型完成245序列/220,703帧test，指标73.9162/70.3989/58.5871 | supported |
| 这份结果是全882闭环训练最终模型的结果 | unsupported |
| 达到PR≥78、NPR≥72、SR≥62 | unsupported；三项均未达到 |
| 相比用户提供baseline 72.74/69.16/57.84，高1.1762/1.2389/0.7471个百分点 | needs qualifier；仅相对用户提供的数值，不是本次baseline重测或完整比较协议复核 |
| 本方法保存服务时间105.9423ms、9.4391fps | supported，须保留计时边界及服务时间定义 |
| 已证明同设备baseline速度优势 | unsupported |
| 已证明三个核心模块的独立贡献 | unsupported；本次没有此类消融或因果证据 |
| 两test task的原生exit0 | supported，由保存的native凭据与文件SHA支持 |
| 原本地controller的原生exit0 | unsupported；实际字段为null |

距目标还差 PR 4.0837691、NPR 1.6011106、SR 3.4128888 个百分点。

## 报告层面的处理事项

1. 对外名称明确保留“C20参考模型完整test”。
2. 保留原始冻结文件和本次原工具包口径；不能静默修改无效GT处理规则。
3. `reference_test_selection_summary.json:66` 的 “No test result yet” 是冻结时历史状态，应作为历史快照解释，不能当作当前完整test状态。
4. controller ExitCode 保留 null，不用任务exit0或COMPLETE_PIPELINE代填。
5. 速度优势、全882最终结果及模块独立贡献继续标为未证实；本审计不从test序列错误提出调参建议。

## 实际检查命令与退出状态

所有 Python 检查均通过 PowerShell 单引号 here-string 提供 stdin，调用同一个已有解释器：

`@'<inline read-only Python program>'@ | & 'C:\Users\gb\AppData\Local\uv\cache\archive-v0\7oLWbeZ6v0oUodecKf4d1\Scripts\python.exe' -B -`

- 环境/原工具包导入：tool chunk `bfd016`，exit0；Python3.12.12、numpy1.26.4，torch未导入。
- 原GT归档检查：`80c50d`，exit0；245个文件，归档SHA相符。
- 完整GT/预测/trace/冻结/源码SHA/原工具包曲线重算：native session90437，启动chunk `99ba82`，完成chunk `42cd0e`，真实exit0；核心CPU程序耗时33.677195秒。
- 输入文件及目录SHA核验：`dc6a76`，exit0。
- C20训练CSV与保存长度预算核验：`5b339e`，exit0。
- train/dev交集、候选范围、baseline数值差与目标差：`cb133a`，exit0。
- 实际参与计分的245个工具包GT文件摘要：`7dac7d`，exit0。
- 一次环境路径探索命令 `b81882` 返回exit1：向rg误传PowerShell的-ErrorAction，并确认本地data目录不存在；该失败没有执行NN或计分，没有当作成功记录。随后使用已存在的明确环境路径完成检查。

完整源码读取及上述 stdin 程序保存在本次 reviewer 原生工具轨迹中；应随本报告保存该完整轨迹，不应声称已由 reviewer 写入磁盘。

## SHA摘要

- Metrics JSON：b810a0c890e298502c3cfec1d11c378c678ade85098ef98fc9bb177826c22a11
- Metrics NPZ：aedaa8caf4164d1499349560a8091231e9c8164d13fa40a7c184115aa5148482
- Frozen selection：84b67409e62593ac5865425c640466f11045a94feb04a1e623e5c259ef410480
- 原GT归档：8850bda8d7da83052435f513b533f3ced61fa418e601442793412a088ec0e39b
- shard0全部373文件目录manifest：fba1f660ade0e18daae617605121add43079e45569337b6cde8748c6f4f727fc
- shard1全部370文件目录manifest：10477bd1a8117bf67653c0193ef8510ba15b1562c224ad615598dd5fec8adf20
- 本次实际计分245个工具包GT目录manifest：fbd87fa6ec07206a16f31d96564618a2cde249f9c05c538eb47d8211138d3b11

目录manifest算法：相对POSIX路径排序，每项包含path、bytes、sha256；以 `json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':'))` 的UTF-8字节计算SHA256。完整文件SHA见机器可读JSON。
