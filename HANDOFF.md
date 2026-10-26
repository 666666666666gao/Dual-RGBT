# Dual-RGBT 交接文档

更新时间：2026-10-26。此文件是唯一交接文档；同步到本地桌面 document/Dual-RGBT交接文档.md、26服务器项目根目录及 GitHub。旧逐次节点以 Git 历史和 reports 原始证据保留，不再追加相互冲突的“当前状态”。

## 当前训练与优先级

用户要求尽快完成训练，并允许每卡多个任务；GPU2/3必须等现有训练进程结束且确认空闲后接入。当前GPU0、1各三项正式训练，共六项，全部从同一原B6250/SOT epoch14来源重新开始，每项完整882条TRAIN×2轮、182182更新/917106次track/1764初始化。新四项真实4步准入与3分钟启动检查均通过；并非六份同配置重复训练。

| GPU | 正式配置 | wrapper→trainer | 实际启动UTC |
|---|---|---|---|
| 0 | joint-quality，K8/cost0.05 | 828655→828658 | 15:01:34 |
| 0 | joint-quality，K8/cost0 | 1170719→1170720 | 18:10:48 |
| 0 | joint-quality，K4/cost0.05 | 1185099→1185117 | 18:18:32 |
| 1 | joint-gainpair，K8/cost0.05 | 1054384→1054385 | 17:06:57 |
| 1 | joint-gainpair，K8/cost0 | 1170862→1170878 | 18:10:49 |
| 1 | joint-gainpair，K4/cost0.05 | 1185311→1185312 | 18:18:33 |

原两项最近17:56:39 UTC节点为13900/3600更新、已记录loss有限；新增四项各在启动检查时150更新、loss有限，执行器/骨干冻结、重载/执行重放通过。真实证据为`JOINT_GAINPAIR_TRAIN_EXECUTION`、`CONCURRENT_TRAIN_WAVE_K8COST0_EXECUTION`、`CONCURRENT_TRAIN_WAVE_K4COST05_EXECUTION`对应20261026 JSON。没有新完整dev/test成绩，目标仍未达到。

GPU2/3各预留一个K4/cost0完整训练。已启动唯一等待器：本地PID48484/SSH21516、原生exec9899，远端等待器1192814。只通过GNUtail每240秒检查旧GPU2 PID1028233、旧GPU3 PID1028232是否退出，不读取其日志/训练指标，不反复建立SSH；实际tail能力/死PID探测已通过后才写READY。两者退出后自动调用`tools/launch_concurrent_training_wave.py --phase reserved23`，逐卡再要求无其他GPU计算进程、显存<500MiB，随后真实4步准入→原B正式训练→3分钟启动检查。它不代表原任务训练成功；任一设备另有任务时不抢占、不自动重试。当前本地队列记录仍为WAITING，见`GPU23_TRAIN_HANDOVER_QUEUE_20261026.json`。

下一唯一六训练观察器：本地PID44296/原生exec93755，登记`CONCURRENT_SIX_TRAIN_NODE_PLAN_20261026.json`，到1793040513.2902958（18:48:33 UTC）一次检查六项正式任务，评估持续并发吞吐。此前不查进度/PID/句柄，不新建重复观察器；原18811/65517/68247/6046已结束，不再加入。GPU23等待器9899独立保留，未结束不要重启/重复加入。

最初单任务实测速率约1.2～1.3更新/秒，完整剩余训练为数十小时；新并发的全程速度与完成时间待稳定节点重估。短测量只能支持受控放量，不能声称完整C已经加速或训练成功。不读取温度/功率，不中途复制可能写入的active latest.pth、不热改哈希绑定源码。

## 目标、来源及运行位置

当前接续目标优先LasHeR：PR≥78、NPR≥72、SR≥62；旧严格规划78.74/75.16/63.84保留为历史，均未达到。还须完整RGBT210、RGBT234测试。

用户baseline为72.74/69.16/57.84，沿用、不重测精度。初始化始终为**去掉CADTrack论文三个模块、SOT来源backbone、epoch14**，不是完整CADTrack或DropMAE。

| 项目 | 当前位置 |
|---|---|
| SSH | `2026`，26服务器 |
| 项目/代码/权重/输出 | `/data2/gb/Dual-RGBT` |
| Python/conda环境 | `/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python` |
| 本地源码 | `C:/Users/gb/projects/Dual-RGBT` |
| GitHub | https://github.com/666666666666gao/Dual-RGBT |
| 预训练 | `checkpoints/CADTrack_ep0014.pth.tar`，SHA256 cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731 |

25服务器原活动项目已核验归档到`/data2/gaob/Dual-RGBT_archive/retired_server25_project_20261026`，原`/home/gaob/Dual-RGBT`已不存在；不再启动25旧队列。共享数据与外部原始预训练保留。26环境Python3.7.16/Torch1.12.1+cu113/timm0.9.12/NumPy1.21.5，环境迁移、真实CUDA更新/重载、11帧tracker和独立文档执行验证均通过。

环境证据：`reports/SERVER26_ENVIRONMENT_ACCEPTANCE_20261026.json`、`SERVER26_ENVIRONMENT_RELOCATION_20261026.json`、`SERVER26_GPU_KERNEL_WITNESS_20261026.json`；provider ledger `.aris/compute/ssh.md`。新训练另有真实反向更新准入，不能仅凭import成功启动。

## 数据与评分协议

| 数据 | 26项目接入位置与规模 |
|---|---|
| LasHeR TRAIN/dev | `data/LasHeR/train`→`/data/wangwj/dataset/LasHeR0428/trainingset`，882TRAIN＋97dev；dev54646帧 |
| LasHeR test | `data/LasHeR/test`内245条链接→`/data/wangwj/dataset/LasHeR0327/TestingSet/testingset`，220703帧 |
| RGBT210 | `data/RGBT210`→项目`data/datasets/RGBT210testingset`，从25迁入，210条/104706帧 |
| RGBT234 | `data/RGBT234`→`/data/wangwj/dataset/RGBT234/RGB_T234`，234条/116650帧 |

26的/data、/data1、/data2已查找：210需迁入；LasHeR/234复用核验后的共享数据，不修改共享原件。新清单位于`data/benchmarks/server26/`。训练`data/manifest.json` SHA dfc64432210b053fc9db517bae0771fdeb750a5d373b732949ef43e8b65b1a7a；旧清单封存`inputs/server25/manifest.json`。全部引用存在非空、GT/初始化已核验；4722张首/中/末帧字节与解码像素25/26相同，属于抽查而非全图像SHA证明。

RGBT234继承差异：afterrain原始visible首帧宽高24/45，原CADTrack toolkit注册10/80；25/26原始文件SHA同为d57d8478baa37976c283327815730d53e6c61b0db93b4bb1af54b6f41bcb0790。沿用原toolkit/manifest，不能静默改GT后混表。LasHeR沿用原toolkit取整和特殊GT处理：首帧用GT，取整GT任一分量≤0的534行在PR/NPR计成功、SR计失败且保留分母；原始IoU诊断另排除260个无效尺寸行，不能混作正式SR。210为PR/SR，234为MPR/MSR。test不参与训练收益标签或参数选择；210/234独立dev协议仍待登记。

## 保护结果与权重

| 完整test版本 | LasHeR PR/NPR/SR | RGBT210 PR/SR | RGBT234 MPR/MSR |
|---|---|---|---|
| 用户baseline | 72.740/69.160/57.840 | — | — |
| 历史E0 cost0，LasHeR最好 | 74.749703/71.067672/59.089007 | — | — |
| Reference-E0 cost0.05，统一配置 | 72.524939/69.068894/57.732435 | 88.061759/65.741170 | 90.806676/68.910136 |
| E1 cost0.15，完整失败 | 67.612357/62.545797/50.842354 | 84.429457/54.041235 | 87.774864/57.523560 |

不能把历史E0 cost0的LasHeR与Reference-E0 cost0.05的210/234拼成同一配置。Reference-E0始终保护；新训练暂无dev/test结果。

保护依赖：E0 SHA ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c；E1 SHA d4454775f085e1af65c61f53d4d21f284706289b6ff3aeb1167f73d79d31a362；正式B6250父权重`runs/global_candidate_policy_formal_20261014/latest.pth` SHA b84d00454717cae56048753e0fe54c1d6b92cfcd2820c01fa65296508488ccb3；factory `runs/stage_b_refreshed_20261013/latest.pth` SHA ba49b506cd4bdcdfb355e8741feaf6c5c61e2edf9cd0a5c91459c3e5957c80ed。

旧Rank/STOP权重SHA527a3bc0f56f6a26e31adcc5e656f971b6bd34bbcd220251151d7db1cc228753已拒绝晋级，但仍是活动训练计划的不可变出处依赖，**目前不能删除**。同理不删除当前checkpoint、父权重、恢复依赖、正式预测及唯一原始证据。已校验的重复缓存、废弃sanity和传输包可按原授权清理；不因“失败”误删仍被计划绑定的权重。

## 旧Rank/STOP已完整拒绝，不跑test

旧C完整882×2/182182更新已结束。它当时直接启动，没有原生终态退出码；该出处警告永久保留。26本轮三个dev均有真实run_task COMPLETE/exit0，每档97条/54646帧，经原toolkit评分和独立结果审查：

| cost | PR | NPR | SR | 总参数M | 服务FPS | mean/P95 ms |
|---|---:|---:|---:|---:|---:|---|
| 0 | 74.420935 | 68.425355 | 54.648371 | 103.218887 | 7.488519 | 133.538/188.009 |
| 0.05 | 74.027542 | 68.057275 | 54.511459 | 103.218887 | 7.702367 | 129.830/183.889 |
| 0.15 | 71.345271 | 65.726933 | 52.833813 | 103.218887 | 9.426419 | 106.085/153.115 |

原E1 dev参照76.557717/71.365998/56.433144；最佳cost0 SR仍差1.784773点。最终决定`REJECT_REPAIR_NO_TEST`，见`reports/RANKSTOP_DEV_FREEZE_DECISION_20261026.json`。审查`RANKSTOP_COMPLETE_DEV_RESULT_AUDIT_20261026.json`为WARN/0阻断：真实GT、完整曲线/序列覆盖/时间及所有399401个训练收益标签核验；same-family/provisional，模型路由UNATTESTED，不能称跨模型独立接受。

三个dev计时负载不同：cost0先单卡后双卡，cost0.05与GPU0性能任务重叠，二者OpenCV默认80线程；cost0.15采用OpenCV1且GPU0训练并行。FPS为含初始化/解码的逐帧服务时间、排除模型加载，不是同负载配对加速证明。不能与用户test baseline直接相减。

## 现有三个模块与真实边界

1. 目标条件主模态调度：双路64预览、初始模板描述、8维历史，共享CNN3→16→32→32、quality74→64→1；在完整ViT之前选择主模态。
2. 定位收益证据请求：主框/主空间特征＋256全图预览32×32网格＋候选几何，gain955→64→1；最多8个全图位置local/wide＋两个主区域请求＋STOP。廉价打分后只执行0或1次辅助请求。
3. 共享编码局部空间修正：从辅助原图裁ROI才进入共享ViT。local128为193总tokens/8×8，wide256为385/16×16；独立查询坐标可输出旧主crop外框。缓存主特征投影、与辅助特征拼接、共享decode/head定位。

**仍有按需局部空间融合，不能宣称完全无交互/无融合。**主路径两个128模板＋256搜索共385tokens。总参数103218887（103.218887M）；按用户要求不评估可训练参数量。正式结果须带总参数、完整FPS、mean/P95、GPU/输入尺寸与辅助比例。跳过分支不等于减参，当前没有同设备baseline配对加速证据。

## 当前两种训练目标（同结构、同起点、同预算）

GPU0 joint-quality计划`reports/global_candidate_joint_quality_trajectory_plan_20261026.json` SHA014101511170dc2742c1cca4662035bae7a2b106f0f6e38cc10310eda090d42f；入口`train_candidate_joint_quality_trajectory.py` SHAa19e7ae96787250ded975e89f2aaeb87a0f9cb3ab7b631e4fe9294ba4d0c2331。源码SOURCE_PASS，原48源码不变，新49源码/23输入绑定，21项CPU检查。真实sanity4更新/20track/1init、原生0、冻结逐张量相同、执行重放与严格重载PASS，COMPLETE SHA f4d568cf2d48268afc5be16efc4435459dff07f8fdefc031a2d74f98592387e8。

GPU0从原B6250开始，联合训练现有预览CNN、quality/gain，冻结backbone/decode/boxhead/localreadout。新增相对质量监督：d*=IoU_RGB−IoU_TIR，d=Q_RGB−Q_TIR，Lrank=mean(|d*|·ReLU(|d*|−sign(d*)d))；保留原绝对质量回归和SmoothL1 gain＋固定STOP margin。预览LR1e-5、头1e-4、WD1e-4、seed2027、cost0.05、K8、stride5，OpenCV1/Torch1/TF32关闭。

GPU1 gain-pair正式计划`reports/global_candidate_joint_gainpair_trajectory_plan_20261026.json` SHA931b2ce6c4f50c49b923cfd7192419fdac64594d324df5e4f55b2f455234be77；入口`train_candidate_joint_gainpair_trajectory.py` SHA6cede0c60dbba2cb76483ffef0e0dd31571838b6bd63fb3cb8e122d538f8357b。它保留GPU0质量项/训练范围/14个可变state张量，仅将gain项改为0.5MSE真实signed IoU gain＋0.5MSE同状态selected-extra与同role sampled的收益差，移除固定STOP margin。推理仍argmax(predGain−λtokenCost)、STOP=0，不新增层、参数、teacher调用或标签。最终源码审查50份源码/26输入绑定通过，报告`reports/JOINT_GAINPAIR_TRAIN_SOURCE_REVIEW_20261026.json` SHA29f167345f4f66c924a5969bf0dd532f8d4768678cf427338803e8bd8b8cabf3（same-family/provisional，实际路由UNATTESTED）。实际准入已通过并正式启动；证据`JOINT_GAINPAIR_TRAIN_EXECUTION_20261026.json`与`JOINT_GAINPAIR_TRAIN_STARTUP_20261026.json`。

两条均先按自身当前预测前进，再读当前TRAIN GT监督；模板/历史不受反事实teacher污染；sanity权重不进入正式训练。不能仅凭平方损失的理想条件均值性质，声称有限共享网络已校准或闭环有效。下一步必须完整dev验证。

## 已核实的问题及被拒绝方向

- E1完整test退化，不能靠减少辅助调用即宣称有效；固定RGB、强制primary query及readout-only完整dev均已失败，不重复其test。
- 旧TRAIN182182状态中59246次另一路primary IoU更好超过0.01（32.52%）；这只是演化训练状态统计，不是最终固定模型泛化误选率。
- 旧完整TRAIN44443次query中11057有害、14535有益，平均raw gain仅0.01610；STOP状态仍有28630次存在一项采样query正收益。支持相对收益/STOP学习，不能当闭环上界。
- 全3dev诊断见`RANKSTOP_THREE_DEV_DIAGNOSTIC_20261026.json`：全图local低IoU比例很高；主框未逐次保存，不能把query低IoU等同“相比主框有害”。不同cost轨迹历史不同，不能作同状态因果差。
- 模板事件仅根据每5帧/响应≥0.65**推断触发**，不是实际写入tensor日志。多数低IoU触发此前已长期失败且发生在STOP帧；不能认定模板污染导致最初漂移。
- `RANKSTOP_UTILITY_TARGET_DIAGNOSTIC_20261026.json`完整399401标签/35037同状态pair：λ0.05下7626 pair存在更好的另一项，2094 selected净负而sampled净正，23473两者净负。候选相对关系和STOP都重要。
- 旧固定margin在理想单标签标量上可把真实小gain0.02推向0.05，而local cost约0.025；61543标签位于该损失冲突区。但这不是实际共享网络校准偏差证明。
- 初稿utility-hinge在审查CPU Bernoulli反例中不能保持条件均值，已拒绝且未部署/未NN。源码和失败论证保留`.aris/traces/experiment-audit/2026-10-26_joint_utility_source_review/`，不要复活该保证。
- GT挑best8只是候选诊断，不能部署，也不是整轨迹性能上界。候选覆盖、排序、query执行和接受替换须分别检查。

## 可借鉴机制与贡献边界

已核对作者源码：[KeepTrack/PyTracking](https://github.com/visionml/pytracking) commit7eb9e74bd3d40e29dbcec444902237da13de247b，候选身份关联和不确定更新；[MultiL2D](https://github.com/rajevv/Multi_L2D) AISTATS2023 commit3f2ad6bc93afdfd250faac21225f6b55074b6cf8，专家求助/代价决策；[MoETrack](https://github.com/Zhangyong-Tang/MoETrack)先完整提取两路再比较专家响应。

我们拟研究预编码的目标相关空间请求、同状态执行收益与STOP竞争，以及独立坐标证据读出。尚不能把三个模块写成三个已证实原创贡献。KeepTrack重型候选特征、分类专家校准理论不能直接移植成我们省编码/IoU保证。CAD式模板更新是公开策略对照，不算原创模块；架构内参数或层数专用配置须匹配重训和披露，先保护成功版。主目标差约4点，仍需方法/训练修复，暂不seed收尾。

## 显存与速度：已完成有限测量

独立8条TRAIN×128前缀，共1024帧，进程1/2/4/8/重复1、OpenCV80与1线程比较，所有保存框/路由/最终状态摘要相同。峰值显存分别2291/4557/9087/18149MiB。OpenCV1单进程10.069～10.615 FPS；8进程合计49.330892 FPS、GPU平均利用率73.5%、峰值17.724GiB，启动至join23.136925 FPS；各worker6.166～7.459 FPS。OpenCV80单进程8.442～8.823 FPS，8进程合计38.451 FPS。

**49.33是多序列并行吞吐，不是单目标在线FPS。**该前缀仅19次辅助请求、没有全图action3/4，不能外推难例完整test或训练。固定顺序/负载漂移和结果范围已披露；summary `reports/RANKSTOP_INFERENCE_SCALING_SUMMARY_20261026.json`，独立结果审查WARN/0阻断。CPU读图/均值/裁剪有开销，但尚未实现新裁剪优化。用户要求尽快训练后，N10草稿已撤销，不再增加测速任务。

## 后续执行规则

1. 保持GPU0/1两条完整训练；依据新增并发实测选择有意义的同架构有限配置并行，不重复同一配方，不中途换正在运行的源码/权重。GPU2/3现有任务结束后才接入。
2. 按预计节点检查有限损失、实际native终态/覆盖/完整计数；不提前反复查状态。正式完成后收回原始日志、配置、COMPLETE/校验身份，保留必要权重。
3. 完整97dev三档cost，按独立dev冻结checkpoint/config，再决定完整LasHeR/210/234 test。旧Rank/STOP已拒绝，不重复test。新方法需优于保护参照才有实质价值。
4. 全量预测/原toolkit指标、总参数、完整FPS和诊断同版本交付；不拼不同权重/seed指标，不用训练loss填PR/NPR/SR，不预先声称加速/减参/创新有效。
5. 只维护这一个交接文档；过时运行叙述由本文件替换，历史证据和不可变哈希输入不重写。源码/文档及时白名单提交GitHub，环境、权重、原始数据/私有逐帧GT不提交。

## 并发训练实测与容量决定

独立任务`profile_candidate_training_scaling_20261026`已COMPLETE/exit0，16个worker全部原生exit0，共1280次真实反向/AdamW、6400次track。每档相同8条TRAIN前缀×32更新＝256更新；原GPU1正式训练一直并行。因此“追加1/2/4/8”分别是卡上总2/3/5/9个进程。

| 追加进程 | GPU1总进程 | 附加训练更新/秒 | 含spawn/join更新/秒 | 结束驻留MiB | 原formal近似更新/秒 |
|---:|---:|---:|---:|---:|---:|
| 1，首档 | 2 | 1.166733 | 1.085096 | 4773 | 1.2454 |
| 2 | 3 | 2.167552 | 1.895653 | 7127 | 1.2561 |
| 4 | 5 | 3.544380 | 2.826138 | 11835 | 1.0181 |
| 8 | 9 | 4.949214 | 3.475540 | 21243 | 0.4710 |
| 1，复测 | 2 | 1.148609 | 1.067944 | 4773 | 1.2267 |

首尾追加1漂移−1.5534%；追加2相对两基线1.8578～1.8871倍。结束驻留**不是连续NVML峰值**，结束GPU利用率亦非工作期间均值；allocator峰值不含其他进程/驱动。原formal速率来自每25步刷新的日志短窗，状态不同，不能当严格同状态因果时延。保存的更新帧预测/非计时路由相同，但89行loss最大差5.96e−8，不能声称权重逐位相同；只保存更新帧路由，不能统计全部6400帧辅助比例。这是训练吞吐，不是单目标在线推理FPS，也不是完整882训练提速。

独立审查`TRAIN_CONCURRENCY_RESULT_AUDIT_20261026.json` SHA5f2f6d858a92b8cbbc23d0bccfed6b05cd87e41b473879ef5b51e8d6b151ebb9：579项检查、51源码/28输入/67归档副本一致，RESULT_AUDIT_PASS_WITH_LIMITATIONS/0阻断，容量证据PARTIAL。直接支持单卡短测总3进程可行，两卡及完整不同配置仍属外推。按`TRAIN_CONCURRENCY_CAPACITY_DECISION_20261026.json`先在GPU0/1各放到3项，保持原任务，30分钟节点复核。9进程附加总吞吐最高，但原formal明显变慢；目前只有有限8个有意义配置，不为占满显存添加重复训练，保留两项等GPU23空闲。

两种目标×K4/8×轨迹cost0/.05为预登记8配置；注册表`TRAIN_CONCURRENCY_CANDIDATE_GRID_20261026.json`/计划索引不是实时状态，不重写哈希绑定登记文件。六个新增计划源码准入及独立真实M0依次执行，汇总`CONCURRENT_TRAIN_GRID_SOURCE_REVIEW_20261026.json` SHA6b1b333e2d66c846d84756e1395f9f22e33260b62aa93cc2c588d027a9939ee6；控制器`launch_concurrent_training_wave.py` SHAcbcd5a4465cabd55dfa95d8797b8ff9832b3e7d3766f37600ec58ccae4324f6b，等待器SHAc75c125131008971918cb852437d94203e6ce3079c915faf2a2e7a2a960d57df，均SOURCE_PASS/0阻断。新`run_task_four_gpu.py`仅扩展GPU选择，原run_task不热改。审核same-family/provisional，实际模型/effort路由UNATTESTED，不宣称跨模型独立接受。

清理：校验原生0、COMPLETE/SANITY、SHA及所有活动计划引用后，仅删除原两项不作正式初始化/输入的4步sanity latest.pth，共827322294字节；保留metadata/CSV/native。新增四项sanity权重待首次持续并发节点稳定后同样检查清理。所有正式/父/预训练/E0/E1及当前计划绑定的旧Rank权重仍保护。见`SANITY_WEIGHT_CLEANUP_20261026.json`。
