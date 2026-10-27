# Dual-RGBT 交接文档

更新时间：2026-10-27。此文件是唯一交接文档；同步到本地桌面 document/Dual-RGBT交接文档.md、26服务器项目根目录及 GitHub。旧逐次节点以 Git 历史和 reports 原始证据保留，不再追加相互冲突的“当前状态”。

## 当前训练与优先级

2026-10-27 06:35 UTC实际远端核对：四卡共19项正式训练RUNNING，GPU0/1/2各5项、GPU3四项；所有wrapper与子进程存在，原两项已68100/59400更新。每项原B6250/SOT epoch14初始化、完整882条TRAIN×2轮、182182更新/917106次track/1764初始化；不读取活动latest权重。原八项配方及活动训练源码不热改。新增项只是在预登记有限32配置中改变K/轨迹cost，不增添网络模块，不作为精度有效证据。

资源同次原始样本：GPU0/1/2/3为11803/11795/11795/9599MiB，瞬时利用率56/83/53/47%；MemAvailable37636920KiB（35.89GiB），SwapFree1092KiB。显存目标18GiB/卡尚未达到。当前瓶颈是每进程约4～5GiB主存、机器物理RAM约125.5GiB；不能把单看显存得出的32项容量当作可用容量。暂不启动最后8项，先验证最多24项的实际吞吐、原任务减速及交换增长。

串行扩量控制器27875的本地句柄已不存在，CIM也未发现控制器；报告停在FORMAL_IDENTITY_PERSISTED。实际已有三项phase6新增训练正常推进：GPU0 wrapper2304707→2304708、GPU1 2307201→2307202、GPU2 2309694→2309695；没有训练失败证据。控制器中断的异常原文未取得，不能将90秒启动检查尚未记录成功写成已证实根因。原始部分报告封存为VRAM18_WAVE6_INTERRUPTED_EXECUTION_20261027.json，实际19项见VRAM18_WAVE6_PARTIAL_NATIVE_DIAGNOSIS_20261027.json。接续入口已独立SOURCE_PASS/0阻断，实际启动本地61089（tools/resume_vram18_serial_training_wave.py，SHA2a9ecff4），预计06:58 UTC结束逐项180秒检查；只接续尚未启动5项，复核已有3项身份、全部8项M0原生exit0和输入哈希；不得重启原控制器或重跑已有训练。

观察器66790已按16项节点exit0完成，93197已主动取消以免重复；二者均关闭，不再加入。当前没有定时观察器；接续启动及完整24项节点以新实际回执更新。此处是当前状态，后文旧节点仅为历史证据。

无新版完整dev/test或正式在线FPS。LasHeR目标仍未达到，保留历史最佳E0和统一三基准Reference-E0；现有103218887总参数的新候选模型等待完整训练与独立dev选型。

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

1. 保持当前四卡八项完整训练（3/3/1/1）；不重复同一配方，不中途换正在运行的源码/权重。GPU23已在原占用退出且实际空闲后接入，旧等待器已结束。
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

清理：校验原生0、COMPLETE/SANITY、SHA及所有活动计划引用后，仅删除原两项不作正式初始化/输入的4步sanity latest.pth，共827322294字节；保留metadata/CSV/native。新增四项也在18:48持续节点稳定后逐文件校验SHA/原生0/引用关系并删除，仅1654644588字节；累计六份准入临时权重清理2481966882字节（约2.31GiB）。所有metadata/CSV/native保留，正式/父/预训练/E0/E1及当前计划绑定的旧Rank权重仍保护。见`SANITY_WEIGHT_CLEANUP_20261026.json`与`CONCURRENT_SANITY_WEIGHT_CLEANUP_20261026.json`。

## 新训练完成后的评测入口

已新增独立`run_joint_candidate_evaluation.py`和`tools/joint_candidate_results.py`，不编辑活动训练绑定的原51份源码，也不热改旧评测入口。原因是旧入口写死旧E1训练身份、K8和GPU0/1，不能直接承接本批联合预览/策略训练的K4/K8权重。新协议登记八项既有训练计划，沿用原模型、tracker、GT和评分工具；当前仅完成源码准备，没有新GPU推理、完整dev或test。

使用顺序：训练原生COMPLETE/exit0及完整882×2计数→CPU核验最终权重/原B来源与冻结张量→真实11帧TRAIN烟测和严格重载→完整97dev的cost0/0.05/0.15→独立结果核查→若SR超过旧E1完整dev参照则冻结一份配置，完整测试LasHeR/210/234，否则拒绝test。正式评分附总参数103218887、含初始化/解码的服务FPS及mean/P95；多分片吞吐不写作单目标FPS。

源码审查首轮发现报告cost未绑定运行配置、曲线长度仅检查序列维；已在共享校验处增加实际配置cost一致性和PR/NPR97×51、SR97×21精确形状检查。首轮失败证据封存；修复后8项针对性合成CPU检查和55份源码哈希复核通过，复审SOURCE_PASS/0阻断（same-family/provisional，实际模型路由UNATTESTED）。这些是尚未执行的新入口校验问题，未影响现有训练或历史指标。入口严格拒绝在训练仍RUNNING时读取其latest.pth，不借准备评测提前轮询当前训练。独立入口已部署，不启动新GPU评测、不读取活动权重；CLI的prepare/score/freeze/commands本地帮助检查通过。该05:23八项节点后来被16项节点替代；两个原观察器均已关闭，当前状态以上方为准。

新增GPU23的12份sanity配置/COMPLETE/重载检查/CSV/native状态与日志已归档，共39968字节，不复制权重或私有标签。30分钟全训练健康节点后，核验全部八计划的输入/父权重引用、原生exit0、COMPLETE SHA和二进制SHA，仅删除两份不作输入的4步检查权重827322550字节；累计八份已清理3309289432字节（约3.08GiB）。见`FOUR_GPU_RESERVED_SANITY_METADATA_COLLECTION`、`FOUR_GPU_RESERVED_SANITY_WEIGHT_CLEANUP`对应20261026 JSON；正式八项、B/factory/预训练/E0/E1/旧Rank依赖仍保护。

## 用户18GiB/卡扩量要求与实测主存问题

10月27日04:03 UTC用户要求尽量让四卡各到约18GiB。每进程实际2350MiB显存，预计8项/卡、共32项接近18.4GiB；新增24项为原两目标×K4/8×轨迹cost0/0.0125/0.025/0.05/0.075/0.1/0.15/0.2的有限32配置中尚未运行的组合，不重复原八项、不换seed/学习率/batch/结构。原八项源码、输入计划、环境及权重均保持不变。首批八项已通过真实准入并正式启动，首批16项节点已完成；其后phase6实际启动三项，目前19项，当前情况以上方为准，不把计划32写成已启动。

04:06:50实际核对八项均RUNNING/PID正确/全部loss有限，更新58025/49025/42725/42875/43900/44350/24250/24350；四卡显存7092/7129/2373/2532MiB，CPU11.68%，可用RAM约77.8GiB，项目分区空余约6.9TB。每进程PSS约4.7～5.3GiB，直接32项会超过125.5GiB物理RAM。只读smaps进一步定位一项约4.2GiB私有页在CPU heap；这里不是显存不足，须先减轻CPU缓存。没有读取温度/功率。

原M0内存采样在/proc子进程退出瞬间遭ProcessLookupError，清理逻辑只杀回收该临时组2165689（实际-9），留下native RUNNING旧记录而model COMPLETE4/probePASS；整次拒绝，不作native0或正式初始化。原八项未受影响。R4新任务分别native0：默认2180647、分配器2182381；模型265张量逐位相同、轨迹/标签CSV相同，但五份Adam缓冲最大差4.66e-10，因此原“优化器逐位一致”门槛失败，完整保留，未据此启动新增正式训练。分配器环境仅把M0末次采样PSS从约5.38降至3.76GiB，仍不足以直接扩32项；峰值采样含重载clone，不能当正式稳态内存。

当前独立单次新准入在测试：新增薄入口仅在原模型加载后执行一次gc.collect＋libc.malloc_trim(0)，然后完整委托原trajectory_stage；不改模型、参数、优化器、RNG、数据或更新公式。复用已完成默认R4作为明确对照，不重复它；前瞻门槛仍要求265模型张量/CSV逐位一致，优化器有限且atol1e-8/rtol1e-6，记录实际是否逐位及最大误差，不宣称全轨迹等价。原严格失败不删除。这是工程内存优化，不是新增论文模块。

扩量按每卡4→6→8项分批，下一批必须有实际同波次至少10分钟吞吐、原八项减速、进程Swap增长和RAM余量记录。总吞吐须提高，原任务速率保留≥0.4，Swap增长≤64MiB，MemAvailable≥8GiB；未过则保持当前波次，不停止/重启已有任务。新GPU上下文未建立的CPU初始化阶段按已注册槽位计数，不能误判为未知GPU进程；逐任务立即保存启动身份。新模型正式评测仍是完整dev/冻结后三数据集，附总参数和服务FPS，不用并发更新吞吐当在线FPS。

05:08 UTC首批扩量actual FORMAL_STARTUP_PASS，16项/每卡4项。八项新M0均真实4updates/20track/native0、严格重载/冻结/重放通过，正式从原B重置。四卡9447/9484/9439/9598MiB；可用主存48.52GiB，新进程PSS约4.365～4.369GiB、当时Swap0。加载后trim只归还82KiB；另一个独立第四次优化更新后trim只归还116KiB，不能声称CPU堆问题已解决。两处新探针均有native0，模型265张量/两份CSV逐位相同，优化器仅有限容差内相同、非逐位。暖机后trim仅为失败的工程假设，不进入正式新训练；现有16项保持不变。

| GPU | 首批新增正式wrapper→trainer | K / 轨迹cost |
|---|---|---|
| 2 | 2231480→2231482 | 4 / 0.0125 |
| 2 | 2231657→2231658 | 4 / 0.025 |
| 2 | 2231809→2231810 | 4 / 0.075 |
| 0 | 2231989→2231991 | 8 / 0.0125 |
| 3 | 2232197→2232199 | 4 / 0.0125 |
| 3 | 2232377→2232378 | 4 / 0.025 |
| 3 | 2232524→2232525 | 4 / 0.075 |
| 1 | 2232676→2232677 | 8 / 0.0125 |

05:24:59 UTC首个固定波次节点16项RUNNING、命令/父子关系及所有loss有限。约1011秒区间实际合计15.7002更新/秒；旧八项前段参考9.9133，运行估算比1.5837，旧八项单项保留0.875～0.991。该区间有临时内存探针重叠、状态也在演化，不是严格同状态或完整训练加速证据。进程Swap增长27016KiB，计分时MemAvailable约48.80GiB，容量门通过。见`VRAM18_WAVE4_HEALTH_20261027.json`。

原05:23八项观察器93197已核对真实CIM命令后仅停止本地PID42236；原生exit1是主动取消重复观察器，不是训练失败。替代16项观察器66790/本地5108已按05:24:47节点完成exit0；二者均结束，不要重启或重复加入。下一批采用新独立外部调度器`launch_vram18_serial_training_wave.py`：M0两批各4，正式逐项90秒检查后再启下一项，继续使用原已登记计划/训练器/起点和源码；保持活动V1控制器、索引和16项输入不变。新调度器须独立SOURCE_PASS且旧波次实测门通过后才启动。32项同栈主存预计超过128GiB，18GiB显存目标尚未达到；不能为了数字填充虚假占位或隐瞒主存限制。

结果审计已完成WARN/provisional、确定性检查PASS/0阻断：88份元数据、63个源码哈希及wave4更新计数一致；仅支持短程数值容差和运行容量结论，不支持整程等价、malloc_trim明显省RAM、受控加速或在线FPS。见VRAM18_WAVE4_AND_MEMORY_RESULT_AUDIT_20261027.json/.md。旧13份临时权重的比较已封存，内存探针方案退休，不再以其临时权重作为未来准入输入；活动正式权重、B/工厂/原预训练/E0/E1/旧Rank依赖均保留。

06:46 UTC维护：13份已审计并入Git的退休M0临时权重已按精确SHA、native结束与全部32计划输入依赖核对后清理；只删除指定latest.pth，不删除任何metadata。清理回执VRAM18_RETIRED_M0_WEIGHT_CLEANUP_20261027.json；phase6八份M0仍暂留，待24项健康节点后再清。

06:51 UTC接续中途只消费原生61089输出（没有额外远端轮询）：GPU3 gainpair K4/cost0.1 wrapper2341806启动检查75更新、GPU0 quality K8/cost0.075 wrapper2343989检查50更新，GPU1 gainpair K8/cost0.075 wrapper2346331已登记正在启动；原19项未重启。唯一后继观察进程为原生58963，私有脚本dual_rgbt_vram18_register_capacity_node_20261027.py先等到06:59:30 UTC，要求实际phase6 FORMAL_STARTUP_PASS/24项后才登记唯一health节点，再按最后实际启动+1200秒观察。61089预计06:58结束前不重复查询；58963在健康节点前不提前加入或另开观察器。新只读observe_vram18_capacity.py经SOURCE_PASS/0阻断，在同一节点保留原始meminfo/GPU与带时间戳的约10秒CPU ticks；不查温度/功率、不改训练源码。
