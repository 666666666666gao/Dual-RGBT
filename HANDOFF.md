# Dual-RGBT 交接文档

更新时间：2026-10-27。此文件是唯一交接文档；同步到本地桌面 document/Dual-RGBT交接文档.md、26服务器项目根目录及 GitHub。旧逐次节点以 Git 历史和 reports 原始证据保留，不再追加相互冲突的“当前状态”。

## 当前训练与优先级

**最新用户指示（10月27日）：先以精度提升为准，FPS先不管。** 暂停新增阶段测速、速度优化及轻量化分支；已完成的配对测速仅保留历史证据。现有32项训练按原计划完成，优先完整LasHeR dev选型与正式test。不能为减少辅助调用牺牲精度，也不把FPS提升作为当前精度版本晋级前提。原计划的cost网格不热改，它们用于比较精度；实际成本不是本轮选型目标。

2026-10-27 14:50 UTC固定节点：26服务器24项、25服务器8项，合计32项完整正式训练全部原生RUNNING、实际wrapper/child身份正确、已有loss有限。每项同一原B6250/SOT epoch14初始化，882条TRAIN×2轮、182182更新/917106次track/1764初始化；不热改活动训练源码/计划/输入，不把训练loss当精度。

| 主机 | 正式任务 | 本次区间合计更新/秒 | 显存/卡 | 可用RAM | CPU busy |
|---|---:|---:|---:|---:|---:|
| 26 | 24，每卡6项 | 20.7219（约2小时） | 13.82～13.97GiB | 12.52GiB | 31.89% |
| 25 | 8，每卡2项 | 14.0238（约80分钟） | 4.61GiB | 200.09GiB | 14.49% |

CPU采样分别36.90秒/11.63秒，含快照读取开销；GPU利用率是节点瞬时值。不同轨迹/窗口的更新吞吐不是严格配对scaling，也不是在线FPS。原八项26依次98550/90525/82200/82925/85550/87050/65300/65975更新；25八项13300～14050。首quality已进入第二轮，gainpair尚未到91091轮界；均未完整C×2。按本次区间线性外推，26各项约剩22.5～57.6小时，25约26.0～27.4小时；不是完成承诺，未计未来轨迹及保存变化。

26物理RAM约125.5GiB，额外8项会超过实测主存预算，继续HOLD，不为占显存增加任务。原26 phase8八个配置已经移交25，禁止重复启动；原32注册索引保留历史。所有当前权重、B/工厂/SOT预训练/E0/E1/旧Rank依赖受保护；已归档退休M0累计按原回执清理，metadata与失败证据保留。

14:50两个旧观察器20467/42977均已actual exit0消费；CPU安装观察41316也已消费。新唯一26观察16720、25观察25243均已actual WAITING到20:50 UTC；此前不另查全机训练进度，不重复观察器。用户新要求的配对FPS任务另做必要的同卡所有权准入与恢复核验，不当作全机提前轮询。最新统一证据：reports/DUAL_SERVER_FULL_C_PROGRESS_20261027.json及其绑定两机原始节点。

当前最好LasHeR仍为历史E0。32项新训练暂无完整dev/test，78/72/62目标未达到。之后仍按完整native0训练→完整97dev三cost→独立结果审查→冻结配置→三基准执行；正式评估报告总参数/FPS，不统计可训练参数量。

## 25服务器作为新增算力（2026-10-27用户新授权）

25的四张RTX3090在新建项目时已核实为空，无compute PID，RAM总251.54GiB、准备时可用约242.33GiB。用户明确项目和conda均放/home/gaob：已新建/home/gaob/Dual-RGBT，并把原conda完整复制回该项目的conda-envs/dual_rgbt。不是恢复旧队列；原归档保留，26四卡24项任务保持不变。当前两机固定观察节点及句柄见上文。

原环境实际Python3.7.16/Torch1.12.1+cu113/timm0.9.12，无pip升级。源码审查后，父见证实际exit0完成CUDA前反向/AdamW更新/严格重载以及11帧真实TRAIN；fresh agent按docs/SERVER25_REJOIN_20261027.md原样命令再执行一次，实际exit0、17.12秒，框/路由/模板/历史重载逐位一致，总参数103218887。该短程smoke FPS分别4.966/4.711，仅作环境见证，不能写成正式FPS或PR/NPR/SR。环境状态ENVIRONMENT_ACCEPTED，新的训练仍需单独真实4步准入。

已恢复原25 manifest SHA389ba37d，882TRAIN/459435帧、97dev/54646帧；245 LasHeR test/220703帧、210/104706帧、234/116650帧。每序列首/中/末双模态存在性抽查共10008文件，不是全内容比对。LasHeR在/data2/wangwj/dataset/LasHeR-0428，210在/data/wangwj/dataset/RGBT210testingset，234在/data2/wangwj/dataset/RGBT234/RGB_T234；共享数据不复制也不修改。代码/依赖权重/环境全在/home/gaob项目内，完成必要恢复后可用约23.70GiB。

只读检查原26 phase8八个未启动任务/输出路径均不存在后，为这八个已登记K/cost组合生成独立server25计划，仍同一原B6250/SOTep14、seed2027、同网络/三模块、882×2完整训练和182182更新；每卡两项，避免与26已有24项重复。新25使用原生joint_quality/gainpair训练器及默认分配器，不用无明显收益的heap-trim包装；注册表SERVER25_REMAINING_GRID_20261027.json。新25调度已SOURCE_PASS/0阻断，源码54份与旧网络/配方一致。实际wrapper209569于12:30:08 UTC启动独立准入控制器；先8项M0各300秒统一节点，原生exit0/冻结/重载通过后才从原B启动8项完整formal，正式启动再等180秒检查。第一次观察预计12:40:08 UTC。12:40:21 UTC已得到实际控制器COMPLETE/exit0及8项formal RUNNING/PID/命令/输入/所有loss有限回执，每卡两项约4716MiB、可用RAM约200.35GiB。调度完成不是完整训练完成；现在两服务器共32项正式训练。首25观察34849已实际exit0消费，25观察30699已在13:30实际exit0并消费，八项继续RUNNING且全部loss有限、约4716MiB/卡、可用RAM约200.21GiB；49.86分钟区间合计13.5385updates/s，含GPU0两次主动暂停约49.55秒，不能当纯吞吐配对试验。14:50 UTC两机观察已结束：25八项继续RUNNING/finite；下一两机固定节点20:50 UTC，观察句柄与原始证据以上方当前状态为准。不得用旧26计划在25直接运行，也不得因新增机器改动活动26源码。

## 目标、来源及运行位置

LasHeR优先：78/72/62仍是当前主动目标的阶段里程碑；用户最新贴出的审查重新强调三项各＋6，即78.74/75.16/63.84为更高最终目标，不能用阶段里程碑宣布各＋6完成。两者均未达到。还须完整RGBT210、RGBT234测试；当前先追求精度，FPS工作暂停。

用户baseline为72.74/69.16/57.84，沿用、不重测精度。初始化始终为**去掉CADTrack论文三个模块、SOT来源backbone、epoch14**，不是完整CADTrack或DropMAE。

| 项目 | 当前位置 |
|---|---|
| SSH | `2026`主任务＋`2025`新增四卡 |
| 项目/代码/权重/输出 | 26 `/data2/gb/Dual-RGBT`；25 `/home/gaob/Dual-RGBT` |
| Python/conda环境 | `/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python` |
| 本地源码 | `C:/Users/gb/projects/Dual-RGBT` |
| GitHub | https://github.com/666666666666gao/Dual-RGBT |
| 预训练 | `checkpoints/CADTrack_ep0014.pth.tar`，SHA256 cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731 |

25服务器原活动项目已核验归档到`/data2/gaob/Dual-RGBT_archive/retired_server25_project_20261026`，该路径曾在退役后不存在；10月27日按用户新授权在原路径新建独立25项目，仍不启动旧归档队列。共享数据与外部原始预训练保留。26环境Python3.7.16/Torch1.12.1+cu113/timm0.9.12/NumPy1.21.5，环境迁移、真实CUDA更新/重载、11帧tracker和独立文档执行验证均通过。

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

07:22 UTC固定24项节点已由fresh same-family审计PASS/0阻断（provisional、实际模型UNATTESTED）：重算21700/1031.522=21.036875updates/s，区分16项与更早8项分母，确认CPU实际25.025秒与两个RAM读取时刻，RAM HOLD不因吞吐门通过而解除。见VRAM18_WAVE6_CAPACITY_RESULT_AUDIT_20261027.json/.md。

## 原始baseline参数与速度（用户2026-10-27要求）

对象由checkpoint SHA cbcc96ba确定：去CADTrack论文三个模块、SOT来源、epoch14的CADTrack_ep0014.pth.tar；用户本轮口述约72.4，沿用原历史baseline72.74/69.16/57.84，不重测精度。本次用原lib/test/tracker/cadtrack.py及原配对模板更新规则，RTX3090 GPU0、FP32/TF32off、search256/template128×2、batch1。

原生Runtime2 exit0：总参数103138437（103.138M）。固定TRAIN三序列bluegirl/bike2/comingboy'shead，各20warmup＋200measured，共600计时帧。含RGB/TIR读图解码的端到端20.878209FPS，mean47.896828ms、P9568.888811ms；不含读图的在线服务49.901808FPS，mean20.039354ms、P9526.821299ms。初始化单独记录；PyTorch峰值allocated543021056B、reserved612368384B，不是全卡显存。只是600帧TRAIN样本，不能写成245条完整testFPS；不与历史不同负载的Ours8.786直接算加速比。不统计可训练参数。

为排除同卡算力争用，实际只SIGSTOP25 GPU0上我们自己的两组223759/223760（子223766/223768），其他6项25与24项26不动。Runtime2计时段finally恢复，暂停45.250452秒，实际状态S/S/R/R；后续两项已3300/3275更新、RUNNING且loss有限。不能把这次人为暂停导致的吞吐变化当训练故障。

完整保留失败：SourceR1漏绑配置/原模型依赖，补齐后R2通过；Runtime1因迁移缺lib.train.data.loader在导入时失败，未计帧、无FPS，finally已恢复两组（4.304秒）。只补回7个原始旧工具文件，全部已有源文件严格比对不覆盖，CUDA不可见CPU导入实际exit0，再绑定完整原依赖112项并R3审查通过。测速脚本、模型、权重和600帧选择未变。该迁移/哈希修复是正确性工程，不能当论文创新。

原始600行CSV、nativeexit、所有权与恢复回执位于runs/cadtrack_baseline_latency_r2_20261027及reports/CADTRACK_BASELINE_LATENCY_R2_NATIVE_20261027.json；fresh same-family结果审查PASS/0阻断，重算FPS/mean/P95及每帧decode+service=endtoend一致，actualmodel UNATTESTED、provisional。

13:30稳定节点后，25上8份四步M0权重经原生exit0、精确SHA、所有新25输入依赖和Git归档metadata复核后清理；活动8份formal权重/原B/工厂/SOT预训练/旧Rank及baseline均保留。回执SERVER25_SANITY_WEIGHT_CLEANUP_20261027.json。

原26 phase8的八个K/cost组合现已转由25的新任务正式运行；即使26后续内存释放，也不得再启动旧26 phase8，防止同seed/同配置重复训练。原32索引保留历史不改，当前所有权以SERVER25_REMAINING_GRID及实际原生回执为准。

## 双服务器评测路径准备（2026-10-27；尚无新模型评测）

原joint评测入口通过benchmark_inputs常量绑定26绝对路径，不能直接核验25的真实native命令。现仅将3份独立评测代码的命令生成/核验改为使用protocol中明确的provider路径；NN绑定的benchmark_inputs、全部活动训练源码、计划、输入和权重不改。原8项/26完整24项/25新8项分别注册协议，原8是26的子集，26与25的24＋8配置不重叠。89份活动源码与计划逐字节SHA复核未变。完整native0、182182更新、882×2、265张量/251冻结/14变化、真实smoke、完整97dev三cost、独立DEV审查、冻结配置及三基准门槛继续保留。训练仍RUNNING时拒绝读取其latest.pth；没有启动真实模型prepare或新GPU评测。

审查R1发现两个实际问题：25的680个annotation SHA键使用Windows反斜杠，Linux原样拼路径实际FileNotFoundError；以及沿真实推理import链漏绑hann等7个既有依赖。R2只在评分哈希查找时规范分隔符，并补既有依赖SHA，不改manifest、原工具包、GT或指标公式。部署预检又发现26未调用的adapter.py与本地仅换行符不同，预检拒绝写入。R3将模型依赖范围收紧为真实import图中的11份文件，43份实际import依赖仍全部绑定，不覆盖26该未用文件或任何活动训练源码；最终源码集合78/81/81。fresh same-family源码审查通过后安装精确guard（provisional，实际模型UNATTESTED）；原失败与修复前字节、合成CPU命令/拒绝活动权重检查保留在.aris/traces/experiment-audit/2026-10-27_backend_evaluation_binding/。这些是入口正确性修复，不是方法增益或论文创新。

原25归档没有toolkit，现从用户Desktop/document/cadtrack/toolkit原样复制到25项目vendor/cadtrack/toolkit，7个绑定评分源码及680个原始annotation SHA全部相同。独立CPU导入发现原NN环境Python3.7不支持该工具包list[str]注解，实际exit1原文保存在SERVER25_EVALUATION_TOOLKIT_GT_IDENTITY_20261027.json。**25的独立CPU conda已验收：conda-envs/cpu-scoring，Python3.10.14/NumPy1.26.4/Matplotlib3.8.4；26独立CPU运行时也已完成离线安装与独立文档验证。** 新CPU环境未修改活动NN环境或原工具包。父级与fresh文档原样命令均实际exit0，复核245/210/234所有原始标注、7份评分源码/680份annotation SHA、7项分析可知答案的合成评分检查；不是模型PR/NPR/SR或FPS。见CPU_SCORING_ENV_ACCEPTANCE_20261027.json。

## CPU评分环境与配对测速补充（2026-10-27）

独立CPU spec为.aris/compute/cpu-scoring-env.json，canonical SHA5e574a08；72个conda包的exact explicit清单已保留。R1直连下载约12KB/s、180秒native28，R2误用HTTP CONNECT对SOCKS端点native56，均在创建环境前失败；R3通过临时SOCKS5h代理12秒取同一官方micromamba2.0.5，199秒conda create0，随后错误list--explicit child109使原builder留下stale RUNNING/PID已不存在。原始文件不改写；更正为工具实际支持的env export--explicit，独立post-install export及CPU witness均0，没有重装。renderer仅修正这条真实失败命令。临时代理进程33876已关闭，32705句柄实际exit1消费；不是训练失败。

fresh文档执行的第一个附加PowerShell捕获包装器工具返回-1，起始目录未创建，原样SSH是否启动无法证明；保留EXECUTION_UNATTESTED。核实精确远端输出不存在、无相关存活命令后，fresh直接执行文档原SSH命令，actualexit0/4.60秒、逐字节原报告SHA98ea0489，无文档差异。实际模型路由UNATTESTED，same-family/provisional；不是跨家族验收。仅CPU NumPy计算/保存重载/真实标注身份与合成工具检查，NN/权重/模型预测读取均0。

用户要求baseline/Ours参数与FPS对表后，新增独立外部脚本tools/profile_baseline_e0_pair.py及固定计划。R1源码审查补齐实际native task/ppid/命令准入及6份真实import初始化文件，R2 SOURCE_PASS/0阻断；119源码、63份实际import依赖全部绑定，没有改网络。固定历史E0权重ee60704e从退休归档恢复到25 home项目，原训练native出口出处不足警告永久保留；未读取当前32活动权重，不重测任何精度。按baseline-before/E0cost0/E0cost0.05/baseline-after四轮，同600 TRAIN帧、每序列20warmup、FP32/TF32off进行计时。native526316已COMPLETE/exit0；同卡两个已核实本项目组暂停145.90秒后finally恢复，之后loggedupdates由15750/15550增至16200/16000。逐帧CSV2400行及原生回执已保存；fresh结果审查RESULT_PASS、范围WARN、0阻断，按原CSV重算一致，不把历史不同负载8.786/13.767与baseline单独20.878直接相除。

| 同条件配对速度，RTX3090 FP32 | 总参数量 | 含双模态读图/解码FPS | 不含读图/解码的服务FPS |
|---|---:|---:|---:|
| 原baseline，前后两次合并 | 103.138437M | 26.032034 | 54.524579 |
| 历史E0，LasHeR-best cost0 | 103.374215M | 16.186571 | 24.042206 |
| Reference-E0 cost0.05 | 103.374215M | 16.739742 | 25.213747 |
| 当前在训32版本，同架构不同策略配置 | 103.218887M | 待完整权重测量 | 待完整权重测量 |

每个E0为相同3条TRAIN序列/600帧；baseline同600帧前后各测一次，合计1200帧按总时间合并，不是算术平均FPS。初始化单列，20帧暖机不计入；既有模板更新与因果历史照常运行。baseline前后service/端到端漂移约0.2033%/0.2058%；另一卡训练与主机其他CPU工作继续。主表是该固定采样的同机比较，不是245条完整testFPS、不能替代精度评估。前面baseline单独20.878/49.902是不同时间的独测，保留原始记录；做此次速度对比以本表为准。

**新证据：现有E0参数略多，且本次服务与端到端速度均慢于原baseline，尚不支持“推理更快”。** E0 cost0/0.05分别445/310次辅助请求（600帧），降低请求量也未产生同比例加速；不同因果轨迹不能据此分解单次调用成本。下一步效率改进需先用分段计时定位主路径、ROI预处理、辅助编码和空间读出的实际占比，不把token成本等同延迟，不在活动训练源上热改。当前32版参数比baseline多80450（约0.078%），不是减参结果，其速度与精度均待完整产物验证。

原始数据runs/baseline_e0_paired_latency_20261027/、BASELINE_E0_PAIRED_LATENCY_NATIVE_20261027.json；比较计算BASELINE_E0_PAIRED_LATENCY_COMPARISON_20261027.json为审查前的计算快照，其AUDIT_PENDING状态原样保留，最终审查状态以BASELINE_E0_PAIRED_LATENCY_RESULT_AUDIT_20261027.json为准。未新增模型或训练源码修改；仅创建外部计时脚本、计划和证据文件。

## 融合口径及当前精度优先决策（2026-10-27）

已再次读当前源码：baseline共享ViT权重、两路分别自注意力，末端cat及decode_fuse_search仍做融合；Ours主路径将另一半置零，局部query把主特征投影到query坐标后cat/decode，仍属按需空间融合。去掉CADTrack论文三模块不等于去掉一切融合。不能用candidate的K循环直接解释历史E0实测慢点；两个cost的轨迹不同，35.22ms外推并非实测固定开销或纯额外开销。不含读图的服务FPS包含预处理/传输/状态更新，不是纯GPU模型FPS。用户随后明确FPS先不管，因此不启动新的profiling、GPU裁剪、卷积切片或轻量融合改版。查询验收、身份补充等仍需由新完整结果决定，尚未实施；不因讨论就热改32项活动训练。

## 26独立CPU评分环境已就绪（2026-10-27）

从25已验收环境缓存复制同一72个conda包，逐个MD5及SHA256核对；连同原始explicit来源和原toolkit3558文件生成121794560字节离线包，SHA e70ee61f63cce3a2319a6d45e22641d96736471d1d7b29e6c2ecfa01d78506b4。首上传因26的bootstrap目录不存在失败，创建目录后正常传输；未提前启动安装或覆盖环境。一次离线create原生exit0耗时5.05秒，父CPU见证exit0耗时6.73秒，安装脚本COMPLETE且PID已退出。

前缀/data2/gb/Dual-RGBT/conda-envs/cpu-scoring，Python3.10.14/NumPy1.26.4/Matplotlib3.8.4，无Torch。fresh agent按docs/CPU_SCORING_SERVER26_20261027.md原样执行一次，native exit0；与父见证内容相同，LasHeR245/220703、RGBT210210/104706、RGBT234234/116650标注一致，7项合成评分通过，核函数SHA与25相同。独立执行为same-family/provisional，actualmodel UNATTESTED。证据CPU_SCORING_SERVER26_ACCEPTANCE_20261027.json及原生日志；不是新精度结果。两机活动NN环境、训练源码与计划未改，不读取活动权重，不提前查询训练；下一训练观察仍20:50 UTC两原句柄16720/25243。

## 精度参照的两套E0 dev轨迹（2026-10-27本地离线核对）

新完整dev不能只对失败E1作结论，还需分别对比保护E0。读取已保存完整预测/GT/路由，确认同一ee607权重、同cost、同97序列顺序/54646帧、相同float32 GT与初始化，历史dev_all882与Reference-E0仍有不同轨迹：各档全部54549个非初始化框在保存的6位小数上不完全相等；这不意味着全部定位错误，也不说明差异都很大。

| cost | 历史E0 dev PR/NPR/SR | Reference-E0 dev PR/NPR/SR | Reference减历史SR |
|---|---|---|---:|
| 0 | 79.435312/75.695667/63.175478 | 77.671580/74.181214/61.995126 | -1.180352 |
| 0.05 | 79.232365/75.573756/63.082031 | 78.333296/74.355057/62.173793 | -0.908238 |
| 0.15 | 77.882978/74.019060/61.715290 | 77.883983/74.323249/61.855838 | +0.140548 |

两套入口为run_with_crop_api.py+run_tracking.py与run_benchmark_tracking.py；后者及当前joint评估入口明确cuda.matmul/cuDNN allow_tf32=False，旧run_tracking未显式设置，历史实际有效值未记录。尚未做受控回放，不能将全部差异归因TF32、crop或某个模块；未重跑baseline精度、未启动GPU诊断。三个cost的action/modality不同帧数14140/16301/13831，说明不只是指标四舍五入。原始结果与冻结选择均不改。新完整dev并列报告相对两套E0的差值，Reference-E0与当前入口具有相同显式数值开关，但不能因此声称所有执行条件已严格匹配。

证据reports/E0_DEV_REFERENCE_IDENTITY_20261027.json，含两套config/指标与每序列预测SHA及逐序列差异；仅离线DEV已有结果，不是新模型成绩。当前正式晋级门仍按已登记协议执行；超过旧E1只是修复门，不能宣称超过保护E0，更不能以dev达到数值作为test目标完成。精度优先、FPS暂停及20:50 UTC观察节点不变。

## E0数值设置的真实短片段复现（2026-10-27，精度排查）

在25 GPU0运行独立历史E0诊断，source120经fresh SOURCE_PASS，固定ee607权重、cost0、dev顺序前3序列各11帧，cuDNN TF32按False/True/False/True切换，matmul TF32始终False；同一次进程，原生run_task COMPLETE/exit0，29.09秒。共120次在线预测和12个初始化框；没有暂停训练、更新优化器、读取活动权重或修改现有源码/训练计划，未统计FPS。新进程实际默认matmul False/cuDNN True/benchmark False。

cuDNN False的两轮均在3×11个已保存6位小数框上精确复现Reference-E0；True两轮均精确复现历史E0。两次重复的原始浮点框与路由完全相同。两设置间各序列前缀最大框坐标差0.057640、0.020447、0.079078像素；这30个在线帧的action/modality相同，因此不能声称短片段已经显示路由翻转。

结论限定为：该数值开关足以复现被检查短片段的两套历史预测差异。尚未重放完整97dev，不能宣称完整1.180352个SR点全部由TF32造成，更不能将切换设置写成模型精度创新。当前32项训练及joint评估都明确关闭TF32，保持一致、不热改、不按分数挑开关；原baseline精度未重测。E0历史与Reference继续分开列，数值设置是后续精度复现必须记录的条件。

原始证据runs/e0_numeric_replay_20261027/COMPLETE.json、runs/pipeline_tasks/diagnose_e0_numeric_replay_20261027/status.json及reports/E0_NUMERIC_REPLAY_PLAN_20261027.json；SOURCE_REVIEW与独立结果审查位于.aris/traces/experiment-audit/2026-10-27_e0_numeric_replay_{source,result}/。诊断句柄58005已actual exit0消费；下一全机训练节点仍20:50 UTC的原16720/25243，无提前训练进度轮询。

该短片段结果已fresh RESULT_PASS、0阻断：独立重算12行/132框/120次在线预测，匹配六位小数比较、重复原始框/全部路由、native命令/GPU0/exit0及120项源码。same-family/provisional，actualmodel UNATTESTED；报告E0_NUMERIC_REPLAY_RESULT_REVIEW_20261027.json。此审查不扩大前述短片段结论。

## 完整评估补存主框与请求收益（2026-10-27，精度诊断准备）

用户明确精度优先，FPS先不管；停止扩展速度优化，已有32项完整训练按原计划继续。仅修改尚未启动的评估入口，保存同帧已计算主框、所选请求gain/token成本/utility，供完整评估判断查询相对主框的即时定位收益；不额外运行主干，不改决策、模型或训练。STOP漏询、另一主模态regret和真实模板写入事件仍不能由这些记录推断。

固定历史B6250真实TRAIN片段对照完成：3序列各11帧、cost0与100（100仅为STOP准入夹具），6对轨迹；观察组50 STOP、10 QUERY（真实query均为action1，未覆盖其他query动作）。记录前后原始框、非时间路由、历史和最终模板SHA一致，权重不变，优化器更新0；native COMPLETE/exit0，34.18秒。不是新的完整精度结果。fresh SOURCE_PASS及RESULT_PASS均0阻断，same-family/provisional，actualmodel UNATTESTED。证据reports/QUERY_EVIDENCE_CAPTURE_*与runs/query_evidence_capture_admission_20261027。

三个评估协议重新封存源码80/83/83项，保持完整C、11帧准入、97dev三档cost、冻结后正式三基准规则。部署核验发现旧准入记录仍引用旧协议哈希，已用fresh审查产物原字节替换；两机CPU协议核验结果另存QUERY_EVIDENCE_CAPTURE_DEPLOYMENT_20261027.json。活动训练源码保护范围25机72项、26机119项，未读活动权重或提前查询进度。下一训练观察仍20:50 UTC原句柄16720/25243。

## LasHeR完整dev的查询收益离线分析（2026-10-27）

新增tools/analyze_lasher_query_benefit.py，输入已经正式评分的完整97条LasHeR dev报告，核对配置/预测/GT/路由/证据文件SHA与全部帧覆盖；不读取权重、不运行模型、不改评分或晋级规则。命令：CPU评分环境python tools/analyze_lasher_query_benefit.py --report <完整dev评分JSON> --output <新的诊断JSON>。输出逐帧CSV及按action/modality汇总：辅助查询相对同帧主框的IoU变化、像素和归一化中心误差减少、gain预测误差、负实际utility次数。

该分析使用连续原图xywh几何，初始帧排除、非有限或非正面积GT排除，并单列数量；收益正负容差1e-5用于避免六位小数预测保存误差。它不是官方PR/NPR/SR，也不是不查询情况下的另一条完整轨迹，更不能推断STOP漏询或未来漂移因果。主框本身很差时，低IoU的query仍可能有正收益；相反高IoU的query也可能比主框差，必须按同帧差值判断。

本地4项解析/完整覆盖/无效GT/篡改拒绝检查通过，全部为明确标注的合成几何夹具，不是新模型精度。两机CPU执行回执保存reports/QUERY_BENEFIT_ANALYSIS_VALIDATION_20261027.json。尚无新的完整dev可分析；工具只做准备，当前32项训练源码/权重/计划不变，FPS工作继续暂停。

## 查询接受质量与状态写入的开源机制核查（2026-10-27）

用户要求优先提升LasHeR精度，FPS暂不处理。针对当前candidate_tracker.py在选中query后立即执行state替换、只以每5帧confidence≥0.65决定模板写入的问题，核查了官方开源visionml/pytracking固定commit 7eb9e74bd3d40e29dbcec444902237da13de247b中的ATOM、PrDiMP源码和论文。

源码证据显示：ATOM/PrDiMP用分类搜索分数生成not_found/uncertain/hard_negative状态标志，分别控制位置状态与分类记忆；IoU/密度分支用于候选框精修或top-k拟合，没有直接作为统一框拒绝阈值。这个机制可借鉴“定位接受”和“记忆写入”分开，但不能把其阈值移植到本项目。

本项目的主路径经过Hann window，query的128/256路径分别是8×8/16×16且当前不传window；score head是中心热图，训练为heatmap focal loss，不是IoU质量分数。因此跨主/辅助区域、跨模态、跨网格直接共用0.65没有源码校准依据。固定现有权重的最小后续实验应在完整LasHeR dev后按主/local128/local256/wide及RGB/TIR路线分别用同帧GT IoU校准query接受与模板写入，不能跨路线复用阈值；几何重叠只能作诊断，不能唯一拒绝主crop外恢复。这个实验尚未实施，没有收益承诺。

研究记录：docs/QUERY_ACCEPTANCE_QUALITY_REFERENCES.md；来源与本项目源码SHA：reports/QUERY_ACCEPTANCE_QUALITY_REFERENCES_20261027.json。仅源码/论文核查，模型状态UNATTESTED，不把借鉴机制计作本项目创新。活动32项训练、源码、权重和评估计划未修改。

## 独立查询接受门原型（2026-10-27，待完整dev校准）

基于当前代码的真实问题：query被选中后在candidate_tracker.py中直接替换state，而主路径与query路径的score受window、8/16网格、ROI覆盖和模态影响，不能直接共用0.65。新增独立模块dual_rgbt_global/query_acceptance.py，只按主模态与请求路线保存8组阈值，分别区分状态接受和模板写入；当前未接入tracker、训练或任何评估入口。

本模块是待完整LasHeR dev后校准的最小实验接口，不是默认阈值、不是新精度结果，也不承诺收益。校准时需要按RGB/TIR及local128/local256/wide路线用同帧GT IoU选择阈值，再做真实闭环dev；不跨路线复用阈值。合成单元测试4项本地通过，代码已同步两台服务器。

## 查询接受门的LasHeR dev校准工具（2026-10-27）

新增tools/calibrate_query_acceptance.py：读取已完成的LasHeR 97条dev评分产物、trace/evidence/prediction/GT，按真实RGB/TIR与local128/wide路线拟合query confidence阈值。它只做同帧离线校准，不改预测、不改权重、不选择test，也不代表闭环精度。阈值冻结后必须重新运行tracker，才能观察query拒绝对后续模板和轨迹的影响。无query样本的路线明确标记NO_QUERY_ROWS。

两个校准合成测试和接受门测试通过；25/26服务器只运行CPU测试。当前32项训练、活动评估计划和原有模型代码未改，FPS仍暂停。

## 三个精度优先核心模块的贡献边界（2026-10-27）

当前统一定义为：目标条件主模态调度；同状态定位收益驱动的辅助区域请求；独立坐标局部读出与安全状态更新。第一项对应相对quality训练，第二项对应真实执行gain与STOP/候选比较，第三项对应局部ROI执行及待验证的query接受/模板写入分离。

已有E1失败结果说明减少辅助调用不能自动提升精度；ATOM/PrDiMP源码核查说明分类定位、框质量、状态和记忆门控不能直接混用。本项目的接受门和校准工具均未接入活动tracker，待完整LasHeR dev后验证；当前32项训练、评估计划和权重不改，FPS暂不作为晋级条件。完整贡献边界见docs/CORE_CONTRIBUTIONS_PRECISION_FIRST.md。

## 最终边界影响训练与固定随机评估（2026-10-28）

本轮边界影响训练已经按“两次相邻严格下降即停止”规则正常结束，没有中途改动网络、数据、优化器或活动训练配置。E0 作为保留基线，后续 E1–E4 仅用于验证当前联合质量／收益训练目标是否能改善精度。

| 版本 | 训练节点（更新／轨迹完成） | 结论 |
|---|---:|---|
| E0 | 5677 / 4920 | 当前保留最优 |
| E1 | 5675 / 4882 | 相对 E0 下降 |
| E2 | 5674 / 4887 | 相对 E0 下降 |
| E3 | 5673 / 4864 | 相对 E0 下降 |
| E4 | 5668 / 4856 | 触发停止规则，未产生新精度收益 |

这些节点不是 PR、NPR、SR。所有候选均使用相同的 SOT epoch14 初始化和三模块宏观结构；最终保留 E0，不能把 E1–E4 的 loss 或完成计数写成测试成绩。

随后进行了一次预先固定的同输入、同随机数协议评估，用于排除“评估随机性导致下降”的解释。相对于同一 E0，首个 E1 快照为 `-4` 次更新、`-40` 个轨迹完成计数，终止 E4 快照为 `-7` 次更新、`-70` 个轨迹完成计数。该结果说明固定评估随机性不是本轮下降的主要原因，但仍只是训练状态的配对审计，不等价于完整三数据集测试，也不能据此宣称某个新模块有效。固定评估归档 SHA 为 `706f04cea5490f7145082aa34f30ad1db0a6f4a9438921f315b09fa6de8e5e07`；E4 训练审计归档 SHA 为 `ea3404894cf2c1af7d6314c141b1ec967793f0f9937c7eb5ff6bd6b41131d7cf`。

当前精度结论：联合质量、GainPair 与边界影响训练没有超过保留的 E0，不能进入新的主结果。LasHeR `78/72/62` 阶段目标仍未达到，三项各提升约 6 个百分点的严格目标也仍未达到。FPS 优化按用户要求暂缓，后续评价优先级仍是 LasHeR 精度。

后续只从保留 E0 出发提出新的、可单独验证的精度假设；不继续重复同一边界影响配方，也不将 E1–E4 的失败结果覆盖 E0。正式新方法仍需完整 LasHeR dev 选型后，再冻结配置进行 LasHeR、RGBT210、RGBT234 三数据集测试。

## Query 验收校准的证据缺口（2026-10-28）

本地核对了历史 LasHeR dev 运行目录 `reference_e0_lasher_dev_cost0p0_shard0_20261015` 及对应 E0/E1 指标报告。已有目录包含 prediction、groundtruth、trace 和指标文件，但没有 `evidence/` 目录，也没有每帧保存 query 前后的候选框与 query confidence 对应关系。因此，`tools/calibrate_query_acceptance.py` 不能直接对这些历史结果执行闭环阈值拟合；缺少证据不是脚本错误，也不能用指标文件反推。

当前 `dual_rgbt_global/query_acceptance.py` 和 `tools/calibrate_query_acceptance.py` 仍应视为 dev-only 原型。要验证查询验收是否提高 LasHeR 精度，下一次完整 dev 运行必须同时保存主框、query 框、confidence、route、action 和模板写入事件；然后只在 LasHeR dev 上拟合阈值，再用同一 tracker 重跑完整 dev，最后冻结后才允许进入三数据集 test。当前没有因此启动新训练或测试，也没有把阈值原型计入结果。

## E0 LasHeR-dev 失败片段分解（2026-10-28）

基于保留 E0、cost=0 的完整 97 条 LasHeR-dev 结果，新增了仅用于诊断的 `reports/E0_LASHER_DEV_FAILURE_SLICES_20261028.json` 和同名 Markdown。按序列 SR 排序，最低片段包括 `highright2ndboy`（SR 1.307）、`ab_pingpongball`（4.509）、`blackbag`（4.580）、`dotat43`（5.074）、`bikeboy128`（5.804）、`girl\`sheadoncall`（5.935）和 `AQblkgirlbike`（9.070）。

这些片段能帮助后续优先复盘，但汇总指标不能说明失败是候选覆盖、主模态路由、query 替换还是模板更新造成的。下一次 LasHeR-dev 运行必须保存逐帧 evidence，再对这些序列做主框/query 框、同状态收益和模板写入分析；本次没有据此修改模型或宣称精度提升。

## 低 SR 序列的逐帧失效形态（2026-10-28）

对上述最低 SR 的前 10 条序列，使用保存的 E0 prediction/groundtruth 做了离线逐帧 IoU 分析，结果写入 `reports/E0_LASHER_DEV_FAILURE_SHAPES_20261028.json` 和同名 Markdown。该分析不运行模型，也不是新的正式指标。

主要现象是：多数序列在前几十个有效帧内就进入长期低 IoU 状态。例如 `highright2ndboy` 首个 IoU<0.1 出现在第 18 个有效帧，之后低 IoU 比例 97.713%、最长连续段 769；`dotat43` 第 9 帧开始，低 IoU 比例 99.032%、最长连续段 921；`belowrightwhiteboy` 第 69 帧开始，低 IoU 比例 81.069%、最长连续段 884。这个形态更支持“早期候选／主模态选择／搜索恢复失败”优先于“长时间模板更新逐渐污染”的假设，但不能仅凭 prediction/groundtruth 归因到某个模块。

因此下一轮精度实验应优先保留初始化后前 100 帧的逐帧 evidence，分析候选覆盖、主模态选择、query 是否替换以及模板写入；不要先扩大模板更新搜索或只调长期更新周期。

## 逐帧 evidence 保存改动（2026-10-28）

为解决历史 dev 结果缺少 query evidence 的问题，新增了无行为改变的轨迹字段。`dual_rgbt_evidence/tracker.py` 和 `dual_rgbt_global/candidate_tracker.py` 现在会记录主框、query 框、最终执行框、primary/query confidence、是否执行 query、是否发生模板写入；`scripts/run_tracking.py` 和 `scripts/run_benchmark_tracking.py` 将这些字段写入 trace CSV。

这次改动不改变 action、box、模板或训练损失，只增加逐帧审计输出；已用 Python 字节码编译检查通过。由于评估源文件发生变化，后续正式 dev/test 前必须重新生成并审核 evaluation plan/source hash；本提交本身没有运行 GPU 评估，也没有产生新的 PR/NPR/SR。

## Evidence trace 协议登记状态（2026-10-28）

新增报告 `reports/EVIDENCE_TRACE_SCHEMA_SOURCE_DELTA_20261028.json` 记录了四个源文件相对于旧 benchmark plan 的 SHA256 差异。当前状态明确为 `PENDING_SOURCE_REVIEW`：旧 evaluation plan 不能直接用于这次 trace schema 改动后的正式 dev/test，必须先重新审核 source hash 和 CSV 字段，再启动评估。本轮没有修改旧 plan，也没有用未审核的代码读取新权重。

## Evidence trace 本地 source review（2026-10-28）

新增 `reports/EVIDENCE_TRACE_SCHEMA_SOURCE_REVIEW_20261028.json`。四个改动文件均通过 AST 解析；16 个新增字段在 E0/E1 两个 tracker 返回值和两套 CSV writer 中均存在，blocking issues 为 0。该报告明确标记为 `SOURCE_PASS_LOCAL_ONLY`，不包含 GPU、checkpoint 或正式评估，也不替代独立 source review；正式 evaluation plan 仍保持待审核状态。

## E0 trace 与 IoU 对齐的新证据（2026-10-28）

新增 `reports/E0_LASHER_TRACE_IOU_ACTION_ANALYSIS_20261028.json` 和同名 Markdown，将历史 E0 trace 的 action/confidence 与 prediction/groundtruth 逐帧对齐。低 SR 片段中，query 几乎没有减少：`blackbag`、`bikeboy128`、`girl\`sheadoncall`、`AQblkgirlbike` 的低 IoU 帧 query 率均接近或等于 1；`dotat43` 的低 IoU 帧 query 率为 0.983，但平均 confidence 仍为 0.844。

这说明当前失败不只是“没有请求辅助模态”，而是“请求后仍直接接受错误结果”，且绝对 confidence 不是可靠的接受标准：部分序列中低 IoU 帧的 confidence 高于正确帧。下一步应训练或校准相对证据（primary 与 query 的框、confidence、身份和时序一致性），再决定是否替换主框；不能直接把统一的 0.65 阈值推广到所有 query 路径。

## Evidence trace R4 评估协议登记（2026-10-28）

新增待审核协议 `reports/benchmark_evaluation_plan_evidence_r4_pending.json` 及对应 `reports/BENCHMARK_EVALUATION_SOURCE_REVIEW_EVIDENCE_R4_PENDING.json`。R4 只更新四个 trace 相关源文件的 SHA256 和输出字段说明，旧 R3 plan 保持不变；R4 当前为 `PENDING_SOURCE_REVIEW`、blocking issue=1，在独立 source review 完成前禁止用于正式 dev/test。
