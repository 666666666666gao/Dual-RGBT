# 实际问题记录

| 编号 | 现象与证据 | 处理 | 状态 |
|---|---|---|---|
| E01 | GitHub直连schannel TLS失败；7897无监听；7898实际可用 | 此仓库使用现有SOCKS代理7898，未改全局Git配置 | ls-remote和clone成功 |
| E02 | 原权重torch.load报缺少lib.train.admin.local | 提供实际序列化所需EnvironmentSettings类，路径指向本项目 | CPU读取成功，epoch14，103142866 state元素 |
| E03 | conda离线clone需已不存在的包缓存，失败于EnforceUnusedAdapter | 使用conda-pack迁入项目环境；保留conda_create.log | 迁移完成，真实GPU0 M0 PASS |
| E04 | conda-pack包没有__main__入口，首次命令失败 | 按API调用conda_pack.pack，原日志保留 | 打包完成 |
| E05 | conda-unpack的env shebang找不到PATH中的python | 直接用项目环境的python运行conda-unpack | 命令成功，真实GPU0见证通过 |
| M01 | 独立审查发现首版±0.75裁剪边长的中心扰动，目标中心在裁剪外的概率5/9，热图无正峰而仍做框回归 | 缩为±0.375裁剪边长，使目标中心始终在[.125,.875]；复审后才跑GPU | 源码复审PASS，真实M0 PASS |
| M02 | 审查发现首版只保留校准集均值，无法分析已计算的逐样本辅助收益 | 保存同一sample_id各动作loss/IoU/区域覆盖CSV，以及signed收益分布、负收益比例、IoU0.5纠正/损害数 | 源码复审PASS，两轮校准和逐样本记录已完成 |

当前尚无正式测试精度或在线延迟；辅助收益诊断已完成，见下文。不能把这些环境问题解释为方法失效。

## 阶段A真实发现（1000次更新已完成）

- M03：第二轮主候选ROI覆盖GT平均RGB0.902/TIR0.870，但局部修正平均定位收益
  -0.010859/-0.009632，mean IoU由0.689434/0.623036降至0.684997/0.619012。
  证明当前执行器没有获得净辅助收益；不能据两轮短程判定局部证据方向无机会。
  固定权重诊断已完成，池化与摘要干预证据见M08。
- M04：次候选动作当前直接以次候选框作为修正基点；RGB/TIR分别纠正3/4、损害40/44。
  目标平均覆盖仅0.508/0.522。需先分解候选切换损失与修正损失，
  实测候选切换gain=-0.880998/-0.674639，修正gain=0.003180/-0.001131，
  主要损害发生于直接切框；下一版查询区域只决定补算位置，定位头重新判断最终框。
  不能用更大的收益头掩盖错误执行器。
- M05：阶段A的预览网络被有意冻结且尚未学习，预览候选平均覆盖RGB0.398/TIR0.302。
  这不是训练好的请求策略的性能证据。B需要实际训练预览并在C刷新新分布上的路径。
- M06：prototype Tracker未采用原CADTrack Hann响应权重；STOP的raw score与请求路径的
  IoU-sigmoid用同一个0.65模板更新阈值，语义不同。C/D前须明确解码并在训练校准集校准。
- M07：token_cost同时作为收益头输入和决策惩罚，不能直接改成秒导致输入分布变化；
  实测延迟成本应另存/校准，保持训练特征语义。当前不作实际提速声明。
- M08：GPU1冻结epoch2诊断800条配对记录，原生退出0；PRIMARY池化熵
  RGB/TIR=0.999060/0.998949，最大patch权重0.020178/0.020021（均匀为0.015625）。
  真实摘要置零后loss反而低0.002779/0.005314；真实与错配摘要框L1差约0.0007。
  GT前景池化相对真实池化gain=-0.001545/-0.002952（仅有覆盖的95/94样本）。
  固定MLP对摘要干预响应较小，GT池化也无正收益，不能只调整softmax温度。
  这些是表示干预与dev裁剪证据，不代表物理缺失模态或在线性能。
  处理：保留辅助局部空间网格，复用原decode/head读取主/辅证据，另做同预算A诊断。

- M09：空间版同预算1000更新已完成、原生91336退出0。第二轮PRIMARY
  RGB/TIR平均signed收益+0.093309/+0.246714，IoU 0.681044→0.713193、
  0.615028→0.691380；IoU0.5纠正4/7、损害0/0。旧池化版对应gain为负。
  SECONDARY当前纠正2/5、损害1/4，较旧版损害40/44减少。
  这是空间表示、输出规则及融合监督的系统级比较，不能拆成独立消融。
  PRIMARY仍有40%/28%负收益样本，所以不能改为每帧固定查询。
  单模态IoU较旧版略低，完整辅助与局部辅助更强；继续监测各路径平衡。
  best为epoch2/1000/devloss0.978496，并非闭环最优权重。
- M10：Hann和置信度语义已在准备源码中修复：主/辅同窗口解码、同响应峰置信度，
  源码复审PASS；32帧真实闭环检查待执行，不能称正式精度已修复。
- M11：C复用裁剪GT定位代理，GT会clamp到[0,1]，不是全图OPE真实收益。
  记录真实轨迹中心越界；只学习quality/gain，不把越界中心做预览CE标签。
  实测20条train首帧均有效，ab_moto2north0有45帧非正宽高标注；
  只跳过这些帧的监督，仍逐帧跟踪。两轮17708次track/3504更新是运行前预期。

详见POLICY_CODE_REVIEW.md、reports/stage_a_latest.json、
reports/evidence_diagnosis_summary.json及800行CSV。官方test指标仍未测量。

## 完整训练接续时的发现

- M12：扩大A完成后最优执行器变为epoch7/7250，旧B使用epoch2执行器产生标签。
  因执行器改变，不能把旧B直接接到A7上作为完整方法。处理：重新运行B，
  从A7逐批枚举生成signed收益；完成后再用真实闭环C刷新历史和Hann分布。
  旧B1000更新记录保留，不冒充当前策略。A7 PRIMARY仍有RGB45%/TIR25%负收益，
  这是继续学习是否请求的实际证据，不能改为每帧必查。
- M13：完整评价原始工具包运行于本地Python3.12，服务器项目环境为Python3.7.16。
  工具包包含`list[str]`类型注解；不为此改旧工具包或重建训练环境。
  服务器保存完整预测与每帧计时，本地现有uv环境使用原工具包评分。
  全245序列GT对照已通过；源码审查要求的分片、帧数及冻结参数断言已落实。
  这项评价执行安排不算论文创新。
- M14：真实16帧评价首个前向4.40秒，其后服务时间明显小于总均值。
  原始逐帧计时保留，不能把冷启动均值1.98FPS作为完整系统速度结论。
  后续64帧独立cProfile显示默认PyTorch32/OpenCV64线程，暖机后解码55.1ms、
  跟踪109.4ms；单线程对照39.0/96.8ms。两次不同进程、顺序运行，缓存和系统负载
  并未隔离，因此不能把差值归因为线程数或直接称为加速。默认crop_pair累计4.82秒，
  CPU裁剪与密集定位头开销应纳入后续效率优化。当前不改正在运行B的执行数学，
  先完整训练/评价，按已测约0.16秒服务时间放宽诊断节点：C60分钟、dev每组135分钟、
  test首节点280分钟；其后仅必要时每240秒查看。完整结果预计需要过夜，尚无正式速度优势。
- M15：A7保存的100个配对dev裁剪进一步表明，PRIMARY GT覆盖≥90%时仍有
  RGB38/80、TIR18/76负收益；主IoU≥0.8时平均signed收益均负。
  请求区域覆盖、主模态定位质量和辅助价值无法混成同一判断。
  GT诊断按单模态损失先选主模态时60RGB/40TIR，联合路径选择比顺序选择IoU高0.009132；
  这是校准裁剪上的机会，当前学习策略尚未验证实现。记录在
  EXECUTOR_OPPORTUNITY_ANALYSIS.md，不据此提前修改完整B/C或用GT做线上门控。
- M16：M14已测CPU裁剪开销，现有read_pair先拼接六通道，再把非连续三通道视图
  传入mean/warpAffine。独立源码审查PASS后做真实CPU配对对照，原函数不改。
  40个dev裁剪/10条序列的tensor、目标、几何逐位一致，120对crop调用均值
  46.766→37.609ms，39/40 case的候选均值更低；实际exit0，用时17.91秒。
  这是保留连续输入布局的组件机会，非内核归因证明；计时不含解码和连续数组准备。
  生产端如何避免准备拷贝及完整在线延迟/轨迹等价仍未验证，不据此热改当前C/dev/test。
  详情见CONTIGUOUS_CPU_PROFILE.md，负载、线程、尺寸顺序限制及原始记录全部保留。
- M17：同项目保存的去模块CADTrack sample_target使用整数截取、零填充和resize；
  原PreprocessorMM在GPU归一化。当前方法额外在完整两路图像上求背景均值并在CPU归一化。
  这是真实实现差异，可能抵消较少token的计算节省，尚无配对整机测量证明影响大小。
  新增独立compare_tracking_latency.py，交替两侧独立解码和各自因果状态，包含模板更新、
  初始化及CUDA同步；不修改原基线、不重测其精度。须先源码审查，再在正式任务结束后
  以同一最终冻结权重/成本测量；准备源码不等于已得到速度结果。
  该原生系统对照不能将差值单独归因于主干token，完整延迟与主干机制消融分别解释。
- M18：完整B监督轮均值下降，但其已结束32帧训练检查有31次局部请求，STOP仅1帧。
  原始路线记录证明此短轨迹使用了按需路径，却不支持多数帧会STOP的假设；
  32帧均选RGB仅是单序列现象，不能称全局模态塌缩。当前C继续真实闭环刷新，
  等完整dev/test统计各模态/动作频率、token与延迟。历史分析和损失曲线保存于
  STAGE_B_TRAINING_ANALYSIS.md，不根据短检查提前改当前训练。
- M19：M14中全图背景均值只服务越界填充，重建已有40case确认25图内/15越界。
  在独立CPU helper中保留同affine/插值，以带一像素余量的逆映射四角判定是否需要均值。
  source复审PASS后真实80case三臂均逐位相等，47跳过/33保留；原/连续/条件均值
  43.850/35.047/16.817ms，每臂240次轮换计时，实际exit0/34.595秒。
  保留均值组增加guard后34.275ms，相比纯连续33.584ms略慢；不隐藏该代价。
  解码和数组准备未计入，当前不接入活动训练/评价。它是CPU工程机会，不独立称论文创新，
  未来须计入生产API复制、完整在线延迟和轨迹一致性。详见INTERIOR_CPU_PROFILE.md。
- M20：完整C实际exit0后CPU核对全部3504监督行，567行(16.18%)的GT中心在搜索crop外。
  两轮分别278/1752、289/1752；quality回归loss图外约3.114/3.024，图内0.306/0.258，
  是条件相关，不作因果归因或PR失败比例。当前loss先将GT转xyxy再clamp角点到[0,1]，
  因此收益仍是裁剪代理，不是原始全图误差；EXPAND仅增大全部辅助token，不扩大原图视野。
  后续用完整dev检查覆盖与恢复、核对全图框误差收益，必要时再实现扩大辅助搜索。
  当前全dev成本0/0.05已启动，不热改权重、成本或11项源码；完整指标待完成。
  详情见STAGE_C_COMPLETE.md，所有统计限定在监督帧，不冒充全部17708次track调用分布。
  进一步拆分：8/20条出现越界，三条贡献485/567（85.54%），不能假定均匀存在于所有序列。
  独立CPU完整dev诊断入口已准备，只能从已结束记录统计覆盖/原始误差/路由和相邻ROI重入；
  未保存反事实动作框，不能靠最终预测重建辅助收益。完整数据执行待dev回收，正式评分入口不改。
- M21：A/B裁剪增强将GT中心限制在名义归一化范围[0.125,0.875]，不会产生中心越界状态；
  C的真实闭环已出现567个越界监督帧，但仅覆盖20条、train非初始化帧的1.93%。
  原生CPU采样范围审计exit0：合并A54,000/B50,000个样本均涉及全部882条，唯一搜索帧
  为47,079/43,986（可采样目标10.32%/9.64%），不能将序列出现次数当作全帧遍历。
  全882条两轮C预算917,106次track尚未启动。实际数据与限制见SAMPLING_SCOPE.md。
  针对搜索视野核查官方PySOT固定源码：SiamRPNLT在lost状态确实同时扩大源图裁剪和模型输入。
  可借鉴真实源图范围控制，不能照搬大输入/置信阈值就称收益驱动创新；稀疏外部查询的坐标与训练
  仍未实现。见SEARCH_COVERAGE_REFERENCES.md；当前完整评测和活动11份源码不改。
- M22：M19的kernel-only计时排除了连续数组准备，不能直接代表现有六通道API的成本。
  新独立wrapper将两次复制放在计时内，经源码审查后单次真实CPU运行exit0/33.148秒。
  同80case输入tensor/目标/几何逐位相同，60搜索case完整RNG状态相同；每臂320次平衡计时。
  API裁剪调用均值44.066→24.298ms（44.8595%），P95 70.858→56.617ms，66/80case均值更低。
  保留背景均值组40.851→40.796ms，几乎无优势；不能把旧kernel16.817与新API24.298差值
  全归因复制，更不能称完整系统提速。生产接入和真实在线轨迹/延迟仍待完成，详见CROP_API_PROFILE.md。

- M23：首两组完整97条dev已结束，成本0.05的PR/NPR/SR=78.9065/75.2723/62.7969，
  是dev达标，非完整test达标；不与test baseline直接比较。其6,346/54,289有效跟踪帧
  的目标中心在搜索crop外（11.6893%），95.11%仍请求辅助，最终原始IoU均值0.000002800，
  平均重型token612.805高于图内540.180。成本0的越界比例13.7118%、IoU0.000035751。
  这是完整轨迹中覆盖与资源错配的条件证据，非辅助调用有害的因果证明；仍需真实扩大源图范围
  并使用全图定位收益监督。相邻ROI重入不称准确恢复。详见FULL_DEV_FIRST_RESULTS.md。
  token均值从599.687降到549.221，两组整机均值均约131.83ms；当前无延迟改善证据。

- M24：原condaCPU实际模型构造/strictload清点exit0，无CUDA或NN前向。baseline总参数
  103,138,437，ours 103,374,215，增加0.228603%；
  Preview/quality/gain新增235,778，C trainable221,442。原baseline本就共享主干，
  当前全部旧backbone/decode/head权重保留，不能宣称参数减半或总参数减少。
  参数复用不等于少调用；当前纠正仍再用17,093,253参数的decode/head，实际时耗待配对测量。
  计数是注册Parameter元素、buffers排除而非底层存储/显存口径；详见PARAMETER_INVENTORY.md。

- M25：三组完整 dev 已结束并在 test 前按预定 SR 优先规则冻结成本 0.05。
  成本 0.15 的 PR/NPR/SR=78.310464/74.689270/62.318010，平均 106.050680 ms；
  与前两组并行运行条件不同，尚不能声明策略带来的整机速度优势。
  外部中心 6029/54289（11.1054%），原始 IoU0.000065752；候选均位于原源 crop 内的问题仍在。
  原始冻结 SHA、真实两个测试进程与完整分片范围已记录，尚无 test 正式成绩。
  见 FULL_DEV_FIRST_RESULTS.md 与 reports/full_test_startup.json。
- M26：局部辅助 193 token 中 129 为双模板及 CLS，模板重复执行可能限制 token 稀疏的收益，
  尚无分项整机测量证明其占比。已实际核查固定 commit 的作者 MixFormer 缓存源码，
  不对称模板注意力才使重型 K/V 独立于当前搜索；本模型是双向图，不能直接等价缓存。
  该机制尚未实现，不改正在执行的测试或封存的全 882 训练配置。
  见 TEMPLATE_CACHE_REFERENCES.md；GitHub、源码 SHA 和配置限制均已记录。

- M27：全图均值仍是当前crop实现的实际组件优化机会。sum候选CPU原生693753真实exit0，
  80case全部tensor/target/geometry/RNG相等。三臂均值原38.5716/six21.1497/sum12.3381ms。
  33个保留均值case由six35.9524降至11.9604；47个跳过均值case反而10.7563升至12.6034。
  只35/80 case优于six，但混合平均减少41.6628%；不是全场景统一提升。
  33case旧40.8ms指完整API而非mean单算子。未接入当前模型或封存训练，NN准入/整机收益待验证。
  见SUM_CROP_PROFILE.md/REVIEW.md及原始480调用每臂记录。
- M28：此前同设备延迟对照只覆盖10×64前缀，无法代表完整轨迹分布。最小添加--full-dev，
  保留GT仅初始化与双侧独立预测状态、交替解码及同步，支持97条/54646帧完整配对。
  fresh SOURCE PASS/0阻断，17seal未改；原conda CUDA隐藏CLI PID705573真实exit0。
  NN实际构造/准入/配对延迟尚未执行；等全882训练和最终dev/test结束后使用同最终freeze。
  完整dev速度不冒充baseline完整test速度，不重新评分baseline。见FULL_DEV_LATENCY_PROTOCOL/REVIEW。

- M29：完整已结束dev cost.05的GT geometry诊断PID49400纯CPU真实exit0。
  46782帧不截断、2019帧部分截断、5488帧clamp后面积零；后者5268帧仍查辅助。
  这是dev状态下假想GT目标，不是实际训练标签；当前C没有保存动作框，不能证明收益符号翻转。
  后续必须以同训练状态下的真实执行框比较全图定位收益，不能只删除GT clamp就宣称完整执行收益。
  见CLIPPED_TARGET_DIAGNOSTICS/REVIEW及全部逐序列统计。
- M30：有有效GT的6430次模板更新事件里1053次rawIoU<.1，meanconfidence.818355；
  460次GT中心在源crop外、593次在内部。响应强不等同于定位正确，也不能把内部错误都归因于干扰物。
  事件数不是模态写入数，未证明写入造成后续失败。实际核查KeepTrack作者仓库固定commit
  7eb9e74bd3d40e29dbcec444902237da13de247b，候选object_id关联与更新flag/样本权重是不同判断。
  该机制未移植，不能宣称为我们新贡献；见STATE_UPDATE_REFERENCES及七文件SHA provenance。

- M31：M29的dev几何诊断不能证明实际训练动作收益改变，必须保存同训练状态下的动作框。
  新诊断用固定B、seed2027首20条一轮完整闭环，预期8854track/1752标签/14016对收益，
  同时比较原clipped代理、未截断提案误差及Tracker margin10执行框的完整目标误差。
  SOURCE R1发现manifest['test']不存在，sanity在模型构造前将KeyError；仅改为真实test_sequences，
  R2 PASS_SOURCE_ONLY/0block，原17source不改。真实远端CUDA隐藏CLI与本地CLI exit0，不代表NN通过。
  独立本地queue owner47240单次启动，实际birth/argv匹配，首04:30:50等待原参考COMPLETE后GPU1，
  先10帧/2标签输入sanity，再原B正式诊断。0optimizer/0新权重，尚未启动NN或产出收益结果。
  队列首次回执因PowerShell false字面量错误未正确写入，修正metadata后保留同owner，没有重复启动。
  raw sign flip未滤近零，后续须看幅度，不能直接视为有效策略改变或完整跟踪改进。
  见REWARD_DIAGNOSTIC_PROTOCOL/REVIEW与冻结plan/实际部署和启动回执。

- M32：M23的范围不足不能仅靠增加原crop内tokens解决，也不能借鉴整段视频的未来预览做在线跟踪。
  已核查官方Uni-AdaFocus固定88464883...源码：输入图像选可变源范围、固定96×96采样后才重型编码。
  当前ActivityNet分支预览输入是224，glance_size=96未用，不能从参数名虚构低清执行；
  局部重型路径actions detach，空间策略另由廉价feature-grid辅助分类监督学习。
  本模型辅助scatter回16×16主网格，外部候选需要不同坐标与有效定位路径及重新训练准入。
  实际7文本hash/native git exit0 provenance已保存，未移植/导入/执行该模型，非已经成立的新贡献。
  冻结B动作收益的纯CPU分析器SOURCE PASS/0block、local CLI exit0，正式CSV未产出、analysis NOT_RUN。
  后续看收益幅度与同成本行动分歧，再确定模块二修正；不改当前19项封存/参考test/全882计划。

- M33：首份完整245序列/220703帧test真实结果为73.916231/70.398889/58.587111，
  对用户baseline数值差+1.176231/+1.238889/+0.747111个百分点，仍未达78/72/62。
  该参考C只训练20条×2轮，dev达标不能代替最终test，更不能称全882训练已完成。
  两测试原生任务exit0，原controller.ExitCode没有取到保留null；被动等待器exit0不替代它。
  完整时耗105.942284ms、P95 163.019947ms、9.439102FPS；无同条件baseline速度或稳定性证据。
  原全882计划在test前已冻结，现GPU0正式TRAIN_ALL_882；不得用这份test挑阈值/预算/权重。
  详见REFERENCE_FULL_TEST_RESULTS与原生终态、全数组fresh审计。
  fresh CPU核验actual exit0，全部245GT/预测/trace/指标曲线相等，确定性PASS/整体WARN。
  原工具包52序列/1260帧的GT任意坐标≤0约定保持；原JSON字节SHA也保留，不静默改分。

- M34：全882与收益诊断的SOURCE/CLI检查此前不等于真实NN准入。现在原队列已各自通过
  真实sanity并单次接续正式任务：GPU0 10track/2update/257frozen/reload一致，
  GPU1 10track/2label/16pair/265frozen/0update/两次原B预测一致，原生exit0。
  04:39正式主进程命令、CUDA、实体卡与19项封存匹配，尚未读取正式loss/收益或完训结果。
  全C首诊断08:29:55 BJT，收益诊断05:01:57 BJT；不提前轮询，也不重启原owner16936/47240。
  见reports/c882_reward_formal_startup.json；正式训练范围/优化器冻结和最终test仍需终态核验。

- M35：完整冻结B train20诊断已经actual native/owner exit0，8854track/1752状态/14016对，
  265张量不变、0optimizer。实际CPU分析93a7b3 exit0：197对收益符号翻转中GT截断191对，
  映射/clip执行6对；总平均绝对收益差.003884196，小于当前预测MAE.149812868。
  成本.05下56/1752诊断oracle动作不同，旧oracle执行regret.000164392、真实策略.068398766。
  这是原B访问状态的当下配对，不能认为修改标签必能解决主要误差或是长期改进上限。
  完整冻结/实际任务终态与三文件字节哈希见reward_diagnostic_native_completion；不改GPU0封存训练。

- M36：真正覆盖不足在训练状态得到动作反事实证据。258个中心在源crop外的状态中，
  全部2064对主/辅助动作框的原图IoU均精确0，但实际255状态仍查询、178使用完整辅助。
  平均辅助tokens323.220930，高于四角在内237.575779；当下执行loss改善.023964753，IoU改善0。
  不能把靠近目标的误差下降称准确恢复；需要真正外部源区域及可输出外部坐标的执行路径。
  当前无重叠只是此冻结B和现有候选的诊断，不是新路径的上界，不由test错误调参。
  优先实现编码前外部候选读取与实际全图收益连接，沿用已核验作者源码作参考，独立新实验准入。
  详见TRAIN_ACTION_REWARD_RESULTS及逐状态完整CSV；原全882C训练继续，目标仍ACTIVE/UNMET。

- M37：新独立query坐标执行器中，zero padding的双线性采样在主源crop外半patch仍可能混入边缘。
  SOURCE R1标准库真实反例exit1：main=[0,0,256]、query=[244,0,256]、8网格的越界点(260,16)得到.25。
  仅加实际源半开覆盖mask后R2 SOURCE_PASS/0block：48边界/16角点/1280覆盖点及full16三几何恒等通过。
  GPU张量/真实前向/梯度/更新/重载未执行，非整机提速或恢复证据。size保持query归一化语义，不另乘16/8。
  readout源码2248b1af...在新包，旧17不改，见EVIDENCE_QUERY_EXECUTOR_PROTOCOL及独立审查。
- M38：新GPU准入driver R3在NN前找到两个真实SOURCE blocker：list anchor不支持crop_pair的box切片除法；
  geometry的list(np.float32 array)无法默认JSON落盘。仅改为np.asarray(dtype=float32)与array.tolist，
  不改旧crop_pair、不加JSON encoder/fallback。R4 SOURCE_PASS/0block，实际CLI exit0。
  GPU1单次M0 wrapper1224324原生exit0，10真实帧/四路径/4更新；168冻结state逐位不变、58状态变化，
  双mode同区域full16与原correct逐位相等、strict重载预测逐位相等。
  此为执行器准入，未形成完整查询策略/在线tracker/正式模型训练或精度与延迟证据；M0权重不用作初始化。

- M39：基于M36，新增完整独立区域策略与A/B/C代码，不修改原GPU0全882封存。
  两路64px主预览选主模态，另两路64px较宽预览覆盖2倍main源边长；四请求在aux重型编码前生成。
  LOCAL_EXTERNAL可以来自旧main外，WIDE实际扩大源边长；局部重型128/193tokens，WIDE256/385tokens。
  收益监督改为同一margin10/float32执行框的原图IoU差，不把无重叠的距离改善当成准确恢复。
  A只更新共享decode/head，B训练preview/两个控制头，C冻结执行器及preview刷新完整882真实闭环。
  new query-gain替换不再使用的旧loss-gain头；参数/速度优势需实测，当前无新正式精度或提速证据。
  fresh SOURCE_PASS/0block，25源/两个local helper/882范围/955维/GT边界已独立核对；
  requested astra/max、actual routing UNATTESTED，同family/provisional，仅SOURCE，不认证NN或性能。
  独立计划e4aae856...保留审查前PENDING记录，当前准入依据单独SOURCE_REVIEW机器gate；
  GPU A10准入实际exit0，80样本/十组合各8、168冻结/58状态变化、7项strict重载预测逐位一致。
  B2/C2也已actual exit0，251/257冻结state不变、14/8状态变化，7项strict重载一致；C实际10帧/2监督。
  新正式A已GPU1单次启动wrapper1309002/child1309003，从原正式B而非准入权重开始，6250更新待完成。
  只读startup身份通过、25源/远端plan与实际config一致；首loss/ETA节点07:28:56，不提前查询。
  三模块、精度和速度效果仍须正式训练/全dev-test/同预算对照，不能由准入通过得出。

- M40：2倍搜索源不等于全图可查询。旧C20完整97dev名义覆盖独立复算PASS：
  54289有效非首帧中2S中心越界3001/目标框零交集2545；外部local全64候选并集2.25S为2508/1986。
  保存框六位精度近似，主/次候选和最终offset未限于源ROI，不能写成全部动作或新轨迹性能上界。
  新增whole-image256共享小CNN32grid/global原像素local和wide请求，不先付辅助完整ViT成本。
  这是待实际SOURCE/M0/正式训练/收益验证的设计；不能把新增全图覆盖等同恢复提升。
  不新增无证据scalehead；宽查询也以独立global候选为中心，按真实pixel IoU收益监督。
  旧source25/e4计划保持，正式A6250已actual exit0、164可比较冻结state逐位相同；旧2S正式B/C未启。
  CPU收集器误用旧模块命名的首次断言失败原样保留；修为decode/box_head后核验实际exit0，NN未改。
  三个机制继续围绕调度、预计定位收益请求、独立区域共享修正，不由方框名称宣称独立创新。

- M41：全图SOURCE通过后核验实际部署状态。及时发布已包含global5源，原仅适用“未部署”的helper未调用。
  新只读验证器核验已发布30源/plan/B/正式A/manifest并在CUDA隐藏环境执行--help，实际exit0，未构建NN。
  fresh 52项源码/标准库检查通过，0block，原36+增量1 inputs和77原产物封存，同family/provisional。
  独立checker自身一次SyntaxError保留，实际修正v2退出0，不是生产NN失败。
  GPU1串行A/B/C真实准入已一次启动；源码通过和隐藏CLI成功仍不能作为模型有效/正式训练完结证据。
  R3复核raw尺寸容量和名义投影counts算术通过，但不是实际IoU/可见分辨率/scalehead必要性证明；不据此加模块。

- M42：全图路径真实准入闭合，A/B/C各actual exit0；168/251/257冻结state不变、58/14/8状态真实变化，
  完整7-field strict重载逐位一致，C同访问状态所选反事实重现真实box。只证明执行正确，不能声称新指标达标。
  新正式A已GPU1一次启动1493008/1493009，5×10000/6250，actual config严格初始d198而非sanity；
  原B仅是factory。输入状态、30源码与GPU身份实际startup验证，通过后按09:21:46首节点看loss/吞吐。
  充分训练后必须检查廉价global候选召回和实际查询收益，学习门控不能代替候选/修正器的有效性证据。
  原GPU0完整882 C继续，不改17源或抢占原评价所需资源；最终78/72/62与整机提速仍未完成。

  M42更新：全图正式A首计划节点actual RUNNING975/6250，已写loss全部有限；初步执行正常，
  不是最终指标或完训证据。以实际1.593603更新/秒将必要near-end节点设为10:12:08，原observer已退出0。


- M43：完整global B后的固定区域内容实验已终态，31536比较/265冻结/0更新，独立审计259项PASS、0block/实质WARN。
  主local正确aux像素相对空白+0.062915，但相对STOP只有+0.009589；GTlocal相对空白+0.150052是位置prior条件下的证据。
  外部local/wide整体损害STOP，wide TRUE略低于空白，不能用GToracle或单步诊断宣称三个模块有效/最终成绩提高。
  239 native main-centreoutside中的外部ROI目标中心覆盖11/49；query可独立复算，main缺原框/搜索几何仍native。
  原SSH transport和parent CPU receipt不完整，审计保留provenance WARN；真实native tool result和独立复算receipt仍存在。
  历史SOURCE_PENDING协议保持，不冒充当时已通过；当前事实由新实际completion/source/sanity结果与结果文档说明。

- M44：一次CPU图像头/原32grid几何计算，GT最近有效格点在原local/wide预算均覆盖239/239 outside中心；
  local面积覆盖.995888、wide1.0，仅GTprior几何诊断，不能作为部署输出或跟踪上界。
  原global候选差应优先检查排序/表征训练，而非继续扩大ROI。256预览下outside目标中位宽/高9.6/15.733px只记录尺度。
  现有descriptor已经initial-template central4x4，不能错误声称整个模板背景或recent-template污染已证明是原因。

- M45：针对排序问题新增现有preview-only unit-mass Gaussian spatial CE warmup方案，31source/plan36aeeda7…冻结并独立SOURCE审查中。
  仅现有CNN六张量、原259冻结，固定input256/grid32/共享网络/descriptor/ROI/角色，不加推理参数；
  来源包括开源SiamFC正负损失归一化和OSTrack target-mask机制，只作设计借鉴，不能称其复现或新贡献已有效。
  完整882/50k/6250以及same1752 before-after计划待真实M0后执行；localize权重不作完整C/测试模型。
  共享CNN变更后必须重新校准质量/收益头，旧B收益标签不能被直接当成更新后有效监督。

  M45准入更新（16:13）：SOURCE0block/56稳定input；真实GPU1 sanity16pairs/2update/259冻结6变化/Adam2/7field严格reload PASS，actualexit0。
  before两state实际复现旧proposal，只M0不是改善证据。正式882/50k/6250从fullB b116单次启动2171652，first16:21:27。
  新SSH stdout/stderr/transport确实保存；代码source31/旧30不改。stage=localize，质量/gain头虽冻结但输入改变，B/C/devrefresh仍需。
  新长globalC宜等原GPU0完整参考队列终态和所需两卡释放，不能让原自动dev/test因GPU1忙而失败。

  M45正式首节点（16:21）：950/6250有限loss，mean1006.18118474，rate1.55192/sec；正式before完整1752 peak复现已经过门禁。
  预计17:18结束/17:13 nearend，只有原observer72715/54096及join682，时间未到不读；未把train loss当coverage变好。
  CPU coverage分析器已实现/help pass，正式数据未到，不伪造analysis/result；rank/margin仅native-derived，header/GT/CSV完整SHA绑定。
  原GPU0计划节点已61250/182182/last100 .202766 finite，next20:30；本地实际owner身份核对后捕获，无额外remote查询。

- M45正式终态：full882/50k/6250实际17:05:27 exit0，6变化259冻结；CPUcoverage误差0与独立WARN/0block审计闭合。
  local256→348但wide760→720；outside239 local11→25/wide49→55。nearestGT rank提升不等于argmax覆盖，排序未解决。

- M46：冻结尺度对照实际18:46:28 exit0，1752state/3504compar/265冻结。T64→T32 outside local25→34，
  但overall local348→324/wide720→711，不能以子集改善掩盖整体下降。sameglobal256/grid/ROI，RF context也改变，未隔离scale。
  CPU7969fd exit0复算全部querygeometry且T64baseline字段复现；独立结果审计323checks/73inputs稳定，完整性PASS/整体WARN/0block。SiamFC fixed39572051源码借鉴已记录。

- M47：当前单cosargmax候选覆盖不足。新增最多1/2/4/8分散峰的冻结几何机会诊断，SOURCE_PASS/0block，真实M0/正式实验仍待执行，无NN结果。
  GT只用于事后best single-ROI机会，不查询全部重型区域；保存私有score map便于CPU重建。原gain4-query硬编码未改。
  需先判断候选覆盖潜力，再另行训练收益排序/刷新控制监督；不得把oracle集合coverage当真实闭环提升。

  M47 actual closure：SOURCE/M0通过后formal19:18:24 exit0，1752states/265冻结/0update-backward。完整score map允许CPUd17ee1重建greedy与geometry，error0。
  K1/2/4/8 outside wide55/80/130/184，local25/36/63/113；仍非真实恢复或部署收益。23allstates少于8，histogram必须保留。
  原collector326ddf exit1漏改旧comparisons filename；正确collector1cd667 exit0收回同task，无NN重启。错误/原stderr/tool对象保存。
  一行publisher准备命令bacfed exit1是PowerShell quoting造成Python SyntaxError；未修改publisher或运行NN，随后apply_patch纠正。仅原工具transcript，未补造transport。

- M48：新增明确post-hoc CPU random机会对照，seeds2027–2036事先冻结，same grid/NMS/ROI，双方每state/K/seed同实际prefix数。
  actuald2797c exit0/noNN：K8 outside CNNlocal/wide113/184，random45.5/116.3；all1171/1526 vs740.6/1295.7。
  该集合优于random几何机会，值得训练单query收益选择；候选数相同不等于实际aux执行成本相同。当前4-slot gain尚未改，正式指标/效率/核心对照仍待。

  M47/M48独立audit闭合：限定geometry PASS/0block，181evidence+124423NP+141claim/seal checks，115sealed inputs稳定。
  全greedy/Top1/score-map选择与10random-seeds/70080pairedprefix复现，maxerror0；329prefix较短/34random列表少于CNN均实际数相等，未证明执行成本相等。

- M49（planned）：候选coverage不等于实际定位收益。下一步用旧full-B primary-only轨迹保留状态、另一个fitted-preview模型提候选，
  复用同primaryfeatures执行每个候选actualauxrawROI，监督signed image-clipped IoU gain。先证动作有用，再训单query选择器。
  实现/SOURCE/M0/formal均未开始；旧content28.926min作ETA依据，新估30–35min/first30min/240sec。不要把计划写成结果或NN已启动。

  M49 actual admission（20:50）：utility driver已实现。R1 SOURCE_FAIL/1block在NN前发现两处candidate_predictions的modality参数语义错误：
  该函数直接选auxiliary template/image，query reader则接收primary后内部取1−primary。仅将两个candidate调用改为1−primary，保留query调用。
  R2 fresh SOURCE_PASS/0block，21348 checks/87坏输入拒绝/91inputs stable，source34/plan36dfb234/driver8a067918封存；旧R1 FAIL不覆盖。
  GPU1 actual M0 COMPLETE/exit0：10track/2states/36记录query+2strictreload probes、两个265冻结、259共享相等/6CNN不同、0optim/backward、8fields逐位重载。
  正式wrapper2572336在20:47:16单次启动，预期1752固定训练状态/31384queries，first21:17:16、原observer53561/local42632；结果尚未取得。
  候选几何潜力仍不能代替实际收益；正式诊断也不等同部署selector/闭环上界/完整PR-NPR-SR/延迟。

- M50：CPU checker首次actual0c7c54 exit1，是其裁剪运算精度与真正执行器不一致。
  证据：regions.execute_boxes将float32 box.tolist()传给lib.utils.box_ops.clip_box，内部Python float加减后才转回float32；checker原先直接在np.float32 scalar上加减。
  仅改CPU clip_box的四个scalar先float()，不放宽容差、不改GPU/producer、不重跑NN；actual9bfcde exit0，36条clip/IoU/mainIoU/coverage/delta最大误差均0。
  原失败真实tool对象与checker/analyzer前版本保留private；公开仅修正源码与真实通过报告，不能补造首次失败的stdout或transport。

- M51：原全882 C第四计划节点仍正常，82625/182182，last100loss.0698310860526/quality.0545182014874/gain.0153128851810均有限。
  原20:30:21节点写状态；20:32:24本地captured不是精确GPU查询时戳，原owner16936 birth/argv实际匹配，无额外SSH/NN查询或重启。
  下一原节点10-15 00:30:24.031239 BJT。该进行中C不是新global C；最终dev/test与目标仍未完成，不以训练loss代替正式指标。

- M52：完整utility actual native COMPLETE/exit0，1752states/31384query、265各自冻结、0updates/backward；CPUcc2bad复算clip/IoU/mainIoU/coverage/gain全0误差。
  outside239主IoU0，wideTop1 .040829→best8 .337778；best8有123个gain≥.10/100个IoU≥.5。localTop1 .033736→best8 .230184。
  不是coverage理论机会，而是同state下真实raw辅助query执行框；仍是GT oracle，不是可部署策略或闭环上界。
  all1752强制best8全图local/wide仍平均signed gain负值−.064311/−.060906，因此需要同时学习何处有益与何时STOP。
  最大GTcoverage宽查询IoU.234769低于按实际IoU选择.337778，不能把coverage标签直接当作query效用标签。
  独立审计524392checks，确定性PASS/语义WARN/0block，同family provisional；17真实transport receipts、590preservation seals及source34未改。
  M0 native20:35:37结束、20:38:20收集，旧HANDOFF时间是观察时刻；formal21:21:03结束、总wall33.79min不是在线latency。

- M53：现gain的4请求硬编码限制候选选择。新增独立candidate policy/tracker/sample-B driver，用显式family id把Q泛化、复用955-input head，最终只重型编码一个所选ROI或STOP。
  新B source37/plan9b847f90/driverf8210962，SOURCE_PENDING、M0未执行；不能声称新模型/参数量/速度或最终指标已验证。
  计划8quality/gain张量变化257冻结，32pair/4update/12field reload+10frame causal重放准入，formal5×10000/6250覆盖all882。
  每batch固定family，globalrank按实际有效pool均匀采样，训练时不必全18重型query逐一枚举。
  EvidencePairs是GT-jittered离线state/早期GT templates，GT prior用于训练crop；不是无GT闭环，C必须用真实历史/预测再刷新。
  保存raw/clip私有label boxes源于已有M35/M50裁剪问题；这是具体可核验数据，不加fallback或容差放宽。
  原计划e559只在SOURCE前补raw labels改9b847，原件保留；没有NN运行或sanity权重冒充正式初始化。

- M54：公开上传成功不等于服务器同步成功。原98122/b752c7 publisher在解包前检查新37源而触发不存在新文件错误，真实exit1发生在Gitpush/19blob/SCP成功之后。
  修正read-only/pre-extract验证仅34已有源、解包后核对全部37，resume e61536 actualexit0完成同tar/commit858df31，未再次commit或重启NN。
  原remote script/stdout/stderr/transport保留private，三份HANDOFF5220b619一致；不补造首次成功回执。

- M55：candidate SOURCE_PASS还不是NN证据；现在真实准入已闭合。
  fresh SOURCE163232checks/70inputs stable/0block，两helper绑定实际SHA；GPU1 actual native M0 COMPLETE0，4update/32examples/64labels、257冻结8heads变化、12fields strictreload。
  K1旧四请求/gains逐位一致，10frame causal7actualaux+7replay所选执行逐位一致、265模型state不变。
  32paired image headersCPU-only采集；CPU2ef12e actualexit0，64raw/clip box、IoU及float32 signed gain全0误差，kind/role/seed/frame重建一致。
  这些仅证明可执行与标签正确；不认证学习策略有效、闭环提升或完整78/72/62及效率。
  正式B wrapper2748958在22:44:35单次启动，5×10000/6250/100000labels/all882，从fittedad059而非M0开始。
  原observer73305/local37192 first22:54:35后用实测吞吐安排nearend，必要240sec；loss/正式终态尚未读。

  M55首真实诊断：22:54:36 RUNNING500/6250、last100loss.09769086257/quality.05938930715/gain.03830155546有限，原/proc wrapper命令绑定live。
  .831906695update/sec给出ETA00:49:48、原唯一observer下一00:44:48nearend，节点前不额外读NN或handle；预计结束不是实际完训。
  loss降低不证明学得候选ROI有益；完整B终态后需要同固定reference状态下学得动作/STOP与相同单query预算简单对照，再刷新C。

- M56：收益选择器的离线训练状态与部署状态不同，需闭环刷新而不能用低B loss代替真实轨迹证据。
  当前B的EvidencePairs从当前GT扰动搜索、早期GT裁模板、history=0；candidate tracker从上一预测搜索、按预测写模板并更新真实history。
  代码直接确认这种分布差异，尚未证明它造成多少误差。已有train20查询收益实验说明候选存在机会，也存在负收益；没有证明学得策略已能利用。
  实现独立train_candidate_trajectory_policy.py：frame0GT初始化，先track推进再读当前GT，教师用保存的真实前状态。
  两role各采样一个家族内query，若实际所选查询不同则追加该查询监督；所选输出逐位复现，教师不得写回tracker。
  独立candidate_c序列化，257冻结/8heads训练、12fields strictreload和K1旧四请求一致性检查。
  实际CPU95b772 exit0：37旧源不变、Python3.7语法通过；manifest重算917106track/182182updates/1764init，M0预定20frame/4updates。
  draft plan1011b704为DRAFT且拒绝NN启动；独立draft源码审查进行中。正式B终态绑定、READY审查、真实M0/CPU标签核验和长C尚未执行。
  Where2comm公开固定代码仅用置信阈值通信mask，需求乘积仅注释；不能当成定位收益监督或省辅助编码的实证。
  KeepTrack候选身份/响应/写入依据需分开；本次没有实现其关联，也不把借鉴包装为已验证贡献。详见CANDIDATE_TRAJECTORY_PROTOCOL.md。

- M57：C标签CPU检查器的标量NaN会被max(error,NaN)静默忽略。
  独立审查对新checker727064d3做真实进程复现：main_iou、query_iou、delta_from_stop分别改成NaN，三次都exit0/REAL_CAUSAL_SANITY_CPU_PASS。
  这是合成schema/corruption测试，不是真实C训练结果；C还没启动。私有原始checker/fixture/stdout/stderr/transport保留，原审查FAIL不得覆盖。
  另一复现把总loss设123、quality/gain各.25仍被接受，说明仅检查loss有限不足以核对分项一致性。
  计划最小修正：先检查三个label标量有限，再计算误差；总loss须等于float32质量与收益损失之和。正在保留R1后执行复审。

  M57闭合：checker e1bf有限性与float32 loss和修正；独立R2真实15案例通过，包含6正常/9拒绝。R1真实失败保留；没有C NN结果。

- M58：C准入误把B四家族样本数写成每role都12500，实际6250个batch不能整除4。
  root实际f6bd46按B的全局step%4与batch8重算，得到12504/12504/12496/12496；错误门会拒绝正常B终态。
  只修C gate计数并重新绑定source8826/plan132967，B37源及训练不改。独立R2从B实际循环证明旧门拒绝正确计数却接受错误12500，新门相反。
  R1及旧C源码ea19/计划1011/raw prep保存private；R2 173核心检查+15checker案例、48inputs stable、DRAFT_PASS0block，仍非NN准入。

- M59：新交付目标与主路径能力需分开验证。
  用户最新目标LasHeR三项各+6，变为78.74/75.16/63.84，并要求LasHeR/RGBT210/RGBT234完整成绩。
  起点明确为去三模块CADTrack＋SOT backbone，不能与DropMAE或论文完整CADTrack混算涨幅。
  E0/E1先完成；身份检索、查询收益/风险/STOP排序、查询后验收/模板写入作为失败链改进。C刷新尚未实现后两项新机制。
  仅8策略头不能先验保证+6，完整版本之后做相同数据/更新量的读出和末层适应对照；不把解冻/定位loss当新颖模块。

- M60：显存富余不等于并发更快。
  用户授权GPU0/1并发评估。00:08真实快照：2295/2835MiB占用、GPU18–37%/13–44%，64核CPU约45%、load1=33.06。
  短9秒样本仅说明容量和瞬时利用率；尚未测总完成步数、CPU预处理/数据竞争或原任务减速，因此不能提前宣称并发增益。

- M61：RGBT210 名称子集不能证明图像帧对应。
  本地原工具包中210个名称均属于234，但147条同名序列的GT行数不同；总帧数分别104706与116650。
  服务器已定位RGBT234原图目录，尚未定位独立RGBT210目录或核验官方图像对应，不能直接把234同名完整轨迹按210标注评分。
  需先确认实际图像与各自官方GT逐帧对应；RGBT234继续使用原工具包MPR/MSR。当前无新的210/234完整指标。

- M62：首次并发比较没有实际任务重叠。
  旧candidate B于00:37:03终止，profiler wrapper于00:38:27启动；并发窗口旧B增加0步，不能当作双任务吞吐。
  实际独立profiler168测量更新/180.965秒、真实4步M0、严格重载通过，仅证明profile路径可执行，不证明并发加速。
  新受控pair使用独立primary持续跨三个阶段，期间primary必须存活；用closed timing ledger计步，硬件采样证明GPU1真实进程集合。
  独立CPU审查实际发现helper接受相同PID或第三GPU进程的两种伪受控输入；最小修复distinct PID、exact serial/overlap PID sets。
  修订helper333fc219通过206项SOURCE检查0block；原反例/旧helper保留。新的实际NN测量尚未收集，不预写速度结论。

  M62实际闭合：受控primary/secondary均native COMPLETE0，真实独立GPU1进程集合核验。
  合并单任务.925239update/s，双任务整阶段含startup1.569283（1.696倍）；纯重叠1.777164（1.921倍）。
  双任务GPU1最高采样5656/24576MiB、CPU50.99%；GPU0日志粗略速率下降约15.5%，但日志刷新滞后、计时边界和序列负载不同，不能归因或合计不同工作量。
  同类B任务有并发机会，尚不证明整个流水线或闭环C加速；当前正式依赖/旧E0后续GPU1预约保持，未制造额外正式任务。详见profile_output/RUNTIME_PROFILE_20261015.md。

- M63：RGBT234有一处实质标注版本差异，不能因帧数一致就静默混用。
  已定位RGB_T234完整234条/116650帧，RGB/TIR与工具包GT行数逐序列一致，首尾尺寸均一致；中间帧未全解码。
  afterrain visible、woman89 infrared两文件字节SHA不同；float32及取整后核验woman89数值一致，afterrain不一致。
  后续需记录初始框与评分GT来源，沿用既有工具包协议而不改原数据；RGBT210独立原图仍待路径确认，不能以234同名完整序列代替。

  M61/M63数据准备更新：已定位独立RGBT210 `/data/wangwj/dataset/RGBT210testingset`，实际完整210/104706准备成功，无需替代。210/234初始化与GT明确用本地CADTrack工具包，保留未取整float32 GT；全部清单私有封印。正式完整指标仍待运行。

- M64：新三基准入口R1来源准入实际失败，正确算术不能代替正确输入。
  独立CPU真实进程证明原scorer a01b/plan f431误接收未登记的人工dev97/291帧、借用其他任务命令、分片参数量123/987以及未知variant。20个其他异常样例正确拒绝、七指标全阈值golden一致；人工输出不属于基准成绩。
  主入口stub反例另证明错误C task/GPU/run/配置、重复882名字和FAILED COMPLETE能通过；无真实dev收据、改写预算/三数据集配置也会通过。
  原43源/计划和两个FAIL报告保留，未NN启动。R2只增加完整manifest-plan、实际命令/模型/分片参数量、C身份/原E0完整审计、完整dev原始artifact与固定选型SHA绑定；44源封印旧38不变。复审、真实推理准入、正式指标尚待完成。

  R2实际复审：原四问题闭合，完整dev三cost/三基准test人工CPU协议通过；冻结却接受三个仅含无关marker的声明报告，交换shard标签后保持旧序列归属也会通过，故R2 FAIL2保留。
  R3实际封印fa9bd08a/44source，要求完整真实run集合/config/COMPLETE/native/帧数/timing、exact shard和真实CPU scorer重算全JSON一致，验证后才创建冻结。五真实清单在项目data/benchmarks精确复制，仍Gitignored。独立复审及NN推理准入待完成；旧38/训练配置未动。

  M64 R3 SOURCE闭合：主审41fd2ce5、metric08cb548b均PASS0且原两轮报告字节未动；真正E0/E1各11帧train inference native COMPLETE0，零更新/严格重载。实测审计已关闭WARN/0block，182项检查、118输入封印；在线均RGB主模态、E1未执行全图候选，张量结论依赖native assertions。仅执行准入，尚无新完整C/三数据集成绩。

- M65：新增E1完成控制器首轮SOURCE审查实际复现依赖失败漏读。旧`complete_all_train.wait_task`会先保存`last_observation.state=FAILED/exit7`，顶层仍为`TRAIN_ALL_882`再退出；新控制器只检查顶层，CPU反例实际连续安排240秒等待。未产生SSH或NN调用。
  最小修复只在新控制器依赖循环读取该已有字段，拒绝FAILED或非零COMPLETE；原44份源码、旧owner及评估计划不改。原R1报告/source45/计划e918原字节封存；修复后的controller a347e987、计划deb0d51e通过独立R2 SOURCE_PASS/0block，17个CPU用例实际通过，失败立即停止且正常等待无SSH。07:07:52本地owner33868单次启动、初态WAIT/tasks为空，正式C未启动。此为编排正确性修复，不是网络模块或性能贡献。

- M66：三基准入口可执行不等于E0已经排入三个完整测试。核对旧`complete_all_train.py`，其自动链只跑LasHeR；210/234无自动任务。
  新增独立E0完成控制器复用R3入口、完整dev冻结与全245/210/234评分，等待E1全流程和全部原生任务exit0后才使用两卡。没有改正在等待或训练的45份源码，没有新训练或去模块CADTrack baseline重测。计划1df1b9d2/新源ce66a6a3通过SOURCE_PASS/0block，27个CPU用例和191段虚拟远端代码；07:24:40本地owner31556单次启动、WAIT/tasks为空。首次08:34:30只读E1本地状态；当前未新增NN。此项解决交付缺口，不作方法创新或性能结论。

- M67：错误模板写入发生在漂移的哪个阶段，不能只由同帧IoU判断。已完成C20 dev97/54646帧的纯CPU时间统计：1053次低IoU写入中1044次前一帧已经低IoU，851次此前连续低IoU至少20帧。88个低IoU片段包含这些重复事件，事件之间并不独立。
  独立审计WARN/0block、算术PASS，97份GT与原manifest序列化逐字节一致；全部12未来窗口、置信分箱、片段及97序列项复算一致，1053事件CSV与实际CPU复跑输出逐字节相同。源46不改，无NN或模型更新。优先保持候选/恢复问题；模板写入是否使失败延长须匹配预算的关闭/回滚控制。详细计数、排除分母和非因果边界见`C20_STATE_WRITE_TEMPORAL.md`及`C20_STATE_WRITE_TEMPORAL_RESULT_AUDIT`。

- M68：不能把所有crop内部低IoU直接归为干扰物，也不能把最终响应当成单模态质量。固定条件抽取的四个C20例子已实际查看；`left2ndboy`425帧TIR主模态、查询后的最终响应约0.952，所选框落在红衣人而GT在黑衣人。两个`leftdrillmasterstanding`例子可见不同人的GT/预测分离，但不代表全部593个内部错误。
  这两帧目标对当前E1全图32网格投影宽度不足一格，只提示分辨率/候选分离参数值得控制验证；旧C20未运行E1全图网格，CNN也不能据此被判定存在硬分辨率界。32原JPG（30唯一像素文件）、4图、选择规则及GT序列化经原审查者continued supplement复核WARN/0block，405封印不变。原9af数值审计不改，原图/坐标私有，未新开NN。完整限定见`C20_STATE_WRITE_VISUAL_NOTES.md`与`C20_STATE_WRITE_VISUAL_REVIEW`。

  推理实测审计71d82995现已WARN0block/182checks/118inputs stable。E0动作1/4/5/0/0、E1动作9/1/0/0/0，各10帧RGB主；E1全图请求没有在线执行。冻结与重载依赖native assertions，clone计入smoke显存峰值，不能泛化为全路由/完整精度/速度认证。

- **M69：旧配对速度入口不支持正式 E1（2026-10-15）**：源码只构建DualRGBT/Tracker，不能以旧E0计时证明候选策略E1效率。新增独立E1配对入口与等待控制器，原46封存源不改。SOURCE R1实际FAIL2：远端评估计划SHA未绑定；互相一致的错误命令及身份字段会在模拟收取中放行full。原始报告/输入保留，最小修正后R2 SOURCE_PASS0block（75CPU/67AST，JSONc3adb855…）；等待owner21656已单次启动。原部署调用14107收据未返回，另以只读进程API核对实际命令/出生时间/存活并保留独立恢复收据，未重启控制器，也未伪写原调用exit0。最终须等待E0/E1三基准结束后，原生1×16准入→完整97/54646每侧；两侧独立状态/解码、交替顺序，统计service/tracking平均P95FPS，双模型共享显存明确限定。没有新实测速度。详见E1_PAIRED_LATENCY_PROTOCOL.md。

- M70：候选几何覆盖、候选排序和已覆盖区域的执行质量不能混成同一瓶颈。
  已封存旧full-B train20/1752固定状态、31384实际query的CPU重建通过独立数值PASS/语义WARN0block，730597checks、91inputs与正式64源不变，producer及独立CSV逐字节重现；不是当前E1/test。
  两侧均8候选的1334状态中，分离半径S/4→S/2使wide覆盖1108→1157、local801→738；这仅为几何比较，改后ROI未NN执行，不称精度或效率提升。
  239主crop外状态中184个wide池至少覆盖半个GT，但其中84个最佳已执行query仍IoU<.5；原32网格全可达只排除本子集的中心几何盲区，不排除身份/可见性/容量问题。
  因此不统一调大参数、不热改正式训练，保留按请求尺度分离和预览容量的有限对照、执行器适应与查询验收；均需刷新监督、独立验证和完整闭环/耗时。所有GT-oracle与名义尺寸分层限制详见CANDIDATE_RETRIEVAL_LIMITS.md；审计JSON85a6438e…，same-family/provisional/routeUNATTESTED。

- M71：统一全图候选池不能分别控制local/wide的空间分散程度。
  根据M70实测覆盖差异，新增独立ScaleCandidateEvidencePolicy，仅调整模块二内部候选分离比例；原三个模块、955收益头/成本/执行方法保留，无新增学习层。
  每类至多8请求，合计至多18+STOP；候选位置并集可到16，request_cell_indices/family_candidate_counts明确语义，旧tracker的candidate_count在新路径为去重总位置数。新配置不得使用旧正式E1收据准入。
  actualCPU71925/functions1207/a48c87 true0：1752原响应图/真实几何加人工主预测张量，原输出/K1逐位一致、两类候选NumPy一致；1715状态列表不同、最大15不同位置。无NN/权重/CUDA，87.99s仅检查时间；五条Torch1.12格点整数除法warning已读原stderr，未无关重构。
  独立SOURCE数值PASS/语义WARN、0block，审计9b83210247dbb7d2e8b7b5b65118fc0844f6bf9f8783418a29a544fc0cd3de80、same-family/provisional/routeUNATTESTED；完整GPU M0、梯度/重载、新监督、完整训练和三基准/效率待验证。原64源与正式队列不改，详见SCALE_CANDIDATE_PROTOCOL.md。

- M72：双间距是普通属性，不在state_dict中；新的训练与重载必须显式保存相应配置。
  新增独立scale_candidate_b训练入口及checkpoint loader，保存/恢复candidate_config，不添加默认参数或兼容旧格式。目标仍是原三模块，955头和执行器不变，8训练/257冻结。
  计划5x10000样本、6250updates/100000新执行query标签，来源全882；M0真实4更新/64标签和14项重载比较尚未执行。新记录明确family_candidate_count、unique_candidate_count、request_cell与rank，监督从新ROI执行框重新计算。
  新SOURCE复核PASS0block/JSON 78c3cd5b478d90f6590b794ae8fa53ea436230e5667e3d8ba0cb60898f685abc，原64源不改；CPU准备73146/3414b2 true0重算采样预算，不是NN或实际训练覆盖。首次private AST误读YAML失败已保留并只修helper过滤.py，没有重启任何训练。
  旧12字段/单candidate_count的CPUchecker不能直接复核新14字段；未来M0须按新schema独立检查，不构造旧格式兼容收据。完整C、新三基准/速度和GPU M0待后续，未新增GPU任务或改变正式队列优先级；不能用此准备声称精度、速度或三项+6。详见SCALE_CANDIDATE_TRAINING_PROTOCOL.md。

- M73：新14项重载/家族计数日志须独立检查，且M57的NaN累积风险需要延续修复。
  新check_scale_candidate_sanity.py使用实际M0记录/GT/header，核对margin10执行框、IoU、signed gain、float32 loss和、rank/count及格点ROI；先检查label/loss有限再累积max。无旧格式fallback，原68NN源不改。
  人工synthetic/mock协议样例实际运行：1正常带正负收益通过、15异常拒绝；不是真实NN/M0或数据集成绩，real_new_m0_completed=false。独立SOURCE PASS0block/审计2b223b628c7c9080a16acb5e4c7db859bc458b3971e2dedd38e40133bcc14d0b、same-family/provisional/routeUNATTESTED。
  checker不能认证声明来源，也不替代真实梯度/Adam/冻结/重载/因果重放。未来真实M0收取后须按本schema进行CPU与独立结果审计，再正式B；当前队列和12:30/12:37节点保持，见SCALE_CANDIDATE_M0_VALIDATION.md。

- M74：普通相对初始模板质量门已有先例，不能只加一个质量头就当新贡献，也不能忽略比较所需完整编码。
  固定作者UATrack a600bb29源码的真实测试入口指向BATTrack；每10帧以同一缓存搜索比较初始/候选，条件为relative quality较高OR窗后分类峰值不低。router在完整双模态编码后，比较节点batch2→4；写入在当前框确定后，但全batch极值归一化可能改变当前输出是待数值验证的静态推断。
  该公开版本存在导入缺符号的静态疑点，未执行/修复作者模型或核验权重，不推断论文无效；论文仅读出版预览。保留其普通模板写入对照，区分调用前价值、调用后定位验收与记忆写入，匹配训练/可用信息及实际延迟。现有C20写入统计仍不支持因果归因。
  只新增 docs/UATRACK_STATE_UPDATE_REFERENCE.md 及本记录，原68源与正式E0/E1队列不改，无新GPU任务/训练/三基准成绩；既定12:30/12:37观察不提前。

- M75：token预算不能直接当真实延迟；两个完整dev已有实际调用差异，仍缺配对速度证据。
  同一全882 E0权重、同97/54646协议，cost0→0.05辅助调用率88.278429%→66.793158%，54549非初始track分母；重型token均值628.968744→529.672331（约-15.79%），EXPAND20908→4478。同期service FPS均约9.88，但两任务分处不同GPU并行，不能因相近就断言不加速，也不能宣称效率目标成立。
  fresh数值审计确定性PASS/语义WARN0block，JSON 3a39aab36f3a2a2260c76faaaf60f8eeb4f3dded2ae7f980797f3c73cf870f03，源与结果逐字节身份、真实GT/全部曲线/trace复核通过；范围仅原E0前两档dev。继续原定配对系统计时，必要时按实测剖析各执行段；当前不更改控制器/编码器或伪造基准速度。

- M76：固定首次观察裕量造成可测的阶段间排程空档。
  两个dev native于14:04:35完成，第三cost0.15于14:47:56启动，间隔约43.35分钟；complete_all_train.py固定135分钟首查，与已测约92.5分钟实际任务时间不同。它表示所列任务的完成→下一启动空档，未采样其间其他服务器活动，不能称整个GPU/服务器全程空闲。
  当前封存运行继续；后续新登记计划按本次实测阶段时间安排首查、预计结束前约5分钟后以240秒观察，减少固定裕量。此项用于执行排程，不作为模型贡献或已实现吞吐提升；没有修改活跃NN源、重启任务或增加兼容/恢复分支。
