# E0/E1 完整三基准评价

最低LasHeR目标为PR78.74、NPR75.16、SR63.84。用户baseline72.74/69.16/57.84沿用，不重测精度；初始化仍是去三论文模块的CADTrack、SOT backbone。

E0是既有四请求模型的全882条两轮闭环C版本；E1是全图候选模型的全882条两轮闭环C版本。两者均须完成917106次track、182182次策略更新；完整B或20帧C准入不等于完整C。E0原owner及其自动LasHeR评估保留，不热改旧38源码。新入口提供独立E0/E1完整评估能力，尚未产生正式结果，也未启动长E1 C。

先完成完整LasHeR dev97/54646，在0、0.05、0.15三个成本配置中按SR、PR、NPR、较低平均延迟顺序选择。固定final C round2；E0候选参数为0，E1为8。当前协议三数据集使用同一checkpoint和配置。`tools/freeze_benchmark_selection.py`绑定三份完整dev评分及其原始文件SHA，输出只创建一次的冻结文件；test命令必须明确带该文件SHA。不能根据test结果重新写选型。

完整测试为LasHeR245/220703、RGBT210210/104706、RGBT234234/116650。首帧初始化后，仅模型预测、已获取图像及历史控制搜索与模板；当前GT不参与在线决策。评价入口在整条预测完成后保存评分GT。RGBT210/RGBT234初始框及GT均用已核验的CADTrack工具包标注；各自PR/SR、MPR/MSR以及LasHeR PR/NPR/SR调用原工具包，不增加预测统计归一化。

评分核对登记manifest SHA及完整帧数、真实run_task命令、variant/checkpoint/plan/run/shard/冻结SHA、全部分片参数量一致性。GPU0/1才获授权。NN入口须两份独立SOURCE报告通过且绑定相同计划/44份源码；旧38源码不改。

SOURCE R1实际CPU反例发现完整C身份、dev冻结、未登记短manifest、借用命令、跨分片参数量和未知variant准入漏洞。失败报告原字节保存为`*_SOURCE_REVIEW_R1`；这是人工CPU协议测试，不能写成真实基准性能。R2闭合原漏洞后复审又发现marker-only声明型dev可冻结、shard与序列归属可错位，失败报告另存`*_SOURCE_REVIEW_R2`。R3要求完整实际run artifact集合、exact shard与原生身份，并在写冻结前实际重新调用CPU scorer，完整JSON须与原评分一致。R3两份独立SOURCE现已PASS0/same-family provisional；两项真实GPU1推理准入均native COMPLETE0，实测审计尚待关闭。

真实推理准入计划仅GPU1依次执行E0/E1，各用11帧LasHeR train图像，冻结全部权重、无反向和优化器；保存并严格重载因果tracker状态，框/route/templates/history逐位比较。首次180秒后读取，必要间隔240秒。M0模型权重不用于正式训练选型，11帧结果不计算最终指标。

上述两项现已真实完成并独立审计WARN0block。E0的STOP/两个local动作、E1的STOP/主候选local被执行；两个主模态均只有RGB，E1全图请求未在线执行。冻结/重载张量等值依赖native assertions，未下载独立模型张量；smoke显存峰值包含重载clone。这些限制不改成完整模型精度/速度结论。

完整结果记录源码/配置/权重SHA、逐序列预测、GT来源、完整覆盖、原生exit码、参数量、平均/P95延迟、FPS与查询比例。服务计时包括图像解码及初始化，排除模型加载、结果写盘和SHA核验；双GPU按总服务时间报告吞吐，不能当作单GPU墙钟速度。

用户允许后续按数据集调整模块内部参数或层数。210/234独立验证协议尚未登记；当前统一配置不以此授权静默更换。改变层数/宽度须对应训练权重及成本，专用配置另列、另冻结。

## 完整版本的执行归属

旧E0 owner仍负责既有全882×2 C、旧完整dev及LasHeR test。E1完成控制器已经登记并启动本地等待，由其接续E1全C、完整dev冻结及三个完整test。原有训练和评测源码均保持封存。

核对发现旧E0 owner没有RGBT210/RGBT234任务。新增独立`complete_reference_benchmarks.py`补齐E0的登记R3完整dev及三个test，复用已有等待、原生收据收集和评分函数。该控制器须独立SOURCE通过后启动，仅在E1三个完整结果和全部原生任务exit0之后获取GPU0/1；不与E1争用卡。使用E0原全C权重，不另训练、不改模型，也不重测用户的去模块CADTrack baseline精度。

E0新入口再跑完整dev，是为了取得R3冻结器要求的真实run配置、原生命令和文件链；不把旧格式收据伪装成新格式。三项test都使用这次dev冻结的同一权重与配置。旧E0原结果单独保留，R3结果单独命名；没有结果前不提前填表。

2026-10-15 07:24启动记录（SOURCE结束后追加）：E0完成队列SOURCE_PASS/0block，绑定46源及计划1df1b9d2；27个CPU用例通过，仍只支持编排源码。已单次启动本地owner31556，初态WAIT_E1_PIPELINE、NN任务0，首次08:34:30只读E1本地状态。已审查协议原字节3324653c…另行私有保留，46源不因本段启动记录而改变。实际记录见`reports/reference_completion_launch.json`。
