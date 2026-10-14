# Dual-RGBT 交接文档

## 用户目标与约束

2026-10-13：25服务器仅GPU0、1；项目和conda环境在`/home/gaob/Dual-RGBT`。
优先LasHeR，目标PR≥78、NPR≥72、SR≥62，并明显降低完整在线推理成本。
用户确认去模块CADTrack baseline PR/NPR/SR=72.74/69.16/57.84，按要求不重测。
唯一初始权重`/home/gaob/CADTrack_ep0014.pth.tar`（epoch14，原文件不动）。

## 最新状态：全图正式A完成，收益诊断只读准入已启动（2026-10-14 10:23 BJT）

原全图正式A wrapper1493008/child1493009于10:09:44 BJT真实COMPLETE/exit0，未重启。
完整5×10000对、6250更新、十路径各5000、882个train序列均访问；全部6250 CSV loss有限。
唯一near-end observer native3693/外层cell502已真实exit0终态，不再poll；CPU collector native5699/cell517也exit0。
CPU按真正初始正式证据A d198逐state比较，168冻结state不变、58允许state改变，Adam步数6250，所有state有限。
最终checkpoint SHA4836ade9b7afc417f90a995d90ef70a4681347fd000e3a4f0f1de313af9fe06e。
30源码/训练计划/原B factory/初始d198/用户CAD权重保持；三份config/COMPLETE/steps原字节local-server一致。
这是执行器正式A完成，不是全训练管线完成，不是PR/NPR/SR或效率结果。

新增只读train20同状态机会诊断，先检验局部执行器与可部署候选的收益，再推进新正式B/C。
20序列8854 track、1752状态，每状态两主模态各5动作，以及所选主模态4控制，共24528比较是完整诊断计划。
强制STOP参考轨迹由预测闭环推进；GT中心控制只用于诊断，不写回搜索/模板/路由，不能当轨迹上界或最终指标。
原始控制CSV含GT派生数据，PRIVATE_ONLY，不上传Git；公开仅聚合结果和审查证据。

源码R1 SOURCE_FAIL/1阻断保留：strict-reload clone未显式关requires_grad，虽然已有no_grad且无更新证据。
最小两行冻结修正后R2 SOURCE_PASS/0block、29检查exit0，原30源码/helper未变；同family/provisional、routing UNATTESTED。
新driver SHA c443f34c91149dee16cefa89db42e2eea1921cc21c0f656a40d22a2c70ff6272，诊断plan SHA a0accbdea3c6c8742103484e7a6c1db8bc5507647f1975f5f048477c79ef84e7。
全部原审查产物与首末seal保存private trace；公开R1/R2原MD/JSON不改字节。
先执行已审查absent-only部署器，仅传新driver/plan，远端source31和CUDA隐藏--help actual exit0，没有NN执行。
10:23:06 BJT单次启动GPU1只读sanity task diagnose_global_opportunity_sanity_a_20261014，wrapper1624986，使用正式4836权重。
10:23:47实际COMPLETE/exit0；10 track/2状态/28比较、265state不变、0 backward/0 optimizer、strict七项probe通过。
唯一observer native77808/local41276/cell526在10:26必要节点实际收回exit0，三份原报告local-server哈希一致，均终态不再poll。
原B factory、正式4836、source31及用户CAD权重末次哈希保持。两状态IoU只能证明诊断执行，不能当方法结果。

10:27:16.781 BJT单次启动GPU1完整只读diagnose_global_opportunity_formal_a_20261014，wrapper1633016，输入4836不变。
预计20–35分钟，仅安排估计，不宣称已完成。唯一observer native93026/local29412（10:28:58创建、一次CIM同argv存活确认）
首必要校正节点10:37:16.781，外层唯一join cell532在10:37:36.781读取。此节点前不查询NN进度/日志。
owner随后按实际1752状态进度估计结束前5分钟观察，确有必要才240秒复查；不创建第二observer或重启诊断。
新正式B/C尚未启动。原GPU0全882 C下一12:29:56观察，不提前查询。

目标仍ACTIVE/UNMET：完整test245 PR≥78/NPR≥72/SR≥62及真实系统效率、三个有效核心机制均未证明。
dev97只校准模型/阈值/预算，不代替test；train882是训练，用户baseline72.74/69.16/57.84继续直接沿用。

## 历史节点：全图正式A首节点正常更新，975/6250（2026-10-14 09:22 BJT）

global sanity A/B/C原生exit0，串行owner23472/native71001及唯一join functions492也已actual exit0；
全部终态，不再poll或重启。A80对/十路径各8/10更新，168冻结state不变、58状态改变；
B16对/2更新，251冻结/14变化；C own-state10帧/1初始化/2更新，257冻结/8变化。
三项7-field完整probe保存后strict重载逐位相同，C所选反事实逐位重现实际执行框。
30源/计划未改，9个config/SANITY/COMPLETE原字节local-server匹配；只证明真实可执行及更新，不证明精度/速度。

新正式A run global_evidence_formal_a_20261014于09:11:46 BJT单次启动，wrapper1493008/child1493009，GPU1。
原--executor B用于严格构建；实际初始状态随后严格加载已完成正式证据A d1981fef...，不是sanity权重。
09:13:33实际startup核对argv/CUDA1/PCI顺序/物理卡UUID0a03017b.../30源码/远端plan/config init d198，均一致。
阶段A继续训练新全图候选局部/宽查询路径：5×10000对、6250更新，全图版本的正式B/C尚未启动。
首计划节点实际09:21:58 BJT取得RUNNING：975/6250更新，epoch1，全部975条已写loss有限，最后.112426989。
实际平均1.593603更新/秒，线性结束估计10:17:08.883423，不能当完训承诺或精度改善。
首observer native21057 actual exit0（c879b5），已经终态，不再poll。
唯一near-end observer native3693/local45060于首结果后单次启动，等10:12:08.883423 BJT=1791943928.8834229；
node前不查询loss/进度/正在写权重，不新建第二observer。原正式NN1493008/1493009不重启。
原native3693的输出现由唯一异步join functions.cell502在10:12:28.883读取；恢复cell502，不并行poll3693。

SOURCE_PASS是52项源/标准库检查的同family/provisional结果，routing UNATTESTED；原37输入和79完整原产物已归档。
GLOBAL_EVIDENCE_PROTOCOL及前次启动快照保留历史PENDING/ADMIT_A，不当作当前状态。
旧2S版本正式A完整50000/6250/882seen已完成，但其B/C未启动；其原25源/初始计划/输入权重均保持。
R2/R3诊断只支持进一步验证全图廉价请求，不宣称成功恢复、尺寸估计、精度或提速。
充分训练后须检查主模态／局部／宽查询实际收益、候选覆盖与同预算简单对照；
若oracle查询无益应先修候选/执行器，不直接用更多门控训练解释为改善。

GPU0原全882 C仍由owner16936继续，最近08:29:56节点21225/182182，下一12:29:56自动观察；
17源及原test freeze保持。新完整C启动前依据实测吞吐协调参考任务后续双卡dev/test，不抢占或重启。
目标完整LasHeR test245 PR≥78/NPR≥72/SR≥62、真实整机效率和三个有效核心机制仍ACTIVE/UNMET。
已有73.916231/70.398889/58.587111是旧C20参考完整test，不是当前全882/全图最终结果；未重测baseline精度。

## 历史节点：全图SOURCE通过与准入启动（2026-10-14 09:01 BJT）

fresh SOURCE_PASS/0block，43静态+9标准库几何及实际helper检查通过；原36+增量1输入不变。
requested Astra/max、actual routing UNATTESTED，同family/provisional；该审查不替代NN准入或精度验证。
77原审查产物、seal及post-seal实际工具receipt共79份保存private forensic trace，不上传GT或private命令日志。
审查器自身首次几何checker SyntaxError原v1/exit1保留，v2实际exit0；不是项目NN故障。
公开SOURCE_REVIEW原MD/JSON字节及机器helper哈希保持；新30源、计划b5cf7d...和4个helper精确核对。
实际已发布新5源，所以原absent-only部署器没有调用，明确排除；采用只读发布验证器。
09:00:23服务器校验30源/plan/原B/正式A d198/manifest一致，CUDA隐藏--help实际exit0，无模型构建或新NN。

GPU1串行A/B/C准入单次启动，local owner23472，09:00:23创建，09:01:56实际CIM同argv确认存活；
native71001唯一原调用，当前task admit_global_evidence_sanity_a_20261014 wrapper1472419，09:00:36启动。
owner自动等待A首节点09:03:36，之后B、C逐项通过才继续，每项首180秒/需复查240秒；不提前远端读NN进度。
本次只是sanity，不是新正式A/B/C，不能用这批诊断权重开始正式A。
新正式A继续由已完成原正式证据A d198初始化；新完整正式训练尚未启动。
输入原A已真实6250/50000、十路径5000、882seen/nativeexit0，详见下方终态与原始记录。

R3尺寸/名义像素密度独立CPU复算PASS/0block，1986子集逐帧匹配R2/291输入哈希一致。
理想中心raw宽高容量最小.549246，不是实际IoU下界；完整GT最长边名义投影不是crop内可见/检测大小。
current128投影<16/<32是17/67，initialtemplate假设为1/6，不据此保证分辨率足够或估尺度有效。
原15产物+3seal receipts保存private，公开GLOBAL_SCALE_DENSITY_AUDIT原字节；保持简单方案，没有新增尺度头。

GPU0原全882任务继续，最近计划节点21225/182182，下一自动读取12:29:56 BJT，17源/冻结不改。
新完整C正式启动前须根据真实剩余时间协调原任务随后的双卡dev/test，不能盲目自动启动占用GPU1。
原C20完整test73.916231/70.398889/58.587111；最终全882训练、78/72/62、真实提速和三个有效模块仍ACTIVE/UNMET。
GLOBAL_EVIDENCE_PROTOCOL是审查前冻结协议，其PENDING历史保留；当前状态由SOURCE_REVIEW和本节区分。

## 上轮节点：正式执行器A完成与外部范围问题（2026-10-14 08:40 BJT）

新证据执行器正式A wrapper1309002原生COMPLETE/exit0，08:22:31 BJT完成。
5×10000对、6250真实Adam更新，全部882 train序列出现，RGB/TIR×5路径各5000对。
全部6250条loss有限；CPU读取实际checkpoint step/optimizer step6250、全部saved state有限，
164个可与原B比较的冻结执行器state逐位相同。新gain初始化未归档，不能虚称其正式前后逐位比较。
权重SHA d1981fef9efe797e14627515fe3b8e49c235efbe1d548a25a1af7a8c8412ca07；
config/COMPLETE/steps原字节local-server核验，见evidence_policy_formal_a_completion。
收集脚本首次把decode/box_head误写成decoder/head，冻结范围断言实际失败；
保留原脚本及实际只读重放stderr，只修收集器命名后actual exit0，没有改/重启训练。
原near-end observer/native71592已exit0，勿再poll；其6075节点是历史，不是当前终态。

GPU0原全882 C在08:29:56计划节点依然RUNNING，21225/预期182182更新；
最新100次更新loss均值1.40324825、quality1.36121816、gain.04203009。
原owner16936身份存活，下一自动节点12:29:56，不追加早期远端查询、不重启。
完整882训练及最终dev/test未结束，旧C20完整test73.916231/70.398889/58.587111仍只是参考。

R2独立CPU诊断SCOPE_COVERAGE_PASS/0block（同family/provisional，routing UNATTESTED）。
旧C20完整97dev、54289正宽高非首帧，2S中心越界3001/GT零交集2545，
全部64外部local候选并集2.25S对应2508/1986，复算和291预测/GT/trace哈希一致。
只是六位保存框近似的名义源覆盖，不是全动作性能上界、不是新模型恢复结果或新PR/NPR/SR。
20项原R2产物及seal保存private .aris trace，公开无GT报告GLOBAL_SCOPE_COVERAGE_AUDIT原字节保留。

根据实际覆盖问题另建dual_rgbt_global、train_global_evidence_policy.py和新计划，旧25源不改。
主64预览选主；共享小CNN直接处理完整图256预览→32grid，global候选排除矩形填充区域。
先选择收益和请求，再读取辅助原像素：全图局部source S/2→128/193tokens，
同候选宽查询source S→256/385tokens，保留主/分离次候选和STOP。
新正式A明确严格继承上述已完成正式A，随后B学习真实执行IoU差，C全882因果刷新。
新源码独立审查和真实A/B/C准入目前待完成，不能报告新训练已启动或精度/提速有效。
原2S版本正式B/C暂不启动，避免以不能覆盖全图的候选作为最终恢复方案；其计划/历史证据完整保留。
开源原理及作用边界见GLOBAL_EVIDENCE_PROTOCOL，AdaFocusV2区域选择/KeepTrack候选身份均有作者GitHub。
新C启动须依真实吞吐/预计结束时间协调原参考随后双卡评价，不能盲目抢占GPU1。
最终78/72/62、真实效率优势和三个有效模块仍ACTIVE/UNMET。

## 历史节点：完整A/B/C准入与首次正式A进度（2026-10-14 07:33 BJT）

已将新reader接入完整EvidenceTracker及独立A/B/C训练，不修改原GPU0参考任务。
两个64px main预览提前选主模态；较宽低清预览覆盖2倍main源边长，支持旧main外候选。
四个具体query在aux重型编码之前生成，以实际margin10/float32执行框的原图IoU改善预测价值；
STOP直接输出main，选中的局部128/193tokens或WIDE256/385tokens独立原图读取后共享编码修正。
原旧loss-gain MLP删除，另加955→64→1 IoU-gain；参数/整机速度优势须实测。
A/B随机训练状态与A教师query明确不当成真实在线评测；C只有初始化GT维持状态，之后预测控制搜索/template。
代码和初始协议见dual_rgbt_evidence/{policy,regions,training,tracker}.py及train_evidence_policy.py，
docs/EVIDENCE_POLICY_TRAINING_PROTOCOL.md是审查前冻结协议；plan原PENDING标记保留，当前准入由单独SOURCE_REVIEW授权。
fresh SOURCE_PASS/0block：25源、2个本地helper、955维、ROI几何、GT边界、完整882范围及阶段链接通过；
requested astra/max、actual routing UNATTESTED，同family/provisional，不能当NN或性能认证。
49个原审查产物/命令/真实检查器失败历史保存private forensic trace，25源远端一致且隐藏CUDA CLI实际exit0。
新准入owner38244（07:05:11创建）的调用native21887已终态exit0；勿再poll或重启。
GPU1 A wrapper1287582实际07:05:23启动、07:06:27完成exit0，80真实样本/10组合各8/10更新，
168冻结state逐位不变、58状态改变、7项完整forward probe保存后strict重载逐位相等。
B wrapper1292726、C wrapper1297940均实际exit0，分别2次更新，251/257冻结state保持一致，14/8状态变化。
C完整真实10帧/一次初始化/两次监督，选中反事实执行框逐位复制真实state；两阶段7项strict重载probe一致。
三阶段config的实际planSHA与封存e4aae856...相同，全部准入权重只诊断。
GPU1正式A wrapper1309002/child1309003于07:18:56单次启动，初始权重为原正式B ba49b506...，
没有用准入权重。固定5×10000对/batch8/workers4/6250更新；当前未完训，正式B/C尚未启动。
07:25实际startup身份PASS：argv、CUDA1/PCI顺序、卡UUID0a03017b...、25源和远端planSHA一致，config sanity=false。
首节点不早于07:28:56；只读脚本换行转义失败后，在07:32:46实际取得进度：
RUNNING、CSV1300/6250真实更新、epoch2、所有写入loss有限，最后loss0.096948624。
实际前缀吞吐1.567646更新/秒，线性结束估计08:25:23 BJT，非完训承诺；下一必要检查08:20:23。
不读取正在写的权重，不由低loss推断精度达标。错误编译脚本没有执行NN查询，也没有重启训练。
之后正式B同量新收益学习、正式C全882两轮因果刷新；详见EVIDENCE_POLICY_STAGE_ADMISSION.md。
此前只读startup检查脚本因字符串嵌入NUL实际exit1、原NN未重启；用chr(0)修正后同任务身份核验exit0。
首次进度脚本字符串换行也编译失败；只改rfind(chr(10))并先本地编译嵌套段后实际观察exit0。
局部修正器初次字面计数过宽assert失败及缺失新脚本的调用也如实保留，均非NN/服务器任务失败。
发布模板曾默认GBK解码中文桌面路径；已精确改回用户指定交接文件，误名副本可恢复移动到private，失败证据保留。
指标78/72/62、实际效率优势和三个有效贡献仍ACTIVE/UNMET；旧GPU0 C首诊断仍08:29:55，不提前查询。

正式诊断完成后新增：GPU1冻结B诊断已经COMPLETE_DIAGNOSTIC，20序列/8854track/
1752监督状态/14016动作收益对，265张量前后不变、0optimizer；1752所选执行框重现逐位一致。
原生正式task exit0，原本地owner47240也由事先取得的实际句柄读到ExitCode=0；
native56076与functions cell379均已终态，勿再poll或重启它们/原队列。
实际收集reward_diagnostic_native_completion并核对三文件local/remote原字节、B/19source/manifest。
CPU分析93a7b3真实exit0：197对符号翻转，其中GT截断191、执行映射6；
同成本诊断oracle只56/1752状态改变，旧oracle执行regret.000164392 vs实际策略.068398766。
不能只修clamp就说失败已解决；这也不是重训或完整闭环性能改善上限。
258中心在crop外的状态，两个主框及全部2064辅助动作原图IoU均精确0，
实际255/258仍查询，其中178完整辅助；执行loss收益.023964753并不等于找回目标。
下一项实现需真正读取旧源视野之外的目标相关区域，并使用可输出外部框的定位路径；
不能仅增加旧crop内tokens或改损失名称。见TRAIN_ACTION_REWARD_RESULTS及M35/M36。
GPU0原全882任务继续，不改17source/计划；正式完训、最终test和配对效率仍待完成。
冻结诊断fresh核验确定性PASS/语义PASS_PROVISIONAL/整体WARN、0计算blocker：
968数值字段重算，962 exact、6项仅fsum/mean舍入≤1.39e-17，14016动作原图IoU exact。
本地原manifest字节SHA验证并核实20train与dev/test不交叉；102MB含标注快照仅留private，不上传Git。
实际B权重payload未复制给reviewer，本地fresh权重字节核验仍UNAVAILABLE；父线程真实远端SHA保留。
原协议的NOT_RUN视为历史；审计保留所有真实失败回执及文档增量修复，不重新运行CSV/NN。
见TRAIN_ACTION_REWARD_AUDIT.md/json与POST_UPDATE；actual reviewer routing仍UNATTESTED。

新增独立坐标执行器dual_rgbt_evidence/readout.py已实现，R1实际发现越界边缘插值.25，
仅乘源区间半开覆盖mask后R2 SOURCE_PASS/0block。当前源码SHA2248b1af...；
局部128实际64搜索tokens/总193、8×8共享head，宽256总385/16×16，参数复用而非复制。
共享主特征按原图几何投影，查询框按自身几何输出，能够表达旧main crop外的坐标。
这只是执行器组件，未接入外部候选策略/完整tracker，未证明准确恢复或整机提速。
新check_evidence_query.py driver准备10个真实train帧、四路径4更新、两mode full16对照及strict重载。
driver R3发现anchor list不能传crop_pair、np.float32 geometry不能默认JSON；仅改np.asarray/.tolist，
R4 SOURCE_PASS/0block，完整R1–R4原回复/命令/封存已保存private forensic trace；
requested astra/max，actual routing UNATTESTED，同family/provisional，不把SOURCE当NN结果。
新20项源代码与B/manifest实际远端哈希一致，CUDA隐藏CLI实际exit0。
GPU1 task check_evidence_reader_20261014于06:12:46 BJT单次启动，wrapper1224324，
06:13:24 BJT原生exit0；首次按两分钟节点收集，未提前轮询。
真实11runthree的10帧覆盖RGB/TIR×128/256，4次decoder/head梯度检查及AdamW更新，
168非执行器状态逐位不变，58状态发生真实变化；同区域full16两mode与原correct输出逐位相等。
两项越界张量/原图框几何检查通过；保存后的strict重载5项预测逐位相等。
SANITY原字节SHA d3ec5de3...；组件耗时23.824秒不是推理速度或精度指标。
M0权重SHA2196f8dc...仅诊断，不用于正式初始化；旧17/B/保护CAD保持不变。
准入原生终态与完整原report见evidence_query_sanity_status和full_pipeline/evidence_reader_sanity_20261014。
后续在新包独立实现编码前外部候选请求、实际全图收益监督与在线tracker，先source review再新实验；
当前未训练完整新策略，最终78/72/62、效率及三个贡献仍未达成，原GPU0全882任务不能改或重启。

完整参考 test 为245序列/220703帧，PR/NPR/SR=73.916231/70.398889/58.587111。
对用户已重测baseline的数值差为+1.176231/+1.238889/+0.747111个百分点，未重复baseline精度测试。
仍未达到78/72/62；该参考C只完成20序列×2轮的轨迹训练，不能称全882最终模型。
两分片原生任务真实exit0，原本地控制器COMPLETE_PIPELINE；原controller.ExitCode仍null，
不由被动等待器exit0推断。原native11579与functions cell346均已终态，不再调用/重启。
指标/范围/限制见REFERENCE_FULL_TEST_RESULTS.md，原生证据reference_full_test_native.json。
参考test的fresh同family/provisional审计整体WARN，完整已有预测的纯CPU原工具包重算实际exit0，
245条GT/完整数组/曲线/全局及逐序列指标完全相同；确定性核验PASS，实际模型路由UNATTESTED。
保留原工具包口径与原CRLF metrics JSON字节，不改GT≤0的特殊处理；详见REFERENCE_FULL_TEST_AUDIT。
完整test均值105.942284ms/P95 163.019947ms/FPS9.439102，包含解码/初始化，排除模型加载；
FPS是跨分片累计服务时间口径。尚无baseline同条件配对延迟，不宣称明显提速或稳定提升。

全部882 C原计划、17项源码和成本0.05仍保持test前冻结。GPU0已通过实际输入sanity
exit0：10track/2update/257冻结张量与重载逐位相同。原owner16936已接续正式任务，
wrapper1069006/child1069007，04:30:06启动，预期917106track/182182update/1764初始化。
预计36–45小时，首必要诊断10-14 08:29:55 BJT，其后每4小时；当前TRAIN_ALL_882，未完训。
GPU1原owner47240也已接续冻结B收益诊断，wrapper1072641/child1072642，04:32:08启动。
真实输入sanity exit0：10track/2label/16pair/265冻结张量，原B两次加载预测逐位相同，0update。
正式20条诊断原首观测10-14 05:01:57 BJT；上述最新状态已记录终态及实际CPU分析，不再轮询原诊断。
04:39一次启动核验实际argv/CUDA/物理UUID与19项local/remote source、B和manifest SHA匹配，
两本地原owner同birth/argv存活，stderr为空；没有读取实时loss/帧进度或正在写的权重。
见c882_reward_formal_startup.json。勿重启上述控制器、任务或重跑输入检查。

dev是独立验证集97序列/54646帧，用于选择模型、查询预算和阈值；训练882，最终test245。
参考dev 78.906477/75.272253/62.796895不是最终test达标，也不直接与test baseline比较。
原冻结SHA84b67409...保留；test结果不用于训练标签、阈值/预算选择或权重挑选。
三个候选核心模块及完整整机效率仍待正式验证；goal ACTIVE/UNMET。

## 历史节点：截至01:42的记录，已由上述04:39状态更新

23:44 BJT 更新：三组完整97序列/54646帧dev已全部exit0，成本0/0.05/0.15的
PR/NPR/SR分别为77.671161/74.430912/61.729854、78.906477/75.272253/62.796895、
78.310464/74.689270/62.318010。按预定SR优先规则已选成本0.05，冻结权重SHA
94be6239f7f6864cc193fff7c21d67c9479924c9432039c6ff16cd0d1223f15f；原冻结字节SHA
84b67409e62593ac5865425c640466f11045a94feb04a1e623e5c259ef410480。
原始冻结文件不改字节；Git保存同内容的选择摘要并保留原始SHA记录。
这些是dev成绩，不与用户test baseline直接比较；参考C仍只训练20序列×2轮。
23:39原唯一控制器已启动完整245序列/220703帧test：GPU0 wrapper646863/child646864，
GPU1 wrapper646922/child646923；分片123+122互斥完整，argv/CUDA/物理卡及冻结SHA一致。
启动检查PASS，仅读身份配置，未读帧进度或精度。首次预计观测10-14 04:19:12 BJT，必要时240秒。
test指标尚未产出，不能写目标已达到。见FULL_DEV_FIRST_RESULTS.md、full_test_startup.json。
0/0.05平均131.829/131.826ms；0.15单卡106.051ms，与前两组并行负载不同，不据此宣称策略提速。
0.15的CPU完整轨迹覆盖诊断PID43924真实exit0，6029/54289有效帧的中心在源crop外，
平均原始IoU0.000065752；这不是工具包SR或动作反事实，EXPAND仍未扩大实际源视野。
M23/M25记录完整覆盖条件，M26记录局部辅助193token中129模板/CLS的重复执行问题。
实际核查固定MixFormer作者仓库commit8b87088f45af02c614a277176c63181261e93de3；
其不对称模板图支持KV缓存，本模型双向图不能直接等价照搬；未导入代码、未改当前模型。
22:03已通过crop API真实NN准入，265net值、11因果状态、2更新均一致，见CROP_RUNTIME_CHECK.md。
全882轨迹训练计划在test结果可见前已经冻结：从原正式B SHAba49b506...开始，
2轮917106track/182182update/1764初始化，预估36–45小时，每4小时诊断。
本地owner16936（22:22:53.625创建）仍WAIT_REFERENCE_PIPELINE、tasks=[]，882 NN尚未启动；
首次等待参考终态节点10-14 04:28:50.010 BJT。原控制器35896负责当前test，两个owner本次仍实际存在。
原11文件及新队列17文件源码封存保持不变，不把队列或SOURCE PASS写成训练完成。
22:45真实CPU参数清点：baseline103138437、ours103374215（+0.228603%），C训练参数221442。
参数未减少，真正成本优势及三个独立核心模块的有效性仍需同预算/同设备的正式证据。
详细全量协议见ALL_TRAIN_PROTOCOL.md；当前总体目标ACTIVE/UNMET。
00:11新增实际工程证据：sum crop纯CPU wrapper693753真实exit0/44.979秒，
80case逐位tensor/target/geometry/完整RNG相同；原/six/sum同次38.5716/21.1497/12.3381ms。
新候选在33保留均值case大幅减少API耗时，在47跳过均值case反而增加1.847ms；
只35/80优于six，混合平均减少41.6628%，不声明全场景提速或整机收益。NN准入尚未做，
没有接入当前test或全882封存配置。见SUM_CROP_PROFILE.md及M27。
同设备延迟脚本添加--full-dev，避免10×64前缀替代完整分布证据；SOURCE PASS/0阻断，
原conda CUDA隐藏CLI PID705573真实exit0，17项封存不变。正式配对NN仍NOT_RUN，
按FULL_DEV_LATENCY_PROTOCOL等待全882最终freeze与全部NN结束；见M28。

新增已完成dev诊断：纯CPU原生49400真实exit0/2.254秒，SOURCE正确性PASS/整体WARN/0block。
当前GT clamp规则在dev geometry下产生5488面积零目标，2019部分目标；是dev假想目标，不是训练标签。
有效GT的6430次模板更新事件中1053次rawIoU<.1（16.3764%），meanconfidence.818355；
460次GT中心在crop外、593次内部。只凭响应无法保证写入框质量，尚未证明更新导致后续失败。
M29/M30和CLIPPED_TARGET_DIAGNOSTICS记录范围、输入/源码SHA及下一步实际动作收益比较要求。
实际核查KeepTrack作者源码固定7eb9e74...，object_id候选关联与更新flag/样本certainty分开；
源码留私有，未导入/移植，不能视为原创核心；见STATE_UPDATE_REFERENCES.md。
原35896/16936本轮CIM确认同birth/argv实际仍在，04:19:12/04:28:50检查节点尚未到；
未提前读test帧进度或精度，不改17seal或全882计划，目标ACTIVE/UNMET。

01:06 BJT：针对M29新增冻结B实际动作收益诊断，SOURCE R1发现不存在manifest['test']的
KeyError，仅改为实际test_sequences字段；R2 PASS_SOURCE_ONLY/0block，同family/provisional。
19项local/remote源码与计划字节SHA核对一致，原17项与两个既有队列计划均未改。
远端CUDA隐藏CLI PID791546、local CLI PID33876真实exit0；不是NN准入通过。
唯一新本地owner47240（01:05:50.211创建）实际CIM命令/birth匹配，WAIT_REFERENCE_PIPELINE，
tasks=[]；首次04:30:50.010等待原参考COMPLETE后GPU1先10帧/2标签sanity，再20条完整冻结B诊断。
预期8854track/1752label/14016动作收益对，0优化器/0新权重，预计20–30分钟，首正式检查1800秒。
所有框都比较clipped、未截断和实际margin10执行误差；不以删除clamp直接当作改进证明。
没有提前查看参考test帧进度/精度，没有启动诊断NN，不改全882仍接续GPU0的既定流程。
初次metadata写入使用错误PowerShell布尔字面量，已仅修复回执，原47240从未重启，原null私有保存。
计划SHA f0edc89d...，源码77cdb406.../6aea54f4...，见REWARD_DIAGNOSTIC_PROTOCOL/REVIEW及M31。
用户问dev含义：dev是独立验证集97序列，只用于选权重/预算/阈值；882训练、245最终测试。
dev数值不是test成绩；最终仍需完整训练后的245序列PR/NPR/SR和真实整机效率。

本轮收益分析器准备：analyze_policy_rewards.py只接受正式20seq/8854track/1752状态/14016对结果，
拆分GT clamp与执行像素约束的收益变化，在实际选中主模态下比较同成本STOP/局部/完整查询。
SOURCE PASS/0block/整体WARN，同family/provisional；local CLI PID47124真实exit0，正式分析NOT_RUN。
signflip保留近零并报告幅度，regret不是长期轨迹上界，辅助token均值限定监督状态，不称整机速度。
源码SHA5439d432...；见POLICY_REWARD_ANALYSIS_PROTOCOL/REVIEW.md，19项原封存不改。
针对实际M23覆盖不足核查作者Uni-AdaFocus固定commit8846488310fdd4a18412608006030643e794c36e，
7文本逐字节SHA与真实git读取exit0已保存，未导入/移植/执行作者模型。
该分支确实先从输入图像采样固定96×96 patch，再ResNet50编码；可变源范围与重型输入预算分离。
global预览本分支是224输入，glance_size选项未使用；局部重型训练actions detach，另有廉价特征辅助监督。
整clip 3D卷积/插值不能直接在线；现有16×16scatter不能直接承载外部候选，扩大crop还须有效坐标/定位路径。
这是待实现设计约束，不是新增有效贡献；见PREENCODING_REGION_REFERENCES.md及M32。
01:09与01:22 BJT实际CIM核对35896/16936/47240三owner同birth/argv存活，未查参考test进度或精度。
文档follow-up SOURCE PASS/0事实blocker，reviewer官方7文件字节/commit/date复核一致；非NN验证。

01:32 BJT进入已核验等待：原35896同birth/argv实际存活，单一被动native session11579、
waiter50016（01:32:30.670646创建）已取得原35896进程句柄，执行Process.WaitForExit。
它不查远端或NN进度、不启动任务，原controller仍按04:19节点处理测试结果；
后续仅消费原11579，不重复创建waiter/observer，不因观察超时重启controller或NN。
启动回执见reports/reference_passive_wait_startup.json；动态等待回执在本地.codex_tmp。
01:42:47 BJT：现由唯一functions cell346接续同一native11579；它用异步计时器等到04:19:12，
随后等待原句柄，每次最多60秒、仍运行则再等180秒，间隔合计240秒。没有新增native waiter或NN任务。
后续只调用functions.wait(cell_id=346)，不要并行直接poll11579；终态输出包含原controller的exit码。
cell346只join原本地等待，不读取远端进度，现有16936/47240队列保持原检查时间。
全882训练、实际诊断、完整test正式目标和整机提速仍未完成，goal ACTIVE/UNMET。

以下历史段落中的NOT RUN/未选择/未收到以当时节点为准，当前状态以上述记录为准。

用户再次明确要求完整训练及完整指标。完整目标仍ACTIVE/UNMET，尚无新方法正式PR/NPR/SR。
当前最新：本轮A/B/C既定预算已经完成；C真实exit0及全量监督/冻结状态审计PASS。
19:08原控制器转入完整97条dev：GPU0成本0，GPU1成本0.05，两原子PID及实体卡已核实。
首dev检查21:23:10，其后必要240秒；cost0.15和全245test接续，不能提前报告测试指标。
权重绑定94be6239f7f6864cc193fff7c21d67c9479924c9432039c6ff16cd0d1223f15f，
成本尚未选择。新CPU裁剪候选只独立验证，不改当前11项活动源码。
本轮本地核验原uv31112/Python35896同创建时间与命令存活；未提前查询dev进度。
已完成C的567个中心越界监督帧集中在8/20条，三条贡献485/567（85.54%），
仅说明训练分布，不能推广完整dev。已准备独立CPU完整dev覆盖诊断，完整数据执行NOT RUN。
它统计区域内外原始IoU/中心误差及路由；相邻ROI中心重入不称准确跟踪恢复，
现有记录没有反事实动作框，因此不能据此重建辅助收益。详见FULL_DEV_COVERAGE_PROTOCOL.md。
独立源码审查WARN/正确性PASS/0阻断（同家族、provisional，路由未独立证明）；
实际本地Python3.12 CLI-help exit0及Python3.7语法模式通过，完整数据执行仍NOT RUN。
审查文档为FULL_DEV_COVERAGE_REVIEW.md。不把准备检查报告成完整评测成功。
本轮原控制器uv31112/Python35896创建时间/命令再次核实存活，stderr仍为空；未提前读dev结果。
20:11:02原conda纯CPU采样范围审计PID56099真实exit0/26.19秒，独立SOURCE正确性PASS/总体WARN/0阻断。
当前manifest重建A54,000/B50,000样本均涉及全部882条，但唯一搜索帧47,079/43,986，
占可采样目标10.32%/9.64%；A十路径各5400样本、各涉及879–882条。并非全帧遍历。
C20条覆盖不同非初始化帧8854/458553=1.9309%；全882两轮预计917,106次track尚未启动，
真正全882条闭环训练仍未完成。本轮审计不读取图像/权重或执行NN，9份输入元数据SHA与本地归档一致。
全882首框元数据均有限且正宽高，不为未发现的初始化问题加fallback。见SAMPLING_SCOPE.md。
M21进一步明确A/B裁剪构造将GT中心限制在名义[0.125,0.875]，而C已出现真实中心越界。
核查官方PySOT固定commit d04028f8e18c6b27ee150a7c0d8aab9a1807219e：SiamRPNLT确实
在lost时扩大原图裁剪与模型输入，借鉴范围控制，不照搬阈值就称创新；未导入/运行该代码。
当前跨区域稀疏修正、全图收益标签均未实现；完整dev/test和11份活动源码仍保持原值。
21:09:22新增六通道API纯CPU诊断PID236831真实exit0/33.148秒，源码正确性PASS/总体WARN/0阻断。
复用原已验证interior kernel，把两次连续数组复制计入调用；80真实case tensor/目标/几何逐位相同，
60搜索case完整RNG状态相同，模板case不使用RNG；两臂各320次平衡计时。
原/候选crop API均值44.066/24.298ms，P95 70.858/56.617ms，66/80case候选均值更低。
保留背景均值组40.851/40.796ms几乎无优势；这是组件44.8595%下降，非整机速度或精度。
与旧kernel-only均值的差不能单独归因复制，当前生产未接入/无新NN，见CROP_API_PROFILE.md。
资料与限制见SEARCH_COVERAGE_REFERENCES.md，原始源码仅留私有查阅目录，不上传作者整套代码。
下列保留本轮检查时间线。
18:13:26本轮实际核验：完整B五轮/6250更新已结束，shell exit0；原控制器已完成
B检查与新C sanity，两项wrapper COMPLETE/exit0。B执行器冻结与32帧原生路径检查PASS，
新C sanity从完整B出发，真实10帧/2次更新、257个冻结张量及严格重载逐位一致PASS。
正式C在18:08:17单次启动，wrapper3836285/子3836294，子进程CUDA_VISIBLE_DEVICES=0，
命令明确从完整B latest加载，不使用sanity权重。原本地Python35896同创建时间/命令仍存活，
state=train_c_complete，首检查节点19:08:06，随后必要时240秒；禁止另起控制器或提前读C进度。
这仍是固定20条训练完整序列两轮的轨迹刷新，不是全882条train逐帧训练。
完整dev/test尚未开始，无新方法正式指标。实际快照为complete_b_c_startup.json，
已完成的B CSV6250连续行、每轮1250行、全部四种loss有限；前/后100行meanloss
0.953150/0.650183，仅训练诊断，见stage_b_full_csv_audit.json。
18:14:15独立CPU裁剪对照实际exit0：40个dev裁剪/10条序列、120对计时，输入tensor/
目标/几何逐位一致；旧/连续数组crop均值46.766/37.609ms，P95 70.933/56.747ms。
候选准备与解码未计入，32/64线程、共享CPU负载仍存在；不称整机加速或精度保持。
新profile_contiguous_crops.py不被当前流水线调用，11个活动源码保持原值。
结果与审查范围在CONTIGUOUS_CPU_PROFILE.md及对应review/report，不改运行中的训练数学。
等待C节点期间仅分析已完成B日志：五轮total meanloss0.768393→0.651374，
quality/preview均逐轮下降，gain第二轮略升、最终0.094010；不作固定验证集泛化结论。
同一已完成32帧训练检查有31次局部请求，STOP仅1，均选RGB；不推广为全局路由塌缩，
也不能假定大部分完整帧会STOP。报告/完整损失图见STAGE_B_TRAINING_ANALYSIS.md。
原baseline实际预处理比ours少全图背景均值步骤，新增M17/M18记录此效率风险及路由边界。
独立compare_tracking_latency.py源码审查PASS，原conda CUDA隐藏CLI-help实际exit0，
NN配对测量仍未运行；先完成当前完整C/dev/test，随后同最终冻结权重与成本测量。
只比较原生系统服务时间，不重测baseline精度，不改活动11源码或控制器，详见
LATENCY_COMPARISON_PROTOCOL.md及其独立review。18:18:33原35896同birth/argv仍存活，
当前没有提前读取C训练进度，19:08:06首节点不变。
18:54:05 CPU三臂实测exit0/34.595秒：80个真实dev裁剪输入、目标、几何均逐位相等，
47个候选guard跳过均值、33个保留。原/连续/条件均值43.850/35.047/16.817ms；
各240次轮换计时，保留均值组guard有轻微代价。计时仍排除解码及连续数组准备，
不是整机或精度结论，不改当前11项源码。M19与INTERIOR_CPU_PROFILE.md保存全部范围限制。
新增终态C的CPU审计入口准备：COMPLETE/exit0后才核对固定20序列、所有监督顺序、
3504/17708计数、257冻结张量及8个Adam参数step；不得提前读正在保存的C权重。
此时尚未执行完整C终态审计，完整指标也仍待产出。
19:08:38原控制器已确认C终态：实际18:55:53结束/exit0，20train完整序列两轮、
3504监督更新/17708track调用。本轮A/B/C既定训练预算已完成，尚无正式dev/test指标。
19:10:20实际核验dev成本0/0.05原任务wrapper4038334/4038394、子4038335/4038395
均活着；命令同C latest，CUDA0/1，授权NVML的PID/UUID对应证明实体GPU0/1各一项。
本地原35896接续state=FULL_DEV_CALIBRATION，首dev节点21:23:10，之后必要240秒。
不启动第二观察器/NN或提前读取dev进度；cost0.15与完整245test仍由原管线执行。
终态C CPU审计实际工具exit0：真实manifest选序/全部3504监督三元组及连续step一致，
257冻结张量与完整B逐位相等，quality/gain共8参数Tensor实际变化，8个Adamstep全3504。
末轮权重SHA94be6239f7f6864cc193fff7c21d67c9479924c9432039c6ff16cd0d1223f15f。
CPUAudit不含NN重载前向；对应独立review与真实native报告分开，见STAGE_C_COMPLETE.md。
两轮监督meanloss0.797038/0.750061，不写成精度提升。M20记录567/3504中心越界，
loss把GT xyxy角点clamp到[0,1]，EXPAND仍同原搜索视野；需在完整dev核对覆盖/恢复。
以上路由与越界只统计监督帧，不能推广为全部17708帧，更不能据此提前改当前评测。
A扩大训练已终态：累计7250更新，当前run新增6250，十种模态/动作各625；
全部CSV步号/有限损失已复核，选中best为epoch7/7250、devloss0.9196168765。
原A/B的73461/30286原生句柄在当前上下文不可恢复，不虚构其exit0；
真实完成证据来自COMPLETE、CSV、最终权重元数据及主PID不存在。
A7 PRIMARY局部辅助的dev裁剪IoU RGB0.707919→0.723524，TIR0.615159→0.705281；
signed平均收益+0.068052/+0.298384，负收益比例45%/25%。这些仍不是正式OPE指标。

旧B使用A2，必须刷新。A7/batch8真实2次B更新与32帧闭环检查原生58478退出0，
250个执行器state张量不变，预览/质量/收益头实际变化，严格加载输出一致。
完整B在2026-10-13 17:04:58（服务器时间）单次启动，启动父PID3616703/训练子3616704，
GPU1；5轮×10000个抽样样本、batch8、6250更新，来源是原A7 best，不是sanity权重。
日志`runs/stage_b_refreshed_20261013.log`，实际退出码另存同名前缀exit_code.txt。
首次检查不早于17:49:58.7479835，之后必要时240秒；勿重复启动、提前读取训练进度。

C新增sanity入口复审PASS，GPU0原生97033退出0，实际10帧轨迹/2次更新，
257个冻结state张量不变、quality/gain变化、保存后严格重载输出逐位相同。
这次使用A7的B sanity权重，只验证执行；正式C将从已完成完整B重新开始。
评价入口原生9046退出0，真实16帧/一次初始化，SMOKE明确禁止当完整指标。
全test245条/220703帧GT已CPU核查：全部首帧原始float32相等、全部帧工具包round后相等。
工具包保留原代码，在本地uv Python3.12评分；服务器conda Python3.7.16不重建。

完整接续控制器已复审PASS并于17:27:12单次实际启动：本地uv PID31112，
实际Python PID35896，分别确认原生进程创建时间及命令行，state=WAIT_B。
owner在`reports/full_pipeline_owner.json`，状态`reports/full_pipeline_status.json`，
stdout/stderr为同目录full_pipeline_controller前缀日志；首次远端检查仍为17:49:58.7479835。
没有第二个观察器，不提前读B的日志或进度。11个部署源码SHA256与本地完全一致，
控制器固定SHA256为56ed5936b1d3c59f37741963c88a7b66201772f1c04912034c5e086cc2bbb58f。
顺序为完成B→真实B检查→
新C sanity→正式C固定20条train完整序列两轮（预期17708track/3504更新）→
全部97条dev/54646帧，成本0/0.05/0.15→按dev SR/PR/NPR/延迟固定预算→
锁定同一权重SHA与成本→全部245条test/220703帧，GPU0/1各一片→原工具包完整指标。
C的20条全序列轨迹不是全882条train逐帧遍历；不得作此扩展。
计时包含解码、初始化、裁剪、预览、路由、主/辅助编码、定位与模板更新，排除模型加载。
双片合并FPS按累计服务时间，不作双卡墙钟吞吐。baseline精度不重测，
相对baseline的速度优势仍须同设备/同计时范围实测，当前不宣称。

新增系统剖析：默认32/64 CPU线程，共64帧，frame9–64的56帧暖机解码55.1ms/跟踪109.4ms；
CPU裁剪是已测开销之一。单线程顺序对照受缓存/负载影响，不宣称已加速，
当前完整流程不改变正在运行B的数学或CPU线程。估计检查节点C60分钟、
每组dev135分钟、test280分钟，必要复查240秒；完整结果预计过夜，实际时间待校正。
详见`docs/FULL_TRAINING_PROTOCOL.md`、`docs/FULL_PIPELINE_REVIEW.md`及M12–M14问题记录。
接续代码及真实A结果已上传GitHub源码提交c723452，实际控制器启动登记随后文档提交同步。
首次读源校验会话63248退出-1且无输出，核验未创建控制器；原因不明，不称训练失败。
现有带120秒边界的remote helper随后完成同项只读校验，才首次启动控制器，原B没有重启。
17:32本轮已核验原uv31112/Python35896同创建时间/命令仍存活，WAIT_B首节点不改。
等待期间只分析已完成A7的1000条配对裁剪：来自70个序列/100个样本。
单模态GT诊断选60RGB/40TIR；主IoU≥.8时PRIMARY平均收益均负，
高GT覆盖仍有38/80与18/76负收益。结果及边界在
EXECUTOR_OPPORTUNITY_ANALYSIS.md和reports/stage_a_full/opportunity_analysis.json。
这些不作正式精度、真实轨迹上界或当前策略收益，也不改变正在训练的源或预算。
17:49:58原控制器首次检查后仍WAIT_B；前台仅等待该节点的functions cell66已终态，
没有另起后台观察器。17:52:03实际读取确认原B父3616703/子3616704均活着，
原命令与GPU1绑定一致；CSV4701条真实更新、step4700/epoch4，全部loss有限，
最近100条meanloss0.669448，最后一条0.384797。尚无COMPLETE或exit码，不提前称完训。
线性估计B结束18:07:34，仅估计；原控制器240秒检查不改，随后自动检查并接续C。
该实际快照为reports/stage_b_refreshed_first_node.json。新增参考读取已固定Where2comm
提交d3432523973b8c11df71e8e85750b2cb6fbe9dbb，三源码SHA记录于
reports/where2comm_reference_provenance.json，原源码只存本地private，不复制入项目。

## 路径

- 本地代码：`C:/Users/gb/projects/Dual-RGBT`
- 服务器：`/home/gaob/Dual-RGBT`
- 环境：`/home/gaob/Dual-RGBT/conda-envs/dual_rgbt`
- 数据源：`/data2/wangwj/dataset/LasHeR-0428/trainingset`及`testingset/testingset`
- 仓库：https://github.com/666666666666gao/Dual-RGBT
- 本地文档：`C:/Users/gb/Desktop/document/Dual-RGBT交接文档.md`

## 方法

廉价预览决定主模态；完整主路径之后预测各辅助区域的定位收益；
从原图取少量patch再共享编码并修正候选。核心是收益请求与局部执行的连接。
主模态质量、辅助价值与信息预算分开学习。三个模块仍是待验证候选贡献。

## 历史执行与诊断证据

已建立本地仓库与服务器项目；找到数据、权重和去模块CADTrack源代码。
CPU权重结构读取成功，保存`reports/checkpoint_structure.json`。
环境迁移完成，GPU0真实M0会话61116退出0：10组合实际更新、有限梯度、
严格重载逐位一致、STOP仅385个token主路径，局部辅助193个token。
耗时15.93秒，峰值2999788544字节；这是执行验证，不是推理提速指标。
独立代理逐字复现文档命令：GPU1会话80288退出0、同项检查PASS，耗时14.78秒。
独立源码审查PASS，0阻断/0非阻断，同family/provisional、实际模型路由不可独立确认。
源代码修复裁剪扰动与逐样本收益/区域覆盖记录，细节保留在审查报告及问题记录。
数据实测882 train/97 dev/245 test，帧数/GT审计issues为空。
阶段A于服务器时间2026-10-13 13:16:41单次启动：GPU0，原PID3035566，
源代码提交4ed99e2080fa9382a351c615b731449fd1987b6a；
2轮×2000样本、1000次真实AdamW更新，每种模态/动作100次。
13:29权威读取确认COMPLETE.json为epochs=2、steps=1000，PID已不存在，
CSV恰有1000条真实更新。后台启动未保存原生exit码，不虚构exit0。
这些是训练与校准诊断，不能替代LasHeR PR/NPR/SR。日志为
`runs/stage_a_diagnostic_20261013.log`，记录与逐样本结果在同名run目录。
第二轮同100个GT裁剪校准样本：RGB/TIR STOP mean IoU=0.689434/0.623036；
主候选局部补充为0.684997/0.619012，平均signed定位收益=-0.010859/-0.009632。
主候选ROI平均GT覆盖0.902/0.870，但尚无净改善；次候选分别纠正3/4、损害40/44。
原始逐样本CSV、步日志和两轮JSON已收回到本地reports/stage_a并保留服务器原件。
latest.pth为epoch2/1000；另存first_preserved.pth，实际核验也是epoch2，
只是第二轮副本，不宣称保留第一轮权重。原用户epoch14权重始终保留。
阶段B源码复审0阻断，但原生sanity和C真实轨迹尚未执行；正式D评价尚未执行。
固定epoch2证据诊断已完成，GPU1原生会话33980退出0，800个配对动作结果，
无优化器更新或新权重。池化归一化熵PRIMARY RGB/TIR为0.999060/0.998949，
真实摘要置零后loss反而低0.002779/0.005314；错配摘要只使框均值变化约0.0007。
次候选切换gain=-0.880998/-0.674639，修正gain=0.003180/-0.001131；
主要损害由未经验证直接切框造成。GT前景池化也未挽救固定MLP，不能只调温度。
已查阅OSTrack开源CenterPredictor：保持空间图并在响应峰处取size/offset。
下一步局部辅助token按原坐标填入辅助网格，与已算主特征复用原decode/head，
避免池化丢失空间定位及强制切候选；这仍是轻量空间融合。
空间执行器新上下文源码审查PASS，0阻断/0非阻断；GPU1真实M0会话45225退出0，
10路径更新、辅助原像素非零梯度、严格重载逐位一致、STOP只有385token路径；
总参数103374215，峰值3082387456字节，耗时14.12秒，仍不是提速指标。
同预算新A已单次启动，14:38:14服务器时间，GPU0主PID3242787，原生会话91336，
源提交3fdf8abfb2075435fc95b23e9fb221b88b3df01e；从原epoch14重启一项新实验，
没有重复或恢复旧A。两轮×2000样本/1000更新，首轮诊断预计14:43，完成约14:47–14:50。
14:38:44核验主进程及4个DataLoader worker活着。14:43首节点记录851更新。
原生91336（functions cell35）退出0，14:45权威报告PID不存在，COMPLETE
epochs2/steps1000、CSV1000行。第二轮PRIMARY gain RGB/TIR=+0.093309/+0.246714，
IoU 0.681044→0.713193 / 0.615028→0.691380，纠正4/7、损害0/0。
两轮原始结果在reports/stage_a_spatial；best真实为epoch2/1000/devloss0.978496。
这是同预算诊断正收益，不是正式PR/NPR/SR或完整在线速度。
新A保存latest及十路径dev定位loss最低的best；不兼容旧池化版state，
旧诊断与代码完整保留在41534b8及之前提交，不添加自动加载fallback。
当前尚无新方法LasHeR PR/NPR/SR或完整延迟指标；目标保持未达成。

## 原接续记录（已由最新状态更新）

旧A、固定权重诊断与空间短程A都已终态，不重复启动或轮询原91336。
A恢复源码审查PASS，新恢复路径M0待执行。通过后GPU0从原空间A latest继续
5轮×10000样本/batch8，新增6250、累计7250更新；继承AdamW与旧best。
预计45–60分钟，首轮后校正ETA；不从新M0权重续训。
Hann在线接入、C真实轨迹刷新、B M0检查源码复审均PASS，原生运行待完成。
C固定20条train两轮，17708次track/3504更新是预期；首帧实测均有效，
45帧非正宽高GT仅跳过监督，GT不维持后续搜索。
恢复M0原生70967退出0，batch8真实10更新，累计1010；B M0原生5090退出0，
真实2更新，250个执行器state张量逐位不变，preview/quality/gain均有实际变化。
两个严格加载实例predict逐位一致（不是保存前后比较）；32帧真实train闭环检查PASS，
Hann、有限框/history和实际1/2重型路径通过；不是精度或速度证据。
扩大A/B已单次同时启动于15:06:31服务器时间，源18a2493622577f1697ddbfb8eb0c72bcfdaf5a1e。
GPU0 A主PID3352461、原生73461、run stage_a_spatial_full_20261013，
从原空间A latest继续，非M0；预计15:51–16:06完成，首dev节点15:14再校正。
GPU1 B主PID3352473、原生30286、run stage_b_diagnostic_20261013，
从原空间A best重新开始，两轮×2000样本/1000更新，非B M0；预计15:16–15:20。
15:07:10一次启动核验，两主进程和各4个worker在运行，B已日志step50。
先在15:14估计节点查询，之后必要查询间隔180–300秒；不要重复启动两项任务。
待充分A选中执行器后刷新B/C监督；C完整刷新、dev校准、D与真实延迟仍未完成。
不得跳过M0，也不得把短程IoU、oracle或GT维持搜索的结果当作LasHeR正式成绩。

问题、修复与证据见`docs/ISSUES.md`；实验顺序见`docs/EXPERIMENT_PLAN.md`。
每次源码/结果更新后，将本文件同步到上述服务器、本地文档与GitHub三个位置。
