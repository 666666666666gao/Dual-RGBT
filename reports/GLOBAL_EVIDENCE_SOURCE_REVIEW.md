全图候选实验独立源码审查（SOURCE_ONLY，R1 + 已发布源码验证器增量）

结论：SOURCE_PASS；blocking_issues=0。只适用于本报告封存的30个源码、计划与实际helper路径。总体标记 WARN_SOURCE_ONLY_PROVISIONAL：真实数据NN准入尚未执行；requested Astra/max，actual routing UNATTESTED，same-family/provisional。

计划SHA256：b5cf7d472bb1a09787d57a1a15f95cdd9989f0b102908189a74c975dbe7daafd。初始36输入seal保留原文件，新增只读验证器另建increment_1 seal；最终37个原输入与副本均按SHA复核一致。原25源码及原17运行相关代码在本次审查中没有改动；没有操作原GPU0任务。

审查结果：

- 全部30源哈希与计划一致，初始33个Python文件AST解析通过；新增验证器也解析通过。43项静态检查与9项标准库几何/增量检查全部通过。数值复算采用人工构造坐标，没有读取数据、执行项目模块或模拟NN结果。
- Main64保持8×8预览；global256直接调用共享Preview.net，三个stride2卷积产生32×32，没有被Preview.forward缩回64。quality输入74，gain上下文913，区域/动作字段42，总输入955。global ROI采样保持四个动作各3×3、32通道，布局与继承收益头匹配。
- 全图方形源为[0,0,max(H,W)]。32×32候选中心按实际图像H/W屏蔽填充。8组方形/横纵矩形的合成检查把最大原始分数放在padding，mask后候选仍全部落在原图内。此项验证候选中心，不声称每个卷积感受野不接触padding。
- 三个local原图源边长S/2、128输入、193tokens；global-wide边长S、256输入、385tokens；两种global动作共用全图峰中心。既有sample_main_features按两个独立原图几何投影，域外主特征归零；辅助query仍能在自己的8/16网格定位。400组几何复算最大投影误差4.55e-13、框往返误差4.27e-14；这是双精度代数检查，不是torch浮点执行测量。
- 监督热图按实际score边长生成，覆盖main8、query8/16、global32。目标中心不可见时为背景监督，框回归以visible屏蔽；GT框没有被截到主搜索内。
- A先strict加载原B执行器，再以同键EvidencePolicy net strict加载已完成旧正式A（d1981fef9efe797e14627515fe3b8e49c235efbe1d548a25a1af7a8c8412ca07）。Global子类不新增参数，不以sanity权重初始化正式A。B/C只能读取同formal或sanity家族的前序权重。实际checkpoint字节及strict key匹配仍由运行入口核验。
- A仅global-local/global-wide允许GT附近教师。6250更新使RGB/TIR×5每路恰好5000样本；每个global路有2504教师、2496候选样本（50.08%）。sanity只有一个10步周期，四个global路批次均是教师；部署候选另外由B反事实和reload probe经过。
- B用execute_boxes对主路径和四个query统一执行margin10/float32原图框，再形成IoU差标签，STOP收益/成本均0。C在tracker.track之后才构建当前GT标签，保存track前模板和历史，逐位比较所选反事实与真实执行框。在线track不接受GT参数；未查询时不写辅助模板。
- 计划列出882个不重复名称，C两轮全序列、每5帧有效GT监督。未读取实际manifest，因此不将名单与manifest对应关系或sanity确切10帧报告为已实测。
- A只训练decode/head；B训练preview/quality/query_gain；C训练quality/query_gain。sanity检查每个所选参数梯度存在且有限、每个前缀聚合梯度非零、每个前缀有参数变化、冻结state逐位不变及optimizer step一致。reload为strict加载后的完整单一global-local前向（含quality、gain、query及主/辅助框）；没有声称所有动作都逐位重载测试。
- 新query_geometries四个调用点都补齐image_size，reload probe也包含image_size。builder、counterfactual/policy loss和CLI参数没有发现导致调用报错的接口不匹配。

实际执行路径及变更：

1. 原dual_rgbt_deploy_global_evidence_20261014.py封存未改。它在远端要求new5不存在并mkdir(exist_ok=False)。父代理随后报告已发布new5和plan，因此此路径当前前置条件不满足；必须不再调用。该约束是部署状态变化，不是模型缺陷，也未通过修改旧helper隐去。
2. 新dual_rgbt_verify_published_global_source_20261014.py已在increment_1单独封存。格式化后的远端程序只核验plan/source30/B/initialA/manifest，隐藏CUDA执行CLI --help，未写源码；其格式化脚本也已AST解析。审查者没有执行该helper或CLI。
3. 实际路径为“只读发布验证器 → 三阶段serial GPU1 sanity helper → formal A helper”。serial准入要求native exit0、10/2/2更新及strict_reload标记；formal A要求三阶段完成且配置计划SHA一致，argv不带--sanity或--policy。继承launch仅检查指定GPU显存，不查询温度/功率；run_task设置PCI_BUS_ID/physical1并记录真实退出码。
4. 本报告helper_sha256包含历史和实际四个helper，执行时只能用所列精确字节。helper自身目前只核验计划/模型源SHA，没有self-hash校验，调用方应核验本报告helper哈希。新正式C不在当前launcher中，仍须按协议在启动前协调原控制器后续双卡dev/test。

四个helper SHA：

- C:/Users/gb/.codex_tmp/dual_rgbt_deploy_global_evidence_20261014.py: d8116051ae0964a089a7733fee2c20892ad1939924d372be868105d8f0a3e2a2
- C:/Users/gb/.codex_tmp/dual_rgbt_admit_global_evidence_stages_20261014.py: 06495b21fd664b003d468be3650d827adaa5ba28b541bfc64f4c1bf22cb09c32
- C:/Users/gb/.codex_tmp/dual_rgbt_launch_global_evidence_formal_a_20261014.py: a69e5cc7a76132de2156cb847236741a8e110e777a7c72dc14112c407ff2cc2c
- C:/Users/gb/.codex_tmp/dual_rgbt_verify_published_global_source_20261014.py: 29c741c2e0c9fa58e8a704c30284d556b8ccaa8324843d4fdda906204a317eda

边界与原始记录：

没有导入torch、构建模型、执行NN、加载权重、读图、读test标注、SSH、GPU查询、安装或修改项目/helper源码。名义覆盖诊断3001/1986来自给定协议，未在本轮重复其数据计算，也不证明全图候选精度、效率或三个有效机制。

保留了审查者自身两类失败：最初PowerShell上下文读取命令的管道语法错误，以及geometry_increment_review.py的else256拼写SyntaxError。后者原v1、exit1和修正v2全部保留；v2 exit0。这些不属于被审查项目的故障。最初辅助上下文读取超出工具输出预算，原截断返回仍保留，另存完整replay；全部科学输入及科学检查输出均已单独完整落盘。

证据文件：input_seal.json、input_seal_increment_1.json、static_checks.json、geometry_increment_checks.json、native_commands.jsonl、command_log_full.json、checker_failure_1.json、原始raw*.txt、两个geometry检查器版本、格式化远端程序、代码delta文本。最终artifacts_seal.json封存交付产物。此结论不替代后续真实前向/反向/更新/重载准入。
