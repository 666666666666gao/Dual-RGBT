# UATrack：相对初始模板更新的源码核查

2026-10-15，只读核查作者仓库，固定 commit `a600bb29cc7a45e6a9fb2822aa08722ba99c4c71`（Git commit 时间为2026-01-04）。本文的“入口所选逻辑”指静态调用关系，不表示已运行成功或复现论文结果。没有下载权重、图像、数据或环境，没有执行第三方模型代码。

出版页确认论文为 *Uncertainty-Aware RGBT Tracking*，IJCV 134、article 290，2026-05-27发表；摘要明确提出比较候选与初始模板的相对质量，以及在相同搜索区域上的分类分数。本次只能读取订阅预览中的摘要和元数据，未读完整论文/PDF；具体条件以下列固定源码为准。[出版页](https://link.springer.com/article/10.1007/s11263-026-02878-0)

## 已核实：测试入口、配置与实际选择的类

| 调用环节 | 固定源码所选内容 |
|---|---|
| README → 测试脚本 | README指定`eval_rgbt.sh`；其唯一未注释测试命令为LasHeR、`--script_name bat --yaml_name rgbt`。[README][readme-test]、[脚本][eval] |
| 测试脚本 → tracker | `run_sequence`的`bat`分支调用`parameters(yaml_name, epoch)`、创建`BATTrack`；随后经`BAT_RGBT.track`调用该实例。入口虽然导入OSTrack，所选分支不实例化它。[实例化][entry-build]、[逐帧调用][entry-call] |
| 配置与checkpoint | 参数函数读取`experiments/bat/rgbt.yaml`；模板128、搜索256、stride16、CENTER head、CE keep ratios全为1。命令未指定epoch时，入口默认21并用于checkpoint文件名；YAML的`TEST.EPOCH: 30`没有代替该命令参数。[参数][params]、[YAML][yaml]、[epoch默认值][epoch] |
| tracker → 网络 | `BATTrack`调用`build_batrack(training=False)`并以`strict=False`加载`net`；构造器选择`vit_base_patch16_224_ce_adapter`，box head来自`lib/models/layers/head.py`，不是旁边的`max_head.py`。[加载][load]、[导入][model-import]、[构造][build] |

README称本工作为TUMFNet的扩展，并保留`drgbt/DRGBT603`训练示例；仓库`train_bat.sh`实际命令是`bat/rgbt`，训练分支连接`build_batrack`和`BATActor`。不能只按README中的名称或继承文件名认定运行机制。[README][readme-lineage]、[训练脚本][train-entry]、[训练分支][train-build]

**公开源码的可运行性疑点：** `vit_ce_adapter.py:28`和`attn_adapt_blocks.py:9`无条件导入`CrossModal_ST_Fusion_with_uncertainty`，但固定版本的`uncertainty_fusion.py`全文只有`CrossAttention`、`SelfAttention_For_Fusion_uncertainty`、`TemplateRouter`三个类，没有定义或导出该符号。正常按这些源文件导入会存在缺失符号问题；本次没有执行导入、修复作者代码或核验发布checkpoint的key。这是该公开快照的静态疑点，不是论文或发布权重无效的证据。所以下文只能确认公布入口指向的实现逻辑，不能担保这份源码原样可运行，或等于论文最终评测版本。[导入1][vit-import]、[导入2][block-import]、[被导入文件全文][fusion-full]

## 已核实：候选产生、比较与写入时序

1. 初始化保存同一份预处理模板为`initial_template`和`last_template`，`new_template=None`，`interval=10`。这些是图像tensor；这里的更新是模板状态替换。[初始化][init]、[六通道预处理][preprocess]
2. 每帧将第0个预测的分类图乘Hann窗，取峰值`max_score`并确定当前框，然后把该响应加入`max_score_list`。仅当当前响应严格超过这个列表的均值时，才用当前预测框裁下的模板覆盖`new_template`。覆盖语句在本帧比较/写入之后，因此比较使用进入本帧前已有的候选。均值包含当前帧；生效代码不清空响应历史，所以它是累计均值，候选是**最近一次达标模板**，不是最近窗口最高分模板。窗口最高分/随机选择分支均已注释。[响应与定位][response]、[注释分支与候选条件][candidate]
3. 第11、21、31……帧，若已有候选，网络输入从普通的2个样本扩为4个。前两项用于当前搜索，后两项共享`new_search`，分别检验初始/候选模板。[组批][batch]

| 比较节点的batch下标 | 第一模板 | 网络补入的第二模板 | 搜索图像tensor |
|---|---|---|---|
| 0 | 初始 | 上次已接受模板 | 当前搜索 |
| 1 | 上次已接受模板 | 初始 | 当前搜索 |
| 2 | 初始 | 初始 | 同一个缓存搜索 |
| 3 | 候选 | 初始 | 同一个缓存搜索 |

这个表同时依据tracker的组批与backbone的模板重排；比较项不是两次互不关联的搜索，也不是单独评估未经上下文交互的模板。[组批][batch]、[第二模板重排][reorder]

4. `new_search`不是当前比较帧的搜索。代码每5帧向累计列表追加搜索tensor，每10帧取`int(len(list)/2)`项；只把较早项改为`None`，不缩短列表。按这段索引算术，11/21/31帧的比较分别使用10/15/20帧的搜索。它不是“每10帧重置窗口后取中点”。这是静态时序推导，未执行模型。[缓存索引][cache]
5. 写入条件是 **候选相对质量高于初始，或候选分类峰值不低于初始**：即`relative_score[1,1] > relative_score[1,0]` **OR** `best_score[3] >= best_score[2]`。不是AND，也没有在此处额外要求绝对响应达到0.65或不确定性低于某阈值。通过后把候选赋给`last_template`；拒绝时保留原状态，候选本身没有在该节点清空。这里比较的分类峰值同样来自Hann窗处理后的响应。[条件与赋值][gate]、[响应][response]
6. 当前输出框先由batch第0项确定，之后才执行上述赋值；该赋值供后续帧组批使用，没有“辅助证据不可信就退回本帧主路径框”的显式输出验收分支。不要因此进一步断言四项比较批次对当前预测毫无影响，见下方成本与对照边界。[当前框][response]、[赋值顺序][gate]、[后续组批][batch]

## 已核实：相对质量需要哪些信息

`TemplateRouter`输入来自完整backbone之后：RGB/TIR特征分别归一化，重新拼接后相加，取前64个模板token，按相邻batch两项组成一对。每个模板做全局平均池化，再过共享`768→64→1` MLP，两个标量做softmax得到相对质量。它是依赖搜索和模板交互后的学习分数，不是原图清晰度、单模态独立质量、直接IoU，或编码前可用的廉价信号。[编码结束与router输入][router-input]、[router结构][router]、[维度][dimensions]

训练代码使用同一搜索GT对应的两次预测IoU比较产生标签：较好项为1、较差项为0，平局为`[1,1]`；router损失权重0.01。实际实现把已经softmax的`relative_score`送入`BCEWithLogitsLoss`，不是把原始`score`送入标准pairwise softmax交叉熵。这里按源码记录，不替作者更正或声称分数已经校准。测试更新分支不读取逐帧GT；GT监督需求在训练端。[criterion][criterion]、[IoU标签与loss][router-train]、[测试调用][entry-call]、[更新条件][gate]

## 成本边界与对本项目的含义

| 信息/计算 | 源码证据与成本含义 |
|---|---|
| 两种模态的完整特征 | 输入六通道先拆成RGB/TIR，再分别做patch embedding和12层处理；配置没有删减搜索token。router出现时这些计算已经完成，不能把其分数或融合权重称作“避免了辅助编码”。[输入拆分][encoding-input]、[12层循环][encoding-loop]、[YAML][yaml] |
| 初始/候选的公平搜索比较 | 比较节点把batch 2扩为4，额外两项也进入完整backbone及CENTER head。缓存的是预处理图像tensor，并非复用已完成的特征；不能只计算小MLP的开销。[组批][batch]、[缓存][cache]、[全batch解码][decode] |
| 小质量头与状态维护 | router本身是池化和小MLP；按层维度静态计算为49,281个参数，不含骨干。实际路径还包含模板裁剪/预处理、历史缓存、`.item()`取分数，以及每10帧的垃圾回收/清缓存调用。本文没有测量时延、显存或FPS，也不把batch项数增加直接换算为运行时间。[router][router]、[维度][dimensions]、[裁剪][candidate]、[缓存维护][cache] |

另外，融合权重的`min()`/`max()`没有指定维度，是对整个输入张量取极值。**由源码推断**，在同一次forward中加入两个比较样本可能改变当前样本的归一化尺度；若未来要隔离“仅写入规则”的效果，应固定同一比较批次与计算路径，不能只凭写入发生在定位后，就声称当前输出必定不受比较计算影响。尚未做数值或闭环验证。[融合归一化][fusion-normalize]

本项目当前规则是在每5帧、最终响应≥0.65时更新模板；已有C20 dev证据中，1053次低IoU写入有1044次前一帧已低IoU，851次此前已连续低IoU至少20帧。这说明大量写入发生在已有漂移中，尚不能证明模板写入引发或延长漂移，也不能推断单加相对质量门就能恢复目标。[现有时间证据](C20_STATE_WRITE_TEMPORAL.md)、[四例目视记录](C20_STATE_WRITE_VISUAL_NOTES.md)

## 推断与待验证对照

- **可直接借鉴的普通对照思想：** 保留初始模板作为锚，在相同搜索证据上比较候选与初始的任务响应/相对质量，然后决定是否写入。这个思想已有明确先例，不能把“相对初始模板质量门”本身作为本项目新贡献。上面的OR条件、候选维护和缓存时序才是该公开版本的具体实现；改成AND、当前搜索、窗口最高分或新阈值时，应明确登记为变体。[作者摘要](https://link.springer.com/article/10.1007/s11263-026-02878-0)、[源码条件][gate]
- **本项目尚需区分的动作：** 查询前是否值得付出辅助编码成本、查询后是否接受定位证据、是否把结果写进状态。UATrack这段代码可支持第三项的普通参考对照；它没有提供本项目所需的主路径与辅助查询后输出之间的拒收对照或闭环收益证据。上述三个决策的收益仍须由本项目实际执行后的误差、身份保持及成本验证，不能由router分数代替。[本项目既有范围](STATE_UPDATE_REFERENCES.md)、[源码输出/写入顺序][response]、[源码条件][gate]
- **后续实验仍未实施：** 先完成已登记正式训练与评测；随后在相同执行器、候选预算、训练量及可用信息下比较当前规则、关闭动态模板写入、普通相对初始模板门、查询后验收与写入策略。采用学习质量头时，要登记自己的训练监督、容量及额外前向成本；不是把作者权重直接换入本项目。分别观察失败进入、漂移持续和稳定恢复，最终仍需验证LasHeR三指标各提升6个百分点的目标（PR/NPR/SR≥78.74/75.16/63.84）、RGBT210/RGBT234，以及完整系统平均/P95时延。保持既定三模块职责与CADTrack移除论文三模块/SOT初始化来源；本次没有修改代码、配置、阈值、权重或正式队列。[现有对照约定](C20_STATE_WRITE_TEMPORAL.md)、[项目约定](../AGENTS.md)

私有源码、Git commit/tree元数据及逐文件Git blob/SHA-256校验：`C:/Users/gb/.codex_tmp/dual_rgbt_uatrack_reference_20261015/source_provenance.json`。GitHub API曾限流，commit最终由只读`git ls-remote`及不检出文件的partial clone固定；只下载明确选取的源码文件，没有将作者源码复制进本项目实现。本文是机制参考与创新边界记录，不是独立结果验收。

[readme-test]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/README.md#L41-L44
[eval]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/eval_rgbt.sh#L1-L16
[entry-build]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/RGBT_workspace/test_rgbt_mgpus.py#L112-L115
[entry-call]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/RGBT_workspace/test_rgbt_mgpus.py#L124-L161
[params]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/test/parameter/bat.py#L7-L30
[yaml]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/experiments/bat/rgbt.yaml#L35-L79
[epoch]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/RGBT_workspace/test_rgbt_mgpus.py#L164-L175
[load]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/test/tracker/bat.py#L20-L35
[model-import]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/bat/ostrack_adapter.py#L11-L14
[build]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/bat/ostrack_adapter.py#L190-L237
[readme-lineage]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/README.md#L15-L43
[train-entry]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/train_bat.sh#L1-L5
[train-build]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/train/train_script.py#L52-L76
[vit-import]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/bat/vit_ce_adapter.py#L15-L28
[block-import]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/layers/attn_adapt_blocks.py#L1-L9
[fusion-full]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/bat/uncertainty_fusion.py#L1-L134
[init]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/test/tracker/bat.py#L61-L117
[preprocess]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/test/tracker/data_utils.py#L15-L24
[response]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/test/tracker/bat.py#L189-L222
[candidate]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/test/tracker/bat.py#L226-L283
[batch]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/test/tracker/bat.py#L154-L195
[reorder]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/bat/vit_ce_adapter.py#L339-L369
[cache]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/test/tracker/bat.py#L284-L300
[gate]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/test/tracker/bat.py#L257-L283
[router-input]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/bat/vit_ce_adapter.py#L433-L500
[router]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/bat/uncertainty_fusion.py#L99-L134
[dimensions]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/bat/vit_ce_adapter.py#L535-L541
[criterion]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/train/actors/bat.py#L19-L27
[router-train]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/train/actors/bat.py#L214-L257
[encoding-input]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/bat/vit_ce_adapter.py#L234-L255
[encoding-loop]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/bat/vit_ce_adapter.py#L383-L436
[decode]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/bat/ostrack_adapter.py#L100-L148
[fusion-normalize]: https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/bat/uncertainty_fusion.py#L70-L92
