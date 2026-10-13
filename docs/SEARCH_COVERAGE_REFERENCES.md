# 搜索覆盖问题与开源机制核查

日期：2026-10-13。本轮没有引入下述代码或启动新训练。

## 我们的实际问题

`dual_rgbt/data.py`用GT中心加`u*side`产生A/B搜索中心，u各轴取[-0.375,0.375]。
GT中心在归一化搜索坐标中为`0.5-u`，名义范围[0.125,0.875]；浮点边缘存在计算舍入。
所以当前裁剪构造没有产生GT中心位于搜索区域外的训练状态。
这不表示所有GT框或像素完整可见；框边缘和源图填充仍可能不同。

C已观测567/3504个监督帧中心在搜索区域外。当前C只更新quality/gain，
执行器保持冻结；EXPAND与PREVIEW都从当前主搜索crop取辅助输入。
完整dev要确认覆盖问题是否普遍、与原始误差如何相关，不能先将这些帧全当作失败。
第一版C仅20条、1.93% train非初始化帧，也不能据此声称完整闭环训练已经覆盖全训练状态。

## 核查：PySOT的SiamRPNLT

已实际读取[官方开源仓库](https://github.com/STVIR/pysot)，固定commit
`d04028f8e18c6b27ee150a7c0d8aab9a1807219e`，通过实际`git ls-remote`取得。
三份固定版本原始源码与SHA在`reports/pysot_search_source_provenance.json`，原代码只留本地私有查阅目录。
GitHub未认证API本轮返回403，未获得其commit元数据；没有虚构发布日期。

[LT tracker](https://github.com/STVIR/pysot/blob/d04028f8e18c6b27ee150a7c0d8aab9a1807219e/pysot/tracker/siamrpnlt_tracker.py#L31)
根据上一状态切换搜索尺寸；低响应时进入lost状态，高响应时退出，中间保留状态。
它同时改变原图裁剪边长和送入模型的尺寸，lost时减弱窗口约束、低分时保持旧框状态。
[默认配置](https://github.com/STVIR/pysot/blob/d04028f8e18c6b27ee150a7c0d8aab9a1807219e/pysot/core/config.py#L269)
为正常255、lost831、阈值0.85/0.998；这是源码默认值，不断言所有实验配置都相同。
[原图裁剪实现](https://github.com/STVIR/pysot/blob/d04028f8e18c6b27ee150a7c0d8aab9a1807219e/pysot/tracker/base_tracker.py#L38)
确实按original_sz读取原图，随后才按model_sz缩放。

对我们有用的执行区别是：需要从当前原图搜索区域外读取信息时，必须改变源图读取范围，
多编码当前crop的token不能替代这个操作。这是根据源码和本项目数据作出的设计推论。
该开源路径本身不能证明RGB-T定位收益预测、稀疏辅助编码或本项目的效率收益；本轮未运行它。

## 对后续设计的约束

如果完整dev确认该问题，应将辅助查询的原图范围与token预算分别定义，
再用训练/dev构造外部候选及真实全图定位收益监督；不直接照搬lost831输入或两个固定阈值。
当前共享ViT位置编码/定位网格按固定搜索形状工作，借用更大源图范围还需要明确坐标映射和训练。
新的跨区域修正不能直接沿用当前16×16同区域scatter并宣称外部证据已可执行。

新路径仍应回答是否值得查、去哪里查、用多少token；与同预算置信度扩大搜索比较后，
才能讨论收益请求的独特作用。当前没有实现这些候选、收益标签或实验，
不把经典lost开关包装为本项目创新，也不提前修改正在运行的完整dev/test。
