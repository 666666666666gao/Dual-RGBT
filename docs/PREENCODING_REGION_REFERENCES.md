# 编码前扩大区域覆盖：Uni-AdaFocus 源码借鉴

实际问题来自已结束dev：成本0.05下6346/54289个有效帧的GT中心在原搜索crop外，
其中95.11%仍请求辅助；当前EXPAND只增加该crop内的tokens，不能提供外部目标证据。
这是覆盖与预算错配证据，尚未证明某一次辅助调用导致失败；见M23/M29。

核查[作者官方仓库](https://github.com/LeapLabTHU/Uni-AdaFocus)，固定commit
`8846488310fdd4a18412608006030643e794c36e`，2024-12-17T10:37:36+08:00。
论文为[Uni-AdaFocus: Spatial-temporal Dynamic Computation for Video Recognition](https://arxiv.org/abs/2412.11228)。
7份文本含MIT LICENSE的字节/SHA及真实git读取exit0记录在reports/adafocus_source_provenance.json。
源码留私有查阅目录，未导入、移植或执行作者模型；没有下载预训练权重。

## 实际可借鉴的执行边界

读取的是ActivityNet/FCVID/Mini-Kinetics分支的
[uni_adafocus.py](https://github.com/LeapLabTHU/Uni-AdaFocus/blob/8846488310fdd4a18412608006030643e794c36e/Uni-AdaFocus%20with%20Experiments%20on%20ActivityNet%2C%20FCVID%20and%20Mini-Kinetics/models/uni_adafocus.py)。
forward L154–159用MobileNetV2预览，再预测区域；L216–218在local_CNN之前从输入图像采样patch。
get_patch_grid_scalexy L10–24将位置及宽高映射为affine_grid/grid_sample，输出固定patch_size网格。
opts默认输入224、局部96；区域原图宽高能变化，昂贵ResNet50实际接收96×96输入。
本分支global_CNN直接读取224输入帧，不能因为opts有glance_size=96就宣称预览实际96。
这里输入已由数据预处理裁剪缩放，并非保证包含整幅原始视频帧。
据此可借鉴“源图覆盖范围与昂贵输入预算分开控制”，不能把改变源范围本身称为降低计算。

训练主局部路径L195–197的actions已detach；L175–187在廉价全局特征图上另建可微区域监督，
main.py L351–366使用分类辅助loss。因此不能将这个代码表述为重型局部分类梯度直接训练坐标。
它没有在本分支学习RGB-T实际执行框的定位收益；具体分类机制也不自动适用于跟踪。

## 对当前三个模块的约束

覆盖整段clip的抽样预览、双向时间卷积与跨帧插值使用整段视频信息，这是由代码推出的非因果性；
在线跟踪只能保留当前帧及历史。后续候选应让廉价预览覆盖主搜索范围之外，再决定是否值得请求，
以实际执行框的定位改善监督模块二；局部编码仍在昂贵计算入口控制固定tokens。
当前修正器将辅助特征scatter回16×16主网格，无法直接表达网格外候选。
所以扩大原图crop而不修改坐标映射和可执行定位路径不够，新的路径需要重新训练与原生验证。
以上是基于实际问题的待实现设计约束，非已验证的新方法贡献；原19项源码与全882计划不变。
