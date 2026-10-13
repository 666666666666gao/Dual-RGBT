# 开源参考与使用边界

## 已实际遇到的问题

- 搜索裁剪扰动与监督不一致：对照项目现有CADTrack训练处理与配置，
  将中心扰动从±0.75边长改为±0.375边长。代码来自本服务器去模块CADTrack，
  原作者仓库[IdolLab/CADTrack](https://github.com/IdolLab/CADTrack)已核验可访问。
  这项修复是训练正确性，不算论文贡献。
- 原方法两路权重本就共享：当前250个state张量与源码直接证明共享，
  所以本项目不把共享称为相对它参数减半。待测收益来自减少实际执行。

## 可执行设计借鉴（仍需本项目实验）

[Uni-AdaFocus](https://arxiv.org/html/2412.11228v1)及作者提供的
[AdaFocus开源仓库](https://github.com/blackfeather-wang/AdaFocus)、
[AdaFocusV2](https://github.com/LeapLabTHU/AdaFocusV2)、
[Uni-AdaFocus](https://github.com/LeapLabTHU/Uni-AdaFocus)：轻量预览之后选取原图区域，
再将更小输入交给重型编码。论文和AdaFocus仓库页面已核验；本项目没有运行其上游代码。
我们的执行入口采用原始16×16像素patch gather→共享投影→ViT，局部搜索64个token，
而不是先编码完整辅助图后再ROI选择。跟踪位置编码和定位收益监督是本项目待验证适配。
不能引用视频分类提速作为本方法提速的实验证据。

[SemARC](https://arxiv.org/abs/2610.07466)摘要已核验，按收益顺序获取模态已有分类近邻；
其GitHub实现尚未核验，因此不列为已读开源代码，不据此宣称广义按需模态获取新颖。

后续定位失败、收益预测不准或速度反而下降时，在此按真实问题追加开源仓库、
具体读取文件与机制，再描述如何适配；不先堆叠未验证模块。

## M04/M08：实际负收益后的空间证据读取

2026-10-13已读取[OSTrack的head.py](https://github.com/botaoye/OSTrack/blob/main/lib/models/layers/head.py)
中的CenterPredictor/get_score_map/cal_bbox：保持二维特征，通过中心响应峰取得该位置
的尺寸与偏移；本项目原CADTrack head也具有相同结构。未运行OSTrack上游训练。
当前MLP的辅助池化熵约0.999、摘要置零无损，且直接次候选切换造成主要损害，
因此下一版不继续放大无空间摘要MLP，而将实际编码的局部辅助token按原位置放回
辅助网格，未请求位置置零，复用已有decode和CENTER head在主/辅空间证据上重新定位。
这是针对当前失败的工程适配，不宣称CENTER head或稀疏网格本身是新颖机制。
省计算的来源仍是辅助主干只处理193个token；小定位头处理16×16网格的开销须计入实测。

## M14：实际系统开销后的裁剪参考

2026-10-13读取作者仓库[OSTrack processing_utils.py](https://github.com/botaoye/OSTrack/blob/main/lib/train/data/processing_utils.py)
的`sample_target`：先提取目标搜索区域，再补边和resize。当前本项目的`crop_pair`
在完整输入上求两路背景均值并warpAffine；64帧剖析中其累计时间约4.82秒，
因此后续效率优化需要检验CPU裁剪开销，而不是只缩减ViT token。
两者的边界填充值、整数裁剪和坐标变换不同，直接替换会改变输入；
当前完整训练管线保持原数学，后续以配对执行和实际精度验证决定是否采用。
这是读取上游开源代码后的候选工程改进，未运行OSTrack原项目，也没有本项目加速结论。

## M15：区域覆盖与定位价值分离的开源核查

2026-10-13核查作者仓库[MediaBrain-SJTU/Where2comm](https://github.com/MediaBrain-SJTU/Where2comm)，
实际读取了[Communication](https://github.com/MediaBrain-SJTU/Where2comm/blob/d3432523973b8c11df71e8e85750b2cb6fbe9dbb/opencood/models/comm_modules/where2comm.py)、
[PointPillarWhere2comm](https://github.com/MediaBrain-SJTU/Where2comm/blob/d3432523973b8c11df71e8e85750b2cb6fbe9dbb/opencood/models/point_pillar_where2comm.py)
和[where2comm_attn.py](https://github.com/MediaBrain-SJTU/Where2comm/blob/d3432523973b8c11df71e8e85750b2cb6fbe9dbb/opencood/models/fuse_modules/where2comm_attn.py)。
该公开路径先完成各节点特征提取和检测置信图，再生成空间通信mask，
在已编码特征上施加mask并聚合。可参考空间信息选择与预算记录，
不能将这条路径的通信节省解释为免去辅助主干编码。

此次读取的Communication.forward具体使用置信图及阈值构造mask。
需求与提供者置信度乘积只出现在该文件注释；本次不能宣称这段公开执行代码
已实现逐候选定位收益监督或显式接收方请求。该范围也不能代表论文全部变体。
没有运行或移植上游代码。
本次引用固定为提交`d3432523973b8c11df71e8e85750b2cb6fbe9dbb`，已读回三份源码并记录
各自SHA256及字节数，详见`reports/where2comm_reference_provenance.json`。

当前本项目M15的覆盖充分却负收益问题，要求用实际框误差下降训练价值，
而不只照搬空间置信度阈值。已有signed收益训练继续执行；若完整闭环失败，
再用同预算空间置信度查询作为对照，检验我们的价值预测是否提供额外作用。
