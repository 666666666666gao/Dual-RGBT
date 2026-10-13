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
