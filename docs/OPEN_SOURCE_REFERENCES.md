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
