# sum crop CPU 候选源码审查

2026-10-14，fresh reviewer `/root/review_sum_crop_profile` 实际直接读取源码。
SOURCE_ONLY：整体 WARN，源码正确性 PASS，真正 blocker 0；允许进入原 conda 独立 CPU 实验。
此审查没有连接服务器、运行 NN 或修改文件，不是运行结果。
requested model `gpt-6-astra` / effort `max`，实际 routing UNATTESTED；
review_independence=same-family，acceptance_status=provisional。

脚本 SHA `dfee0ef8b544681db369c8b75c5cfab2fa726fcb2f2fba5aef04c22b783e13d0`。
其余四份依赖 SHA 见 `reports/sum_crop_profile_plan.json`，与审查直接读取的摘要一致。

候选保持原中心、尺度、RNG 调用顺序、float32 affine、插值、border mode、归一化及三元组接口。
uint8 输入且 H×W×255<2^53 时，通道整数和可以由 float64 精确表示；实际运行还将断言
OpenCV 求和除法与 NumPy mean 相等，并逐位对比最终 tensor、target、geometry 和完整搜索 RNG 状态。
没有网络构造/前向/优化器/生产函数替换；CUDA_VISIBLE_DEVICES 必须显式为空。
未发现现有 Python3.7、Torch1.12、NumPy1.21.5 的静态 API 不兼容。

计时覆盖每臂480次、三臂共1440次，候选连续化复制、求和/除法、warp、归一化、Tensor 构造
及搜索 RNG 初始化均计入。解码、六通道输入准备、case 构造与精确性断言排除。

审查限制：

- 历史33个保留均值 case 的约40.8ms是完整 crop API时间，不是 image.mean 单算子耗时。
- 新候选相对六通道候选同时改变均值实现与 guard，不能把差值全部归因于 sumElems。
- 六轮是三种循环轮转重复两次，各臂在每个位置出现两次；没有遍历六种排列或完全平衡两两先后。
- 如分析历史47/33分组，必须先匹配全部新旧 case 身份；整体均值不能代替分组结果。
- 当前参考测试共享 CPU 资源，此组件诊断不支持独占机器性能或正式显著性结论。
- 80 cases 不证明完整轨迹等价、完整系统加速、正式精度或核心模块有效性。
- 本审查仅确认候选不会改写17项封存；远程封存一致性仍由执行者部署时实际验证。
