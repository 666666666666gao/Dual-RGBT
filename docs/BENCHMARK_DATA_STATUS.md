# 三基准数据状态：2026-10-15

LasHeR 沿用已登记的训练882、dev97、test245协议。用户 baseline PR/NPR/SR=72.74/69.16/57.84，不重新测精度。

RGBT234 原图位于 `/data2/wangwj/dataset/RGBT234/RGB_T234`。CPU-only核验：名称集合与本地CADTrack工具包的234条一致；每条RGB/TIR图像数均与工具包标注行数一致，总116650帧；每条首尾两对图像尺寸一致。未解码全部中间帧，未运行跟踪评估。

两处标注文件字节哈希不同：`afterrain/visible.txt` 与 `woman89/infrared.txt`。独立float32数值核验表明woman89只是文件表示不同，数值及取整后数值一致；afterrain可见光标注的数值及取整后数值均不同。因此还不能称所有标注逐值一致。后续使用既有CADTrack工具包评分协议，明确记录初始框/评分GT来源，不能静默混用两版标注或改原数据。实际核验见 `reports/rgbt234_file_readiness_20261015.json` 与 `reports/rgbt234_annotation_numeric_20261015.json`。

RGBT210 已找到独立原图目录 `/data/wangwj/dataset/RGBT210testingset`，无需再提供路径。实际CPU准备核验210条、104706帧；每条visible/infrared图像数均与本地CADTrack工具包的GT行数一致，首尾两对图像尺寸一致。没有用RGBT234替代RGBT210；未解码全部中间帧，尚未运行完整跟踪评估。

第一次扩大目录检索误进入 `/data/lost+found`，实际PermissionError退出，原脚本/回执保留；随后仅列出命名数据目录成功，没有加入捕获异常、自动跳过评估序列或数据替代。以上操作没有NN、GPU查询或修改数据。

三个数据集可以按用户授权调整模块内参数/层数，仍先登记独立验证选型和测试冻结协议；RGBT210 PR/SR、RGBT234 MPR/MSR、LasHeR PR/NPR/SR各自保留原口径。当前新版三基准完整指标仍为空。

全部私有输入现已CPU实际准备并封印：LasHeR dev97/54646、test245/220703、RGBT210 210/104706、RGBT234 234/116650；另有首条LasHeR train的执行检查输入。各清单保存在服务器 `data/benchmarks/`，不上传仓库。公开汇总见 `reports/benchmark_manifests_preparation_20261015.json`。

RGBT210/RGBT234的初始化框和评分GT均采用本地既有CADTrack工具包标注，避免与服务器另一版afterrain标注混用。210/234 GT保留float32原值，预测框按原工具包取整；LasHeR保留其原GT取整与特殊GT处理。RGBT234仍取双模态最小中心误差/最大IoU的MPR/MSR口径，不改成单模态GT。

作者公开页面提供两个数据集的独立下载与工具包入口：[RGB-T datasets and benchmark code](https://chenglongli.cn/Datasets-and-benchmark-code/)。本次帧数和对应核验来自上述实际文件，不以页面代替数据检查。
