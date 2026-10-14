# 三基准数据状态：2026-10-15

LasHeR 沿用已登记的训练882、dev97、test245协议。用户 baseline PR/NPR/SR=72.74/69.16/57.84，不重新测精度。

RGBT234 原图位于 `/data2/wangwj/dataset/RGBT234/RGB_T234`。CPU-only核验：名称集合与本地CADTrack工具包的234条一致；每条RGB/TIR图像数均与工具包标注行数一致，总116650帧；每条首尾两对图像尺寸一致。未解码全部中间帧，未运行跟踪评估。

两处标注文件字节哈希不同：`afterrain/visible.txt` 与 `woman89/infrared.txt`。独立float32数值核验表明woman89只是文件表示不同，数值及取整后数值一致；afterrain可见光标注的数值及取整后数值均不同。因此还不能称所有标注逐值一致。后续使用既有CADTrack工具包评分协议，明确记录初始框/评分GT来源，不能静默混用两版标注或改原数据。实际核验见 `reports/rgbt234_file_readiness_20261015.json` 与 `reports/rgbt234_annotation_numeric_20261015.json`。

RGBT210 尚未在已列明的数据目录中定位到独立原图，已向用户请求路径。本地工具包210个名称都是234名称子集，但147条同名GT行数不同；210标注共104706帧，不能直接拿234完整图像配210标注。须确认独立数据或已有的真实帧对应，再完整评估。

第一次扩大目录检索误进入 `/data/lost+found`，实际PermissionError退出，原脚本/回执保留；随后仅列出命名数据目录成功，没有加入捕获异常、自动跳过评估序列或数据替代。以上操作没有NN、GPU查询或修改数据。

三个数据集可以按用户授权调整模块内参数/层数，仍先登记独立验证选型和测试冻结协议；RGBT210 PR/SR、RGBT234 MPR/MSR、LasHeR PR/NPR/SR各自保留原口径。当前新版三基准完整指标仍为空。
