# LasHeR 评价口径核对

2026-10-13已读取用户本地CADTrack随附工具包：
`C:/Users/gb/Desktop/document/cadtrack/toolkit/rgbt/metrics/metrics.py`、
`utils.py`及`dataset/basedataset.py`。尚未运行新方法正式test评价。

- LasHeR使用单组GT，不取RGB/TIR两套标注的最优值。
- `initial_result_file`以float32读取框并round(0)，GT和结果都经过此入口。
- 首帧结果替换为首帧GT，仍计入序列分母，不是删除首帧。
- PR使用中心距离≤20像素；NPR使用按GT宽高归一化后的中心距离≤0.2，
  不是整条NPR曲线AUC。
- SR使用IoU严格大于0至1的21个阈值，先算各序列曲线再整体平均。
- 每序列等权，不能改为所有帧合并后的长序列加权。
- 本工具包将GT任一分量≤0的帧误差设为-1；这会计入PR/NPR成功、SR失败。
  正式比较必须明确沿用这套已存在的口径，不能悄悄换成删除无效GT帧。
- 工具包会对非正宽高预测复用上一框。当前Tracker每步clip_box margin=10，
  应实际核验所有输出宽高为正；无需为不会生成的预测添加新fallback。

本地工具包`lasher_gt/10runone.txt`前两框与服务器test/init.txt的实读值一致；
全245条GT与序列顺序的一致性尚待核验，不能把两帧检查当成全数据证明。
这份口径来自提供的本地工具包，不宣称已经复现用户baseline的历史评价命令。
用户baseline PR/NPR/SR=72.74/69.16/57.84按要求沿用，不重新跟踪测试。

延迟最终按相同设备测完整系统：初始化、裁剪与归一化、传输、两路预览、
路由、主/辅助编码、定位头、模板更新全部计入；报告图像解码时间是否包含。
当前Tracker计时尚不含图像解码，初始化单独返回，最终统计须加回初始化。
还需测原去模块CADTrack延迟；此项不运行或报告baseline精度。
