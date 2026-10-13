# 完整 dev 搜索覆盖离线诊断

准备日期：2026-10-13。成本0与0.05两组完整dev已回收，离线分析均真实exit0。
实际结果见FULL_DEV_FIRST_RESULTS.md和对应coverage/native JSON；成本0.15仍待原管线完成。
语法、CLI-help与源码独立审查分别记录，不替代实际全数据证据。

## 问题与可用证据

已结束 C 的3504个监督帧中，567帧记录目标中心在当前搜索区域外。
进一步按序列统计，20条中8条出现此现象，12条没有；三条序列贡献485/567（85.54%）：

| 序列 | 监督帧数 | 中心在搜索区域外 |
|---|---:|---:|
| leftblkboyunderbasketballhoop | 624 | 200 |
| left_first_0109 | 262 | 187 |
| easy_rightblkboywithgirl | 166 | 98 |

原始CSV SHA256为`ff4d04fc18a262ae590a4fab5fdcee582cb2750693afcf098379c7b42270ecdc`，
全部20条统计在`reports/completed_c_coverage_concentration.json`。
这是训练中的监督帧分布，不是全部17,708次track，也不能推广为完整dev的覆盖比例。

当前`EXPAND`仍编码同一搜索crop的全部辅助token，不扩大原图搜索视野。
因此先检查完整dev的覆盖与原始定位误差，再决定是否修改候选或收益监督。
原始评价只保存最终选中框，没有保存每帧主预测及全部替代动作预测；
不能从这些文件恢复反事实动作收益，也不能把条件相关写成辅助模态的因果效果。

## 执行与统计

`scripts/analyze_full_dev_coverage.py`只读取已回收的完整dev文件，不导入Torch、模型或tracker，
不进行解码、SSH、CUDA、前向、优化器更新或权重读取。当前11项活动源码保持不变。
入口首先要求`COMPLETE.json`，再检查非smoke、单片、97序列/54,646帧及三类文件的完整名字集合。
逐序列核对预测/GT形状、有限数值、正预测宽高、首帧GT初始化、连续trace帧号，
动作/模态和385主token、0/193/385辅助token必须与当前执行路径一致。

从上一帧保存的预测框重建因果搜索区域：按`crop_pair`的float32状态和
`sqrt(float(w*h))*4`计算边长，返回float32几何。保存框仅有小数后六位，
所以这是近似重建，不是当时完整浮点状态或裁剪像素的逐位重放；区域边缘判定可能受量化影响。

去掉初始化帧后，分别统计有效GT的区域内/外帧数、动作和模态频率、平均重型token、
响应置信度、连续原始IoU和中心像素误差。另报告全部track帧（包括无效GT帧）的路由频率。
有效GT仅指正宽高；与原工具包的有效性定义不同。原始IoU/误差不做工具包取整或
首帧替换，且按帧加权，**不是正式PR/NPR/SR，也不是序列等权的OPE指标**。
没有有效帧的分组均值写null，不冒充零误差。

相邻两个非初始化帧均有有效GT、前帧中心在区域外且当前帧在区域内，记为一次ROI中心重入。
跨序列与无效GT不连起来；这不是准确跟踪恢复、完整越界episode或恢复耗时统计。
每条序列保存预测、GT和trace的SHA256，汇总绑定配置与完成文件SHA及配置中的权重SHA/成本；
本工具不自行加载或验证权重文件。

完成后使用现有本地环境，示例：

```powershell
uv run --python 3.12 --with numpy==1.26.4 python scripts/analyze_full_dev_coverage.py `
  --run reports/full_pipeline/dev_complete_c_cost0p0 `
  --output reports/full_pipeline/dev_complete_c_cost0p0_coverage.json
```

仅在原控制器成功回收该run后执行。不改控制器，不新建观察器，不提前读取活动dev输出。
其他两组成本按同样完整范围分别诊断；正式指标仍由原`score_tracking.py`和未改RGBT工具包给出。
两组完整数据的真实exit与断言结果已保存于reports/full_dev_coverage_native.json；CLI-help与源码PASS不能替代该证据。
