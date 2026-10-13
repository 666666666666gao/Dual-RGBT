# 全882条闭环训练及完整评价接续

2026-10-13在原20条C参考流程的test结果尚未产出时冻结方案，避免用test反馈选择训练设置。
原流程仍唯一控制其dev/test任务；新控制器必须等其COMPLETE_PIPELINE后才能启动GPU任务。
原11份源码不修改。新结果、状态、run和freeze文件分别命名，参考test freeze另存后才更新远端canonical freeze。

## 固定训练与数值准入

已有A/B随机样本训练均涉及882条，但不宣称遍历全部帧。A累计7250/B6250更新已完成。
新C从相同正式B重新开始（不是旧C或sanity权重），SHA `ba49b506cd4bdcdfb355e8741feaf6c5c61e2edf9cd0a5c91459c3e5957c80ed`。
seed2027固定882条排列、两轮、label_stride5、cost0.05、AdamW1e-4，末轮权重固定，不用test选轮次。
计划预期917,106次track、1,764初始化、182,182有效监督更新；train manifest SHA `389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc`。

新入口run_with_crop_api先设已验证crop alias后导入原训练/跟踪脚本，模型及原脚本不变。
CPU80真实case的输入/目标/几何及60搜索完整RNG一致；GPU两臂10train帧/2update、11因果状态及265net值一致，
原native真实exit0，见CROP_RUNTIME_CHECK.md。它是准入，不是全轨迹精度或延迟证明。
新接续仍先从正式B做新入口真实sanity，再从同一B重启正式882训练；从不选sanity权重。

## 观察、审计与评价

预计训练36–45小时；按原完成20序列实际约47.6分钟外推，序列长度/负载/新API会影响时间。
训练每4小时查看native状态及已flush CSV最后100行loss/路由/step，非每240秒重复查询两天。
各NN任务由原run_task脱离SSH并保存真实退出码，失败停止推进，不自动重试。
新控制器遇旧流程已有FAILED状态同样停止，不无限等待这条已知失败分支。
新排队首次检查旧流程不早于 `2026-10-14T04:28:50.010174+08:00`，以后必要时240秒。

末轮真实exit0后用纯CPU audit_all_trajectories逐项核对882排列、全部182,182监督三元组、连续step、
有限loss、257冻结state与原B相等、8个optimizer step、正式保存config与初始/末轮SHA。
不把监督帧路由统计扩展为全部917,106帧，不把CPU严格state读取写成NN重载。
审计PASS才跑全97条dev的0/0.05/0.15成本，按SR→PR→NPR→更低平均ms选择；固定最终轮次。
先将最终权重SHA/成本冻结，再跑245条test/220,703帧，两授权卡各一个NN任务。
沿用现有本地Python3.12/原RGBT工具包，包含decode/init、排除modelload的全帧时间，不重测baseline精度。
新的正式结果将写reports/full_pipeline/lasher_all882_full_metrics.json；当前尚未启动或完成全882训练。

研究目标仍需test PR≥78/NPR≥72/SR≥62、同设备完整延迟优势及三个核心模块独立证据。
当前dev达到阈值不代表研究目标已完成；全882训练也不自动解决已发现的搜索视野问题。

## 队列实际启动凭据

SOURCE_ONLY独立审查PASS/0阻断，见ALL_TRAIN_REVIEW.md；新entry原conda CUDA隐藏CLI实际两次exit0。
22:22:53.625 BJT owner16936已单次启动，uv35984/shim41324，状态WAIT_REFERENCE_PIPELINE、tasks=[]。
下一预计检查旧终态节点10-14 04:28:50.010 BJT；未启动882 NN、不抢原评测GPU。
初次2秒启动检查早于uv启动完成，shell inspection exit1；随后同一进程链确认为存活，无重启。
启动身份及source/plan SHA保存在reports/all_train_launch.json，不把排队写成训练完成。
