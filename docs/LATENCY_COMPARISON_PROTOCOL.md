# 固定权重的原生系统延迟对照

> 2026-10-14：以下为此前C20前缀准备记录；当前完整延迟测量协议见FULL_DEV_LATENCY_PROTOCOL.md。
> 当前脚本新增--full-dev，正式测量等待全882训练和最终dev/test结束，不在参考流程结束时抢占新训练。

目的：在同一张25服务器GPU上，比较原去模块CADTrack与最终Dual-RGBT的实际因果
前缀服务时间。baseline精度沿用用户给定72.74/69.16/57.84，不重新评分。
主测试的全部245序列时长仍由原完整管线收集；这项前缀诊断不能替代完整速度结论。

源码：`scripts/compare_tracking_latency.py`，SHA256
`4885422a5865942474fea78e5e007a05d29d6fe535b096e58a97ef71e03711e2`。
独立上下文源码审查SOURCE PASS、0阻断、same-family/provisional，实际模型路由未证明。
实际本地Python3.7 AST通过；7个关键baseline文件与原保存源码归档逐字节相同，
11个活动管线文件hash未变。服务器原conda在CUDA隐藏时`--help`实际exit0，stderr为空。
**仅CLI导入检查完成；构造器、checkpoint载入、NN前向、配对延迟仍NOT RUN。**

## 后续执行节点

先让当前完整C、dev选择和test两片正常结束，保留唯一原控制器，不加入竞争任务。
读取其 `reports/frozen_test_selection.json` 的同一权重SHA及成本。确认GPU0闲置后，
使用 `scripts/run_task.py` 在GPU0记录命令、PID和退出码；不读温度/功率。
第一项仅1条dev、16帧，检查两侧严格加载、真实因果前向及正常退出；不反向或更新原权重。
它不是训练实验，也不产生精度。随后相同冻结输入做默认10条dev×64帧正式诊断。

命令中的 CHECKPOINT/COST 来自上述真实冻结文件，不能用待结束C、中间B或sanity代替：

```bash
CUDA_VISIBLE_DEVICES=0 /home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python \
  scripts/compare_tracking_latency.py --checkpoint CHECKPOINT --cost-weight COST \
  --sequences 1 --frames 16 --report reports/latency_comparison_smoke.json

CUDA_VISIBLE_DEVICES=0 /home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python \
  scripts/compare_tracking_latency.py --checkpoint CHECKPOINT --cost-weight COST \
  --report reports/latency_comparison_final.json
```

smoke估计小于1分钟，主项预估5–10分钟（由已测单侧约0.16秒服务时间估计，非承诺）。
实际启动后按估计节点检查，必要复查180–300秒；不因观察超时重启。
执行结束后原始两权重SHA必须不变。保留全帧时长、顺序及ours模态/动作/token。

## 测量和解释

两模型常驻同一可见卡，各自从首框初始化；之后状态、搜索与模板只依赖自己的预测。
每帧交替先执行的一侧，每侧都独立读取、解码同一帧，不复用另一侧的数组或crop。
计时包含解码、裁剪、归一化/传输、定位、策略和模板更新，以及两侧相同的外层CUDA同步。
模型加载、哈希与结果写盘不计入。full_service含每序列初始化及全部64帧；
warm_service固定保留frame9–63，默认550帧/侧，不把该截取规则冒充稳态证明。

原CADTrack有整数裁剪/零边界/GPU归一化，ours有affine/全图均值边界/CPU归一化；
模板更新和输出路径也不同。因此总差值属于原生系统对照，不能单独归因于token或某模块。
双模型常驻、文件缓存、交替顺序、共享CPU负载及前缀分布均有限制，详细列在审查报告。
参数量由实际实例分别计数；原基线本就共享双路骨干，不假定参数减半。
若结果更慢，应按真实热点处理，不能用token减少替代完整服务时间。
