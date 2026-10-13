# 最终权重的完整 dev 配对系统延迟

2026-10-14：原延迟脚本只支持10序列×64帧前缀，这不足以证明完整分布上的效率目标。
最小扩展 `--full-dev` 使用 manifest 的全部97序列、每序列全部帧，共54,646帧/侧。
当前脚本 SHA `b889fc86fe315357369472c6bb0bb18d3159d40ce9928871ddc44de505e1b43b`。
旧 prefix 默认仍为10×64，先用1×16检查真实加载/接口；源码独立审查及新版本实际CLI结果另行记录。
此文件是执行协议，不是已得到延迟结果。

使用全882训练结束后在dev选择并冻结的同一最终权重及成本，不以C20参考或sanity替代。
等全882训练、全部dev及test NN均真实结束后，确认物理GPU0/1没有本项目的竞争NN；
仅GPU0执行一个配对进程，两模型常驻、交替前向，每侧独立解码并维护自己的因果状态。
保留原baseline权重及加载路径，不重新评价baseline PR/NPR/SR。

```bash
/home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python \
  scripts/run_with_crop_api.py compare_tracking_latency.py \
  --checkpoint FINAL_CHECKPOINT --cost-weight FINAL_COST \
  --sequences 1 --frames 16 --report reports/latency_all882_smoke.json

/home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python \
  scripts/run_with_crop_api.py compare_tracking_latency.py \
  --checkpoint FINAL_CHECKPOINT --cost-weight FINAL_COST \
  --full-dev --report reports/latency_all882_full_dev.json
```

由原run_task包装器固定CUDA_VISIBLE_DEVICES=0并记录原生进程/退出码；命令中的最终输入
必须由全882冻结文件验证SHA与成本，不能使用中间或测试集选出的输入。
采用同一已准入crop API入口，与全882正式评价路径一致；未准入sum候选不进入本协议。
该延迟脚本未纳入全882队列17文件seal，扩展没有修改任何已封存训练/跟踪源文件。

首个1×16真实检查估计小于1分钟；完整配对预计2–4小时，先用短检查的实际服务时间更新预计节点。
预计节点前不读取中间日志；之后必要时180–300秒间隔检查同一原生任务，观察超时不重启。
使用实际全部帧累计服务时间作平均、FPS和P95，包含初始化、解码、裁剪、预览/策略、网络、模板更新与CUDA同步。
warm统计另行排除每序列frame0–8，固定预计53,773帧/侧；它不替代全部54,646帧的主统计。
模型加载、哈希与报告写盘不计入；原两权重前后SHA一致，更新次数0。

原baseline与本模型的裁剪、模板更新及定位路径不同，因此比较的是完整原生系统，
不能把全部差值归因于token或一个模块。两模型同进程常驻及文件缓存、交替顺序仍有测量影响。
完整dev测量也不冒充baseline完整test速度；正式test中本模型的全部服务时间另有完整管线记录。
只有实际完成、原生exit0并检查97序列/54,646帧及计时完整性后才报告真实效率结论。
