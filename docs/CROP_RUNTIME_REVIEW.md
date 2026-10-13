# crop API 实际 NN 准入源码审查

2026-10-13，独立新上下文 /root/review_crop_runtime_admission；请求 gpt-6-astra/max。
同家族 provisional，实际路由 UNATTESTED，只读本地源码，无SSH/NN/活动结果读取。
最终入口SHA ae5f2388e8b6eb3a4324302014709118c818750d39e116886e5edadf86362435。

原b775版本FAIL：子wait(timeout=120)异常会留下活进程，启动PID收据尚未保存。
修正版启动后立即保存PID/command/start与空exit，等待同一child实际结束再保存exit；不超时重启。
现结论WARN/源码阻断0；补入实际crop kernel与model SHA后两行复核结论保持。
正确导入顺序：先替换data.crop_pair，后导入tracker/原C训练入口；辅助profile main均有guard。
原C sanity保持257冻结state、quality/gain非零grad与变化、2真实optimizer更新、strict reload一致门。
两sandbox要求CSV2行一致、11连续帧state/template/history SHA与路由一致、全部net tensor torch.equal。
固定仅train输入C SHA，两个独立run，GPU1串行；不改生产文件、不选sandbox权重。
源码审查不能证明当前GPU空闲或NN通过；实际执行另存reports/crop_runtime_check.json。
