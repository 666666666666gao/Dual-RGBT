# 实验执行表

| Run | 阶段 | 目的 | 数据 | 状态 |
|---|---|---|---|---|
| size_tail_sanity_b8_20261030 | S1 | 尺寸末层更新、score/offset固定、严格重载 | train32样本/4步 | SANITY_PASS，native exit0，peak allocated0.5906GiB |
| size_tail_sanity_b64_20261030 | S2 | B64/LR8e-6真实容量验收 | train256样本/4步 | SANITY_PASS，native exit0，peak allocated2.0113GiB |
| size_tail_sanity_b384_20261030 | S2 | B384/LR4.8e-5容量验收 | train1536样本/4步 | 已启动，wrapper1088817，待验收 |
| parent zero-update evaluator | S3 | 新评估入口与父权重等价 | LasHeR dev97 | 待实现 |
| size-tail formal | T1 | 至多10个采样epoch | LasHeR train | 未准入 |
| last-five full dev | V1 | 最近5个权重官方指标选优与早停 | LasHeR dev97 | 仅保存保留逻辑及测试完成；完整dev联动未实现 |