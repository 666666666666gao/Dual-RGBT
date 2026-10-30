# 实验执行表

| Run | 阶段 | 目的 | 数据 | 状态 |
|---|---|---|---|---|
| size_tail_sanity_b8_20261030 | S1 | 两个尺寸末层张量的真实更新、固定score/offset、重载 | LasHeR train，32样本4步 | 已启动，待验收 |
| size-tail batch sizing | S2 | batch/显存/学习率配比 | LasHeR train | 待S1 |
| parent zero-update evaluator | S3 | 新评估入口与父权重等价 | LasHeR dev97 | 待实现 |
| size-tail formal | T1 | 至多10个采样epoch | LasHeR train | 未准入 |
| last-five full dev | V1 | 最近5个权重官方指标选优 | LasHeR dev97 | 未实现 |