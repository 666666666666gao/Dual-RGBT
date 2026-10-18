# LasHeR 读出执行器适应对照（2026-10-18）

路由诊断表明固定 RGB 与强制 primary 都低于 E1 原策略，下一步检查候选执行器/局部读出是否限制上限。该实验保持网络结构和三个模块职责不变，只训练 `decode` 与 `box_head`；backbone、preview、quality、gain 全部冻结。

## 配置

- 数据：LasHeR train manifest
- 每 epoch：10000 samples；2 epochs；seed 2027
- batch size 8；learning rate 1e-5；GPU0
- 初始化：`runs/stage_b_refreshed_20261013/latest.pth`
- smoke：`runs/readout_adapt_smoke_r2_20261018` 已通过，有限训练与 dev-crop 评估无 NaN
- formal：`runs/readout_adapt_lasher_20261018`

## 决策

完成后把适应后的 executor state 注入固定 E1 policy，只在 LasHeR dev97 上比较完整 PR/NPR/SR。若没有改善，停止 executor 线；若改善，再建立重新训练 quality/gain 的匹配预算实验，不能直接把 executor 适应结果当成三模块贡献。
