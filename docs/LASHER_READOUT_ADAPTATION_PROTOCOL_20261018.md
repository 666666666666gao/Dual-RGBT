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


## 结果

训练完成后，适应后的 executor 注入固定 E1 policy，在 LasHeR dev 得到 PR 60.563454 / NPR 56.093476 / SR 41.664557，较原 E1 dev 76.557717 / 71.365998 / 56.433144 分别下降 15.994263 / 15.272522 / 14.768586 个百分点。该配置拒绝，不进入 test；这说明单独适应 readout 会破坏 policy 与模板状态匹配，后续若继续只能研究联合闭环适应，并要先做小规模可回滚验证。
