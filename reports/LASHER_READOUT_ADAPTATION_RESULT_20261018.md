# LasHeR 读出适应结果（2026-10-18）

本实验保持架构和模块职责不变，只训练 `decode` 与 `box_head`，冻结 backbone、preview、quality、gain。训练完成 2 epochs / 20000 samples，无 NaN；随后把适应后的 executor 注入固定 E1 policy，仅在 LasHeR dev 评估。

| 版本 | PR | NPR | SR | mean ms |
|---|---:|---:|---:|---:|
| E1 原策略 dev | 76.557717 | 71.365998 | 56.433144 | 114.615 |
| 读出适应后 dev | 60.563454 | 56.093476 | 41.664557 | 95.903 |

相对 E1 dev 为 **-15.994263 / -15.272522 / -14.768586** 个百分点。虽然 crop-localization 控制摘要中的部分 IoU 上升，但完整闭环跟踪显著下降，说明独立 readout 适应破坏了与 policy/template 状态的匹配。

结论：**拒绝该配置，不跑 test，不把它写入最终方法。**该失败结果保留为“执行器单独适应不能直接替代联合闭环训练”的证据。
