# Query 接受质量：ATOM / PrDiMP 源码核对

核对时间：2026-10-27 UTC。官方源码固定在 `visionml/pytracking` commit [`7eb9e74bd3d40e29dbcec444902237da13de247b`](https://github.com/visionml/pytracking/tree/7eb9e74bd3d40e29dbcec444902237da13de247b)，没有下载或验证模型权重；因此本文是源码/论文机制核对，模型状态为 **UNATTESTED**。

## 结论

ATOM/PrDiMP 把两类量分开：分类分支在搜索图上给目标中心定位置信度；IoU/概率回归分支给候选框精修或状态密度。官方实现没有把 IoU 预测值当作“框是否可接受”的阈值，也没有把它直接当作记忆写入条件。分类分支的 `not_found` / `uncertain` / `hard_negative` 标志才控制定位状态和分类记忆的分支，而且两者的门控不同。

这不能为本项目的 RGB/TIR、主区域/辅助区域提供可直接移植的统一阈值。现有项目的 `confidence` 是所选输出 score map 的最大值：候选框已先写入 `state`，只有每 5 帧且 `confidence >= .65` 时才写模板。因此当前行为是“状态写入无质量拒绝，模板写入有一个未经校准的阈值”，不是一个独立的定位接受门。

## 官方实现逐点核对

| 机制 | 固定源码证据 | 实际作用 |
|---|---|---|
| 分类置信度/定位 | [ATOM `track`](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/atom/atom.py#L225-L298)、[ATOM `localize_advanced`](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/atom/atom.py#L341-L406) | `max_score1` 低于 `target_not_found_threshold` 时返回 `not_found`；近邻外第二峰还会产生 `uncertain`/`hard_negative`。这是搜索图内的分类/响应分数，不是框 IoU。 |
| 框质量/IoU | [ATOM `refine_target_box`](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/atom/atom.py#L701-L755)、[IoUNet `predict_iou`](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/ltr/models/bbreg/atom_iou_net.py#L96-L136) | `output_iou` 只用于筛掉极端长宽比后取 top-k，并平均候选框；`predicted_iou` 没有与阈值比较来拒绝框。 |
| ATOM 状态写入 | [ATOM `track` 246–254、290–292 行](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/atom/atom.py#L246-L298) | 只要 `flag != 'not_found'` 就更新位置并精修框；`uncertain` 仍可更新位置/尺寸（是否更新尺度由参数控制）。 |
| ATOM 分类记忆 | [ATOM `track` 267–288 行](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/atom/atom.py#L267-L288) | `flag not in ['not_found','uncertain']` 才写入分类记忆；`hard_negative` 仍写入，但使用更高的 hard-negative 学习率。 |
| PrDiMP 状态/记忆 | [PrDiMP/DiMP `track`](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/dimp/dimp.py#L94-L175)、[`localize_advanced`](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/dimp/dimp.py#L238-L303) | 同样由分类 `flag` 控制：`not_found` 不改状态；`uncertain` 可改状态但不写分类记忆；正常/`hard_negative` 才进入分类更新。 |
| PrDiMP 参数 | [PrDiMP-50 参数](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/parameter/dimp/prdimp50.py#L43-L64) | `score_preprocess='softmax'` 与 `target_not_found_threshold=.04` 仍是目标中心分类搜索分数的处理/门控，不是 IoU 阈值。IoU 精修仍是 top-k 框优化。 |
| 分类与 IoU 是两条分支 | [DiMP 网络 forward](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/ltr/models/tracking/dimpnet.py#L50-L69) | 网络分别调用 `classifier(...)` 和 `bb_regressor(...)`；不能把两条分支的数值当成同一量纲。 |

论文也明确区分这两类量：ATOM 描述在线分类组件与离线 IoU overlap 估计组件各自的职责（[ATOM 论文](https://arxiv.org/abs/1811.07628)）；PrDiMP 将 target-center regression 与 bounding-box regression 分开，后者以框 IoU 为伪标签并对框置信度做梯度精修，前者是在搜索区域内回归目标中心（[PrDiMP 论文，4.1 节](https://arxiv.org/abs/2003.12565)）。论文中的 conditional density 表示不等于跨输入区域、跨模态已经校准的接受概率。

## 本项目当前路径

- [`candidate_tracker.py:25-49`](../dual_rgbt_global/candidate_tracker.py#L25-L49) 同时使用主搜索 crop、全局 broad crop 和所选 query crop；query 可能是 128 或 256 像素边长。
- [`candidate_tracker.py:50-58`](../dual_rgbt_global/candidate_tracker.py#L50-L58) 先无条件执行 `self.state = execute_boxes(...)`，再取 `output['score'].max()`；模板仅在每 5 帧且 `confidence >= .65` 时写入。
- 主路径通过 [`DualRGBT.primary`](../dual_rgbt/model.py#L92-L103) 乘 Hann window；query 路径通过 [`EvidencePolicy.query`](../dual_rgbt_evidence/policy.py#L90-L92) 调用 [`LocalEvidenceReader`](../dual_rgbt_evidence/readout.py#L100-L119) 时不传 window。128 query 是 8×8 readout，256 query 是 16×16；box head score 本身是 sigmoid 后的中心热图（[`head.py:174-201`](../lib/models/layers/head.py#L174-L201)），训练损失也是 heatmap focal loss（[`losses.py:24-32`](../dual_rgbt/losses.py#L24-L32)），并非 IoU 监督。
- 因而 raw `score.max()` 同时受 crop 大小/覆盖、8×8 或 16×16 网格、window 是否相乘、主/辅助特征组合及模态分支影响。源码没有提供跨区域、跨模态的校准或等价性证据；不能把 `.65` 或 ATOM/PrDiMP 的 `.25`/`.04` 当作可迁移阈值。
- [`readout.py` 的模块说明](../dual_rgbt_evidence/readout.py#L1-L6) 明确 query geometry 可独立于主 crop，并可读到主搜索之外；所以仅用主框/历史框的重叠或距离作唯一门控，会阻断真实的全局恢复。

## 下一步控制（待 DEV 完成，不承诺收益）

三模块职责保持不变：policy 只选 query，executor/readout 只定位，tracker 只写 state/template。完成 DEV 后，固定现有权重的最小控制应先做**同帧、按路线校准的 scalar acceptance**：分别记录主路径、local 128、local 256、wide 及 RGB/TIR 路线的 score、候选框和同帧 GT IoU，在 DEV 上估计每条路线自己的阈值或分位数，并只把它作为 query 结果的接受/模板写入实验变量。阈值不能跨路线直接复用；几何一致性可以作为诊断特征，但不能作为唯一拒绝条件，以保留主 crop 外的全局恢复。该控制尚未实施，也没有收益结论。

如果需要把质量从现有输出中学习出来，则需额外训练一个质量头（同帧特征、路线/模态/几何描述，监督为 GT IoU 或 accept label），并保存新 checkpoint、做独立校准和消融；这超出“固定现有权重”的控制，当前没有实现或结果。

\n