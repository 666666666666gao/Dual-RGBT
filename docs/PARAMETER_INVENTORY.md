# 当前架构的实际参数量

2026-10-13原conda纯CPU实际构造与strict加载完成；wrapper545351真实exit0，22:44:40.512–22:45:00.184 BJT。
子脚本在import torch前清空CUDA，实际所有参数位于CPU、CUDA context未初始化；无forward/optimizer/GT/精度。
wrapper的physical_gpu=1只是原任务入场槽，不表示执行CUDA。原17份活动/排队源码不改。

| 口径 | 参数元素数量 |
|---|---:|
| 用户去模块CADTrack总参数 | 103,138,437 |
| 当前Dual-RGBT总参数 | 103,374,215 |
| 增量 | +235,778（+0.228603%） |
| C阶段可训练参数 | 221,442 |

| 组件 | 当前参数元素 |
|---|---:|
| backbone | 86,045,184 |
| decode | 10,619,136 |
| box_head | 6,474,117 |
| preview | 14,336 |
| quality | 4,865 |
| gain | 216,577 |

原基线本就共享双路主干。当前保留相同backbone/decode/box_head，再增加Preview/quality/gain三小模块，
因此没有总参数下降。冻结257个state张量与仅训练quality/gain不等于减少部署参数。
组件decode+box_head合计17,093,253参数，按需修正仍复用它们；权重复用与实际调用次数须分别统计。
后续论文优先证明真正推理效率，并如实报告约0.23%的参数开销；token减少不能替代完整时间证据。

标准参数口径为Module.parameters()去重Parameter对象后的numel，排除buffers；
不按底层存储去重（原cls_pos_embed/pos_embed有存储别名，两方一致），不等于物理GPU内存。
报告绑定当前C94be及实际baseline文件SHA，加载前后baseline摘要均相同：
cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731。
未补造首次历史SHA保护凭据，不声称追溯证明最初训练字节身份。
实际证据reports/model_parameter_counts.json、parameter_inventory_native.json；独立源码审查另见对应REVIEW。
全882训练仍在队列等待，完整test和最终同设备延迟尚未完成；本清点不替代这些证据。
