# 重复模板计算：固定源码核查与适用边界

2026-10-13：当前局部辅助编码共 193 token，其中两模板各 64、CLS 1、搜索 64。
非搜索 token 共 129（66.84%）；这仅说明组成，尚未按算子测出重复模板计算占整机延迟多少。
问题与当前成本 0/0.05 的 token 减少而整机均值几乎不变相联系，不能据此断言唯一瓶颈。

查阅 Cui 等人的 MixFormer 跟踪工作（CVPR 2022），固定作者仓库
https://github.com/MCG-NJU/MixFormer/tree/8b87088f45af02c614a277176c63181261e93de3 。
实际克隆的 commit 日期为 2024-02-28T17:50:34+08:00；六份源文件与 LICENSE 的字节 SHA
记录在 `reports/mixformer_source_provenance.json`。源码仅存私有参考目录，未导入或运行其模型。
论文页面：https://openaccess.thecvf.com/content/CVPR2022/html/Cui_MixFormer_End-to-End_Tracking_With_Iterative_Mixed_Attention_CVPR_2022_paper.html 。
本次浏览论文页面返回 403；不宣称已经全文核验 PDF。

## 实际源码如何减少模板重复执行

ViT `lib/models/mixformer_vit/mixformer_online.py`：
`Attention.forward` 第 60–76 行把模板查询限制为模板键值，搜索查询可读取模板及搜索键值。
`set_online` 第 96–110 行缓存模板 qkv；`forward_test` 第 80–94 行只计算搜索 qkv 并拼接缓存。
`Block.forward_test` 第 132–135 行只执行搜索侧残差/MLP，模板缓存由 `set_online` 重建。
CVT 实现的 `forward_conv_test` 同样追加缓存的模板 K/V。
ViT tracker 的第 68–70、95–99 行仅在 online_size>1 时采用 set_online/forward_test；
online_size==1 使用完整 forward。接受新模板后的更新间隔才重建缓存，不能声称所有配置都启用缓存。

## 对本项目意味着什么

当前 `dual_rgbt/model.py:encode` 拼接 CLS、双模板和搜索，调用完整双向自注意力；
`lib/models/layers/attn.py` 没有限制模板读取搜索。因此模板逐层表示依赖当前搜索，
直接复用上帧重型模板 K/V 会改变数学行为，不能作为当前权重的等价缓存优化。
若后续采用不对称模板图，需要明确 CLS 的归属及更新时机，重新训练执行器 A、刷新 B 收益标签并重训 C。
不将 MixFormer 已有缓存机制本身当作本项目的新贡献。
当前参考测试及已排队全 882 配置均未改动。只缓存冻结预览网络对初始模板的廉价描述可能保持等价，
但尚未实现、未测整机收益，也不能据此承诺目标已实现。
