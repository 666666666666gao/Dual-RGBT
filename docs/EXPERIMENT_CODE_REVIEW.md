# M0 / Stage A 源码审查

- 日期：2026-10-13
- `review_independence: same-family`
- `acceptance_status: provisional`
- `review_context: fresh source review; fixes rechecked by the same reviewer`
- `requested_model: gpt-6-astra`
- `requested_reasoning_effort: max`
- `actual_model: unavailable to reviewer (runtime metadata not exposed)`
- `actual_reasoning_effort: unavailable to reviewer (runtime metadata not exposed)`
- `initial_verdict: FAIL (one blocking training-label defect)`
- `final_source_verdict: WARN (blocking defect corrected; one nonblocking diagnostic gap)`
- `open_blocking_issues: 0`
- `runtime_acceptance: PENDING`

本次直接读取 `dual_rgbt/{model,data,losses}.py`、`scripts/{prepare_data,sanity,train_executor}.py`、`scripts/run_stage_a.sh`、`docs/EXPERIMENT_PLAN.md` 和仓库 `AGENTS.md`，并核对实际依赖的 CADTrack 主干、patch embedding、检测头、热图与原训练裁剪实现。只写本报告；未执行 SSH、GPU 前向、训练或 baseline 精度评价。后新增的 `scripts/train_policy.py` 不在本次验收范围内。

## BLOCKING：首轮发现，已修复并复核

**B1：搜索中心扰动过大，常规训练会产生目标缺失与不一致监督。**

- 位置：`dual_rgbt/data.py:27`；相关消费处 `dual_rgbt/losses.py:10,27`。
- 首轮源码使用 `rng.uniform(-.75, .75, 2) * side`。归一化目标中心因此为 `0.5 - u`，范围 `[-0.25, 1.25]`；两维至少一维落在裁剪外的概率为 `1 - (1 / 1.5)^2 = 5/9`。这是正常采样分布，不是罕见边界情况。热图的目标峰落出 16×16 网格；目标完全出框时，框损失仍把裁剪后的零面积边界框作为定位目标。
- 最小修复已由执行代理完成：中心扰动改为 `[-.375, .375] * side`，保留原随机尺度。当前源码已重新读取确认。目标中心现在位于 `[.125, .875]`，对应热图坐标 `[2, 14]`，且目标与裁剪必有交集。该扰动范围与原 CADTrack 的 `CENTER_JITTER=3`、`FACTOR=4` 相符（`lib/train/data/processing.py:81–85`）。
- 状态：源码层面已解决；修复后的真实数据 GPU 验证仍待 M0。

## NONBLOCKING

**N1：目标区域覆盖仍未记录。**

`docs/EXPERIMENT_PLAN.md:42–46` 要求在局部收益不足时区分区域未覆盖目标与修正器失效。当前 `scripts/train_executor.py:103–115` 保存逐样本损失、IoU、序列与帧，但没有保存辅助查询区域对 GT 的覆盖。最小补充是在现有验证前向中，利用 `output['region_indices']` 与当前 GT 保存一个定义明确的覆盖量（例如区域与 GT 的相交面积 / GT 面积），无需额外神经网络前向。此项不阻断 M0 和短程 A；分析局部收益不足时需要补齐。

首轮另一个记录缺口已修复：原来只存每动作均值，无法恢复收益分布。当前 `scripts/train_executor.py:105–136` 已保存 `dev_samples_epochNN.csv` 和 `utility_epochNN.json`。复核结果：

- 每次动作遍历使用同一个固定 seed 的 100 个 dev 样本；数据按 index 独立生成且 DataLoader 不 shuffle，`sample_id` 顺序一致。
- signed 收益为同主模态 STOP 定位损失减去追加动作定位损失，没有截断负值。
- 分位数、负收益比例以及以 IoU=0.5 为界的纠正/损害计数方向正确。
- 统计复用已有前向结果，没有增加模型推理或 baseline 重测。它们是 GT 裁剪下的样本诊断，不是在线 PR/NPR/SR。

## 确认正确的实现

| 检查项 | 源码证据与结论 |
|---|---|
| 权重加载 | `model.py:38–40` 以 `training=False` 构建原 CADTrack，并严格加载原 checkpoint 的 `['net']`；不会读取 YAML 中旧的 OSTrack 预训练路径。`reports/checkpoint_structure.json` 记录实际 checkpoint 为 epoch 14、PyTorch 1.12.1+cu113，主干/位置编码形状符合当前代码。新封装的严格 GPU 加载尚待 M0。 |
| 先选 patch 再重编码 | `model.py:122–136` 先生成原搜索网格索引；`:79–84` 对原始像素 unfold 后 gather，再执行共享投影及所选原始位置编码。没有先计算完整辅助重型特征图。底层 patch norm 是 Identity，线性展开与原 16×16、stride 16 卷积投影一致。 |
| 主辅调用次序 | `model.py:168–176,182–192` 为预览→一次完整主路径→候选→可选辅助路径；修正器没有再次调用主路径。STOP 跳过辅助编码。辅助全区域动作采用同一执行器的全部 256 个搜索 patch。 |
| 形状与 token 数 | 两个 128×128 模板产生 128 个模板 token；主路径为 `1+128+256=385`，局部辅助为 `1+128+64=193`，扩大辅助为 385。主特征 `[B,256,768]`，补零后的双槽解码输入 `[B,1536,16,16]`。质量头、收益头和修正器输入分别为 74、1690、1544 维，与实际拼接一致。 |
| 共享主干的数学 | `model.py:90–93` 使用原 `Block` 的 norm、attention、MLP、drop_path，保留单流计算。原主干本身共享两路参数；代码与计划没有把共享权重冒称为相对 baseline 参数减半。 |
| GT 与损失 | `losses.py:8–33` 将预测 cxcywh 与真实 xywh 转为 xyxy，计算逐样本 GIoU、L1 和 GT 热图 focal loss；修正质量监督来自真实 GT IoU。没有用另一模型输出充当评价真值。 |
| 时间因果 | `data.py:56–60` 两个模板均严格早于搜索帧。Stage A 的搜索裁剪基于当前训练 GT，这是离线执行器训练；它没有被宣称为在线跟踪评测。 |
| 数据分离与排除 | 已直接核对源码名单：882 train、97 val，无重复或交集。`prepare_data.py:11–30` 只从官方训练目录读取训练/dev 标注；test 只读取序列名用于分离检查。帧数/非有限标注问题记录后导致脚本失败，不是静默成功；无效尺寸帧过滤沿用原 LasHeR loader。已读取本地实际审计产物 `reports/data_audit.json`：882 train、97 dev、245 test、`issues=[]`；本审查没有重新读取服务器数据。 |
| Stage A 参数与优化 | `train_executor.py:42–49,61–80` 冻结 preview/quality/gain，AdamW 覆盖共享主干、解码、检测头和修正器，主干学习率为其他执行器参数的 0.1，梯度裁剪 0.1，BN 使用原 FIX_BN 约定。动作周期覆盖两种主模态×五种动作。 |
| 未训练模块的表述 | 预览和两种策略头在 A 阶段未训练；显式动作训练与 `predict()` 分离。代码注释明确说明该阶段是执行器训练，不能据随机策略头声称有效的在线调度或收益策略。B/C 的监督与真实闭环刷新仍须另行实现和审查。 |
| M0 断言 | `sanity.py:33–58` 对两模态×五动作检查 12 层 token 调用次序、有限 loss/梯度、主干及修正器非零梯度和实际 AdamW 主干参数更新；`:61–78` 检查严格重载后预测逐位一致，并用强成本执行真实 policy STOP，确认仅有一次主路径。脚本明确使用 eval 进行确定性执行检查，PASS 不能替代 train 模式或精度验收。 |
| 启动脚本 | `run_stage_a.sh:2–8` 使用项目内解释器、物理 GPU 0、seed 默认 2027，2 epochs×2000 samples / batch 4=1000 次优化更新，每种模态/动作组合 100 次。脚本为 LF、无 BOM；其用途是 M0 成功后的短程诊断，不是完整训练预算。 |

## 运行验收边界

当前允许按既定顺序进入修复后的真实数据 M0；只有 M0 实际成功后才能启动短程 A。数据审计产物不证明 GPU 加载、反向传播、优化器更新或 checkpoint 重载已经成功；本次没有观察到 `reports/sanity.json` 的 PASS。

可使用的既定命令为：

```bash
cd /home/gaob/Dual-RGBT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  conda-envs/dual_rgbt/bin/python scripts/sanity.py
# 仅在上一条实际成功且 GPU 0 无其他训练任务后执行：
bash scripts/run_stage_a.sh
```

上述命令在本次审查中均未执行。Stage A 输出、M0 elapsed time 和 token 数均不代表完整在线系统延迟。最终 LasHeR PR≥78、NPR≥72、SR≥62 与真实平均/P95 延迟改善仍未达到或验证；用户提供的 baseline 72.74/69.16/57.84 不重测。正式验收仍需冻结策略/预算后的完整在线跟踪及包含预览、模板、裁剪、选择、修正和扩大的实际计时。

## 追加复核：N1 区域覆盖记录已关闭（2026-10-13）

本次仅复核 `scripts/train_executor.py:106–120,142` 的新增诊断字段，保留上方首轮发现和当时的 WARN 结论。

- 当前局部查询索引按行排列为完整 8×8 区域，扩大查询覆盖完整 16×16 网格。首索引给出左上角，末索引的横纵坐标各加 1 后给出右下边界；除以 16 后与归一化 GT xywh 同坐标系，边界计算正确。
- `min(query_rb, target_xy + target_wh) - max(query_lt, target_xy)` 经零截断、两维乘积得到相交面积，再除以 GT 宽高乘积，正确实现 `gt_coverage`。GT 面积已由现有 valid_frames 的正宽高筛选及正数裁剪缩放保证为正，无需另加防御分支。
- 分母是完整 GT 面积，因此 GT 超出搜索裁剪时，扩大查询的覆盖也可以小于 1；该字段准确描述实际查询读取到的目标比例。
- STOP 保留 `None`，写入 CSV 为空字段；收益汇总仅遍历动作 1–4，对相应非空覆盖值求 `mean_gt_coverage`，没有把 STOP 当作零覆盖混入统计。
- 新增字段复用已有 `region_indices` 和当前 GT，不增加模型前向。逐样本、动作配对和此前收益计算保持正确。

最新源码结论：`followup_source_verdict: PASS`；`open_blocking_issues: 0`；`open_nonblocking_issues: 0`。该结论关闭 N1，并取代首轮修复后的源码 WARN。`review_independence: same-family` 与 `acceptance_status: provisional` 继续适用；实际运行验收仍待完成。本次未访问 SSH/GPU，未查看或干预正在运行的 M0，会话启动信息仅来自执行代理通知。
