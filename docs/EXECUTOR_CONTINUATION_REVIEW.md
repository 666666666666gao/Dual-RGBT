# 执行器续训源码审查

- 日期：2026-10-13
- `review_independence: same-family`
- `acceptance_status: provisional`
- `review_context: continuing reviewer; separate executor-continuation source review`
- `requested_model: gpt-6-astra`
- `requested_reasoning_effort: max`
- `actual_model: unavailable to reviewer (runtime metadata not exposed)`
- `actual_reasoning_effort: unavailable to reviewer (runtime metadata not exposed)`
- `reviewed_base: bdf2fd0eff32ee94acf9a97187682b660ea01b6e`
- `reviewed_target: current scripts/train_executor.py, scripts/run_stage_a_full.sh, scripts/observe_stage_a.py`
- `source_verdict: PASS`
- `open_blocking_issues: 0`
- `open_nonblocking_issues: 0`
- `runtime_acceptance: PENDING (new continuation only)`

本次直接审查新增 resume 分支、完整 A 启动参数和参数化 observer，对照当前数据采样、可选 Hann 接口，以及本地已收集的短程空间 A 完成报告、配置、dev 与 best JSON。只写本报告，未修改实现或此前报告；未执行 SSH、GPU、checkpoint 二进制加载、训练或 baseline 精度评价。新的完整 A 尚未由本审查启动。

## 结论

没有发现这次续训方案的具体源码问题。代码从短程 A 的 latest 恢复模型和 AdamW 状态，独立保存续训目录并继承既有 best；指定命令对应新增 5 个 epoch、6250 次更新，完成后累计 7250 次更新。

本地 `reports/stage_a_spatial_complete.json` 记录进程不存在、1000 条更新记录、末行 epoch 2 / step 999，完成标记为 2 epochs / 1000 steps。`reports/stage_a_spatial/best.json` 为 epoch 2 / step 1000 / dev loss 0.9784963741078971；本次从其十条 epoch 2 dev 均值重新计算得到相同值，每路径均为 100 样本。执行代理报告原生任务 91336 退出 0、源 latest 含 optimizer 且 best.pth 存在；本审查没有访问服务器或读取这些权重二进制，不能将该通知写成独立 GPU 验证。

## 核对依据

1. **恢复对象和顺序正确。** `train_executor.py:40–62` 先创建全新的 run 目录，再构建 CUDA 模型、冻结 preview/quality/gain，按原顺序建立 backbone 与 decode+box_head 两个 AdamW 参数组。resume 随后严格载入 `net`，再载入同一 checkpoint 的 `optimizer`，继承其 epoch 和 step。可选 Hann 没有新增网络参数或改变参数注册顺序，A 的 `forward_action()` 也未传 window；旧空间 A 的模型键、优化器参数映射与当前 A 一致。

2. **CPU 读取后优化器设备映射正确，学习率被继承。** 已核对项目记录使用的 PyTorch 1.12.1 官方源码：`Optimizer.load_state_dict()` 按参数组顺序映射参数，将除 step 外的状态张量转到参数的 dtype/device，并保留已保存的参数组设置。因此动量会进入对应 CUDA 参数的设备，step 保持 CPU；组学习率由 checkpoint 恢复。当前旧配置和新命令均为主干 1e-5、decode/head 1e-4，没有新学习率被意外忽略的问题。[PyTorch 1.12.1 Optimizer 源码](https://raw.githubusercontent.com/pytorch/pytorch/v1.12.1/torch/optim/optimizer.py)。当前 AdamW 使用默认 `capturable=False`，CPU step 符合该版本的初始化和更新路径，无需另加手动状态迁移。[PyTorch 1.12.1 AdamW 源码](https://raw.githubusercontent.com/pytorch/pytorch/v1.12.1/torch/optim/adamw.py)。

3. **旧 best 不会被较差的新 epoch 替换。** `:58–61` 从 resume 的同目录读取 best.json，将其 dev_loss 设为当前 best_loss，并复制原 best.pth 和其元数据到新 run。后续 `:162–169` 仅在同标准 dev_loss 严格下降时替换新目录的 best。即便新增五轮均未超过旧结果，旧 epoch 2 权重仍是新目录 best；原 run 的 latest、best 和日志均未被写入。latest 继续保存当前 net、optimizer、global epoch/step 和配置，best 保留其真实选中 epoch/step。

4. **采样固定，新增预算与十路径均衡正确。** launcher 在项目目录运行解释器，限定物理 GPU 0，参数为 `--epochs 5 --samples 10000 --batch-size 8 --workers 4`，resume 指向已完成空间 A 的 latest。由旧 epoch 2 / step 1000 得到 epoch 范围 3–7；每轮 10000/8=1250 次更新，新增总计 6250。`(step % 2, (step // 2) % 5)` 从全局 step 1000 继续，每十步完整覆盖两模态×五动作，所以新增每路径 625 次，连同短程 A 每路径累计 725 次。

5. **确定性采样语义清楚。** 默认 seed 为 2027；`data.py:51–63` 依赖 `RandomState(epoch_seed+index)`，与 worker 调度无关。新五轮 epoch seed 为 22027、32027、42027、52027、62027，每轮连续 10000 个样本 seed，彼此无重复，也不重复旧两轮的 2027–6026。样本量改为 10000 后种子区间有跳跃，这是当前显式公式的结果，未宣称为某个未中断七轮任务的逐位复现。模型/优化器状态得到继承，随机数状态则按配置重新设为 seed 2027。

6. **dev 标准可与旧 best 比较。** `:104–137` 继续使用 seed 2027 的相同 100 个 dev 裁剪、两模态×五动作和同一定位损失，model.eval() 固定 BN 与随机层。dev 没有 drop_last，batch 8 的最后四个样本仍计入；先收集逐样本 loss 再求均值，不会把尾批当成等权整批。best 标准仍是十路径平均定位损失，不依赖 test，也不改变原 signed 收益和覆盖记录定义。

7. **完成记录和 observer 的计数含义正确。** `:171–174` 的 epochs 是本次新增轮数，first_epoch/last_epoch 明确给出全局范围；steps 是累计更新数，previous_steps 与 updates_current_run 分别给出继承和新增量。新 steps.csv 只写此次更新，step 字段保持全局零起始序号，因此结束时应有 6250 行、末行 step 7249、COMPLETE.steps=7250，三者没有差一错误。参数化 observer 的 `optimizer_updates_logged=len(rows)` 是当前目录已写出的行数，不冒充累计值；累计计数从末行及 completion 读取。glob 读取所有实际 dev/utility epoch 文件，能覆盖 3–7，不再只读取 1、2。

完成时预期记录为：

```json
{
  "epochs": 5,
  "first_epoch": 3,
  "last_epoch": 7,
  "steps": 7250,
  "previous_steps": 1000,
  "updates_current_run": 6250
}
```

以上是由源码推导的预期值，不是新续训的实际结果。`git diff --check` 对本次修改无输出、退出 0；新 launcher 为 LF、无 BOM、末尾有换行。

## 验收范围

本次 PASS 覆盖指定已完成 checkpoint 的显式续训、best 保护和统计口径，不代表实际恢复后的 GPU 更新或完整 A 已成功。新恢复路径仍须由其真实执行结果确认。B/C、正式测试精度与完整在线延迟不在本审查范围内，也未因短程 dev 改善而视为完成；baseline 精度不重测。
