# GLOBAL_OPPORTUNITY_SOURCE_REVIEW — R2

结论：**SOURCE_PASS，blocking_issues=0**。本次只完成源码复审，不宣称已运行GPU sanity或正式诊断。

请求配置为 Astra/max；实际路由 **UNATTESTED**，同 family/provisional。R1报告、初始seal与真实失败输出完整保留。

## 修正与封存

相对R1，唯一代码改动是`diagnose_global_opportunity.py:222-223`新增：

```python
for parameter in clone.parameters():
    parameter.requires_grad_(False)
```

它位于strict load之后、clone probe之前，准确消除GO-SRC-001。原`.eval()`和`@torch.no_grad()`仍在；没有为准入执行梯度或权重更新。AST检查并未构建模型。

- driver SHA：`c443f34c91149dee16cefa89db42e2eea1921cc21c0f656a40d22a2c70ff6272`。
- plan SHA：`a0accbdea3c6c8742103484e7a6c1db8bc5507647f1975f5f048477c79ef84e7`。
- deploy helper SHA：`df39ba61b0f7578ed9038193a9b2c68d12cae7b530ca6bdb5a39ba229771c5d5`。
- launch helper SHA：`b896344d92d00a9e491d9520e235469501497312ce997eac2756c4fd26f739f4`。
- plan唯一内容变更是新driver哈希。旧source30、两个helpers、协议、训练计划及5个辅助源码/配置均与R1一致。
- `initial_inputs_seal.json`与`final_inputs_seal.json`记录首末哈希；本次全部42个输入始末相符。机器报告的source_sha25631、plan_sha256、helper_sha256直接使用项目/本地helper末次读取的准确字节哈希。

## 实际检查

- source31与两个helpers全部AST parse通过。
- 最小diff检查确认只增加上述两行，plan只更新对应SHA；clone freeze在strict load后、probe前。
- 缓存NumPy1.26.4加AST检查 **29 PASS / 0 FAIL，真实exit0**；完整命令、输出、`source_audit_checks.exit_code.txt`和`source_audit_checks.json`已保存。
- 4段helper生成的远端程序格式化/compile通过；没有执行远端程序。
- dtype、control几何/输入128或256、14结果/state、浮点差值语义、同状态STOP对照断言、旧head到新head接口、factory输入以及helpers准入均再次通过源码检查。未变部分的完整逐项分析保留于上级R1人类报告。
- 未导入项目/Torch、未构建NN，未读图片、manifest或GT，未使用GPU/SSH，未安装环境，未改项目/helper，也未spawn。

## 尚需真实运行核验

| 原生文件 | 必须核验的字段 |
|---|---|
| 诊断任务status.json | state=COMPLETE，exit_code=0 |
| sanity COMPLETE.json | tracked_frames=10，labelled_states=2，comparisons=28，sanity=true |
| SANITY.json | status=PASS，frozen_state_tensors=265，optimizer_updates=0，backward_calls=0，strict_reload_probe_bit_equal=true |
| SANITY.json probe_fields | 恰为quality、gains、queries、primary_box、box、pixel_box、score七项 |
| 正式 COMPLETE.json | tracked_frames=8854，labelled_states=1752，comparisons=24528，sanity=false |
| 正式A/B输入原生记录 | COMPLETE/exit0、6250updates/50000pairs/sanity=false，以及实际checkpoint SHA |

所有label状态的实际所选STOP必须`np.array_equal`；全部265个主模型state诊断前后必须bitEqual，参数grad=None。源码通过不能代替这些实际断言成功及原生exit0。

三项非阻断证据限制保持明确：GO-EVID-001，当前sanity/helper不会自动拒绝所有非10/2/28或非265/7的结果，控制者需按上表核验；GO-EVID-002，factory input只入口校验，须补存其与plan末次SHA才可声称全部输入始末不变；GO-EVID-003，helper没有读取helper_sha256自行校验，执行时须核对本报告封存的准确helper哈希。

完整action_comparisons.csv继续 **PRIVATE_ONLY**，不得上传Git；GT-centre几何/输出只用于私有诊断，公开聚合报告。四控制仅用于实际选中主模态，非可部署动作或完整动作/轨迹上界；A阶段global preview未训练，当前候选差不能否定可学习性。本检查不替代完整882因果训练、dev97校准或最终test245指标。

R2完成SOURCE阶段；下一步真实只读sanity仍必须保持0 optimizer / 0 backward。原有A10/B2/C2准入不要求在本诊断重复权重更新。
