# 已完成 dev 的裁剪目标与模板更新诊断审查

2026-10-14，fresh reviewer `/root/review_clipped_target_analysis` 实际读取新脚本、coverage helper、
data/losses/tracker、run_tracking、C 的 target 构造、box_ops 与已有完整 dev 报告。
SOURCE_ONLY：源码正确性 PASS，整体 WARN，真正 blocker 0。
requested gpt-6-astra/max；actual model/effort routing UNATTESTED；same-family/provisional。

被审源码 SHA `b2a880e7332c3827097d564e8c8fcbff75783e1756c24b8f670ca51457e1faa7`。
reviewer独立核对输入 config SHA `977155271c89e49c02f84782cd9f42a89f05ce451c91abbd74bf88e3c32cb729`
和 COMPLETE SHA `c7f9f9d7bcee6517eaa596d4d3ad31903cb66c92fc47f1354e1f2f393c2a063d`。

prediction[t−1]、factor4、float32 geometry与crop_pair一致；归一化xywh后转xyxy并clamp，
符合当前localization_loss。有效GT中的unchanged/partial/collapsed为互斥完整分类。
首框、trace 1..N−1、target去首帧与更新帧号对齐；CSV confidence未使用框的六位小数格式化，
frame%5==0 && confidence>=.65与原Tracker一致，没有off-by-one或阈值舍入错配。
报告不把dev状态称为实际训练标签，不推断收益符号翻转或更新造成后续失败。

WARN：geometry是六位小数框的近似重建，边界临界分类仍近似，不添加容差或兼容分支。
template_update.frames只统计有效GT帧上的事件，既非全部事件，也非模态模板写入次数。
STOP写主模态，有辅助动作时写两个模态。SOURCE审查没有执行CPU诊断，运行结果由执行者凭据另行证明。
执行者实际原生CPU exit0与全97序列统计见 `reports/clipped_targets_native.json` 和 CLIPPED_TARGET_DIAGNOSTICS.md。
