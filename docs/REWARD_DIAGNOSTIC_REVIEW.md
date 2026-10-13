# 实际动作收益诊断与队列源码审查

2026-10-14 fresh reviewer `/root/review_reward_diagnostic_queue`，请求 gpt-6-astra/max，
实际 model/effort routing UNATTESTED；review_independence=same-family，acceptance_status=provisional。
审查只读源码、固定 manifest 的既有采集副本及已完成 C CSV；0 服务器连接、0 NN、0 文件修改。

## R1：FAIL_SOURCE_ONLY，1 blocker

诊断原 SHA `d78c775f806d894acaa8b537c7dc63b7d2bbb226771a3680c37b05ad910c7c72`，
队列 SHA `6aea54f40047a54f34e15dbad54a476eed207f4ba000dc5e54c3fdd517524cf7`，
原计划 SHA `0a29dad43a73fad6942ac9384bd52326f69dea840c8bd6d8e6e5ee6fe414e38d`。
第78行访问不存在的 manifest['test']。匹配389ba37d...的真实 schema 是
annotation/train/dev/issues/seed/test_sequences；原代码会在创建模型之前 KeyError。
最小修正用 dev 名称集合与 set(manifest['test_sequences']) 并集，无 fallback。

其余 SOURCE 项通过：uncensored_loss 仅去掉原 GT clamp；执行框计算与 Tracker 完全相同，
同 margin=10、float32。track 前保存模板/历史/框，候选对应 predict 索引；每状态8对收益。
固定B、eval/no_grad/requires_grad(False)，无 optimizer/backward/torch.save；末尾检查全部张量及输入SHA。
sanity 第二次加载同B验证框和gains，正式另起进程加载原B。
独立重算20条8854track/1752label/14016pairs，9个无效GT过滤；与dev/test无序列名称交集。
旧C每轮CSV各1752行，无重复；诊断不声称重放动态更新的旧C。
队列自己的 status 与原控制器分离，只等原COMPLETE_PIPELINE；GPU1准入、等待节点和失败停止正确。

## R2：PASS_SOURCE_ONLY，0 blockers，整体 WARN

修正后诊断 SHA `77cdb4062d80b1a11fdb94e18a8a4192c704404d88859103ae9f0db2b020c0ea`，
队列仍为6aea54f4...，计划 SHA `f0edc89d98077c49a031dec0f86c4091e5a6255187ff6de0f881a9cb8d85374b`。
reviewer 反向替换唯一修改行后精确恢复 R1 源码 SHA；计划只更新诊断 SHA 与换行格式。
19/19 源码 SHA 匹配，原17项不变，两脚本 AST 通过；唯一 blocker 已解决。

WARN：代码内没有自行核对远端计划文件 SHA，部署时必须证明文件一致。
执行者部署回执 reward_diagnostic_cli_native.json 已记录计划字节一致、19项和原17项一致，
远端 --help PID791546 exit0、CUDA隐藏、stderr空；本地队列 --help PID33876 exit0。
这只是实际入口导入/CLI验证，没有 NN、sanity、正式计数或收益结果。
raw sign flip 未滤近零，不能直接解释为有效动作改变。没有宣称跨family验收或完整性能验证。
