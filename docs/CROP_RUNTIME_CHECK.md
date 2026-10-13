# 裁剪 API 原生 NN 准入结果

源码独立审查见CROP_RUNTIME_REVIEW.md。2026-10-13 GPU1原生wrapper415420真实exit0，
22:02:04.663–22:03:02.159 BJT，57.496秒。两child415539/417028依次真实exit0，未并发两条NN。
入口固定C输入SHA94be6239f7f6864cc193fff7c21d67c9479924c9432039c6ff16cd0d1223f15f。

复用原train_trajectory_policy --sanity，每臂实际train10帧/2次AdamW更新；257冻结state不变，
quality/gain非零梯度且值变化，保存后strict reload的路由/box/gains一致。
两臂CSV监督记录相同，11连续因果帧state/templates/history字节SHA和实际路由相同，
所有265个net Tensor值零容差torch.equal。初始化帧包含在11状态中，不写成11次track。
证据：reports/crop_runtime_native.json、crop_runtime_children.json、crop_runtime_check.json。

只有第二child在内存中先替换data.crop_pair，再导入tracker/原训练入口。
原始生产文件、当前11份活动source、正式权重与评测配置不变。sandbox权重不用于评测。
这是短train轨迹的数值准入，不是全882条一致性证明、完整精度、真实延迟或生产接入结果。
后续完整882流程将用单独run_with_crop_api入口，同样在导入前选择已验证API；
它必须等原参考dev/test全部结束才启动，先以正式B再检查真实sanity，之后从同B重新正式训练。
