# 独立区域策略真实阶段准入

2026-10-14：SOURCE审查后的同一25项代码在GPU1顺序执行A/B/C准入，无环境重建或源码改动。
计划SHA e4aae856da98df58c3f4100f048d50b06fd81c5fe559af6f0007f5401b737432。
三项run_task真实退出码均0，串行owner调用native21887也exit0。

| 阶段 | 真实数据/更新 | 冻结state | 改变state | 保存后重载 |
|---|---|---:|---:|---|
| A | 80 train对，RGB/TIR×五动作各8，10更新 | 168 | 58 | 7项输出逐位一致 |
| B | 16 train对，2次当前动作收益监督更新 | 251 | 14 | 7项输出逐位一致 |
| C | 11runthree实际闭环10帧/1初始化/2更新 | 257 | 8 | 7项输出逐位一致 |

7项probe为quality/gains/queries/primary_box/local box/pixel_box/score；它使用真实当前输入，
强制检查外部query前向，不等价于保存后重复完整长轨迹。
每次更新检查目标模块全部参数梯度存在、有限且该模块总梯度非零；Adam state.step等于真实更新数。
C两次选中反事实执行框与当时实际track state逐位一致；当前GT只监督标签，没有维持之后搜索。
全部config原始planSHA与本地封存相同。源码审查依然是same-family/provisional，actual routing UNATTESTED。
初始冻结协议的PENDING状态是历史，当前SOURCE_PASS和NN准入分开记录。

原报告在reports/full_pipeline/evidence_sanity_{a,b,c}_20261014/{config,SANITY,COMPLETE}.json。
完整原生任务终态和报告位于reports/evidence_policy_admission_status.json。
权重只用于准入链，不能初始化正式训练。

## 正式训练与实际问题来源

GPU1正式A从旧正式B ba49b506...重新严格初始化，07:18:56 BJT单次启动。
wrapper1309002/child1309003，固定5×10000对/batch8/workers4/6250更新。
startup实际核对CUDA/物理卡/25source/远端plan/config，未提前读取loss或进度。
首诊断07:28:56 BJT，用实际吞吐校正结束时间；当前A未完训，B/C仍等待。
实际首次有效观察07:32:46 BJT：RUNNING、1300/6250更新/epoch2、1300写入loss均有限，
前缀吞吐1.567646更新/秒，线性预计08:25:23完成，下一必要观察08:20:23；不能据此声称完训或精度达标。
GPU0原882 C任务/17source/原test冻结保持不变。最终PR/NPR/SR、延迟与三个模块对照仍未完成。

具体设计根据train20反事实的258个源crop外状态/2064对全零IoU问题，并非由test标注调参。
已核查开源参考：[PySOT长期搜索](https://github.com/STVIR/pysot/tree/d04028f8e18c6b27ee150a7c0d8aab9a1807219e)
用于区分原图范围与输入大小；[Uni-AdaFocus](https://github.com/LeapLabTHU/Uni-AdaFocus/tree/8846488310fdd4a18412608006030643e794c36e)
用于编码前采样、固定局部重型预算。来源原文及hash见SEARCH_COVERAGE_REFERENCES/PREENCODING_REGION_REFERENCES。
没有移植或运行这两个模型，不把它们的结果当作本项目成绩。

只读startup检测脚本的NUL字符串嵌入错误有实际失败及同代码只读重放证据；改成chr(0)后身份核验成功。
原训练未因此重启。发布模板中文路径解码错误已修正指定Desktop副本；NN源与结果未受影响。
首进度只读脚本嵌套换行字符串也编译失败，修为chr(10)并本地预编译后实际获得上述数据；
修正器初次局部计数assert错误和缺失脚本调用保留，没有恢复/重复启动NN。
