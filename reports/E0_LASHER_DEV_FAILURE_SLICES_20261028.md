# E0 LasHeR dev 失败片段分解（2026-10-28）

来源：`reports/reference_complete_adopted_20261016/e0_dev_cost0p0.json`，checkpoint `ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c`。这是现有 dev 结果的诊断，不是新的 test 指标。

序列数：
97
；per-sequence 字段的简单平均 SR=
61.995
、PR=
77.672
、NPR=
74.181
。官方总指标仍以原报告为准。

| 序列 | PR | NPR | SR |
|---|---:|---:|---:|
| highright2ndboy | 3.177 | 2.541 | 1.307 |
| ab_pingpongball | 29.204 | 28.319 | 4.509 |
| blackbag | 9.048 | 3.81 | 4.58 |
| dotat43 | 5.591 | 1.935 | 5.074 |
| bikeboy128 | 6.25 | 5.469 | 5.804 |
| girl`sheadoncall | 11.304 | 7.246 | 5.935 |
| AQblkgirlbike | 15.584 | 8.225 | 9.07 |
| belowrightwhiteboy | 19.043 | 11.337 | 9.182 |
| theleftestrunningboy | 20 | 17 | 15 |
| midboyblue | 36.852 | 15.926 | 16.704 |
| folderatlefthand | 11.111 | 11.622 | 20.659 |
| leftblackboy | 25.581 | 20 | 21.041 |
| redgirl1497 | 30.995 | 30.394 | 22.477 |
| whitebetweenblackandblue | 33.082 | 32.543 | 28.746 |
| leftfallenchair_inf_white | 5.103 | 24.138 | 29.793 |
| bikeafterwhitecar | 47.514 | 44.199 | 31.939 |
| left2ndboy | 34.026 | 39.617 | 33.075 |
| rightmirrorlikesky | 31.596 | 34.123 | 33.92 |
| midboy | 40.496 | 25.62 | 34.278 |
| moto2north1 | 50 | 37.281 | 35.673 |

观察：这些序列是当前 E0 dev 中最值得优先复盘的失败片段；仅凭汇总表不能判断是候选覆盖、主模态选择、query 替换还是模板更新导致。下一步应在保存逐帧 evidence 后，对这些序列做同状态主框/query 框和模板写入分析。
