# M1 Seed2028 RGBT210 Full Paired Result

The seed2028 M1 guard run completed on host26 with native `COMPLETE/exit0`.
The official toolkit scorer ran once after completion and the matching-source
full-compute summarizer passed. Coverage is all210 TEST sequences,104706 frames,
including104496 tracked frames.

| Configuration | PR | SR | FPS | Parameters | Mean tracked-frame GMAC |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0 control | 88.828508 | 66.124608 | 11.134969 | 103374215 | 64.023388 |
| E0 + seed2028 M1 guard | 89.435150 | 66.532916 | 11.428726 | 103491722 | 64.066284 |
| Guard minus control | +0.606642 | +0.408308 | | +117507 | +0.042896 |

The guard exceeds the stated RGBT210 targets of PR88.7 and SR62.9 by
0.735150 and3.632916 points respectively. Against the earlier seed2027 guard
score of89.516426/66.612586, seed2028 is lower by0.081276/0.079670 points.
These are descriptive full-TEST comparisons only. The test set is
`exploratory_test=true`, `test_eligible=false`; do not use these scores to pick
a seed, checkpoint, or threshold, or claim independent final validation.

The run recorded150 blocked dynamic-template writes out of16870 proposals.
This is an observed trace statistic, not proof that blocking caused the score
change. The measured11.428726 FPS and115.559908ms P95 latency were collected
under concurrent load; they are not isolated throughput claims. Mean MACs
exclude initialization and non-matrix operations.

## Evidence

- Task: `eval_e0_m1_seed2028_rgbt210_guard_r3_p26_20261101`, GPU3.
- Plan SHA256: `90f92df423aa16ae4fd103cdc1c7b3747604bf7951713bf7ad44abaf446fb5bf`.
- Native status SHA256: `a549ca39c6231a5e6262e8ca89da82b3b8e5e539754e0760942e8d4ce48cd949`.
- Completion receipt SHA256: `95252249299c98b39ffe86e42fcfc912257faed72cb38471c851df0f50e78563`.
- Score JSON SHA256: `8b7223683dabbfe76e6aef235f40737aab2b16d8d35b722381b7ce92b21f402d`.
- Score curves SHA256: `d39072864182be10f26e8e7a900ddc6a2bf384f16ad78a708a47fe835e9e7841`.
- Compute summary SHA256: `8055088b5994dcd5e62b00bc1a0e1a747e491ee4300dfa385c23055e7ab32fa8`.
- Compute profile SHA256: `58d4d837e5f2925c33ed469ff7d6523922cfce253f91ed968a9a73b96f44f1e0`.

The first scorer attempt used the Python3.7 training environment and stopped
at toolkit import (`TypeError: 'type' object is not subscriptable`); it created
no score output. The scorer then ran successfully once with the documented
Python3.10 `cpu-scoring` environment.
