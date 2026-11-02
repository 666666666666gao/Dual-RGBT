# R15 E0+M2 Full LasHeR Result

Native `eval_e0_m2_lasher_r15_p26_20261101` completed with exit0 at
2026-11-02 04:46:56UTC. All245 sequences and220,703frames were present.
Official toolkit scoring ran once, followed by the previously reviewed
full trace-weighted compute summarizer; both completed successfully.

| Variant | PR | NPR | SR | FPS |
| --- | ---: | ---: | ---: | ---: |
| Matched frozen E0 | 73.790940 | 70.293435 | 58.617261 | 8.365369 |
| E0+M1 guard | 74.223810 | 70.504968 | 59.018088 | 8.270958 |
| R15 E0+M2 | 68.165608 | 64.880209 | 54.202170 | 6.268035 |
| M2 minus E0 | -5.625333 | -5.413226 | -4.415091 | - |

R15 misses the unchanged LasHeR targets75.8/71.9/60.2 by
7.634392/7.019791/5.997830points. It is not promoted. Its already completed
RGBT210 andRGBT234 results also regressed; no further old R15 grid or
repeat inference is needed.

These variants share frozen parent SHA256
`ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c`.
R15 uses the original candidate identity head SHA256
`7e7df5d25d3e8667fe38a4ca8f6ec3662308ffcb52a09bd16aecaab59a9cb4be`.
It has no M1 gate or M3 memory. TEST is exploratory_test=true and
test_eligible=false; no TEST-derived threshold, module setting, seed or
checkpoint selection follows from these numbers.

## Compute And Runtime

Total parameters103,490,697: E0 contributes103,374,215 and M2 contributes
116,482. Mean latency159.539640ms, P95193.101980ms. Mean tracked-frame
103.284795GMAC, all-frame103.249307GMAC. Counts include8,939 candidate
generation invocations and identity-head calls even when no match was
selected;245 initial anchors are counted in all-frame costs.

The old M2 replaced E0 on7,955/8,939 invocations; this is action frequency,
not replacement correctness. FPS includes decoding, initialization and
concurrent load, excluding model loading. MACs cover measured and
formula-checked convolution/dense/attention operations, excluding bias,
normalization, nonlinearities, crop/ROI, box decoding, memory and
bookkeeping. The old route profile's wrapper-only source difference is
explicitly retained in the compute report.

## Evidence And Next Experiment

All score/curve/compute files and runtime config/COMPLETE/native/plan/
progress records are archived under
`reports/e0_m2_online_scores_r15_20261101/`.
Score SHA256:
`d861e0d5caa0552993c7a0cd2e9b2949ef580201a19a365aeef8c3cd22243c3e`.
Plan SHA256:
`004ed6f3b623b95e95ab075602e80bef1a087c2c9e6c5d9cccfc271984d1ad6f`.
Native SHA256:
`1f823e21c51970f0a41aab3244b2f2e1b3ba363cc232a0194e6ea7befd2d515c`.
COMPLETE SHA256:
`7d7161e214203f980d355424a74ab56e5c8dfe94f17227a1faf91c3a8df0c9cf`.

Before this TEST result, the full TRAIN replay had already found326/2,899
reliable states falling belowIoU0.5 and179/800 lost recoveries. The current
E0 supplement and predeclared relative KEEP recipe address that measured
TRAIN problem. The new supplement's eight reviewed shards are running
on26GPU0/1; original24candidate/anchor arrays remain reused and immutable.
Full acceptance and real relative training are still pending, and no
claim that this change improves any benchmark is supported yet.
