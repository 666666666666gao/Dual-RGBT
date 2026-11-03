# Fixed P1A DropMoE RGBT210 Full Result

The fixed epoch20 P1A executor control finished natively COMPLETE/exit0
on P25, covering all 210 RGBT210 TEST sequences and 104,706 frames.
The reviewed official CPU scorer ran once. The existing compute summarizer
then bound the exact score/profile and complete trace route counts.

| Measurement | Result |
| --- | ---: |
| PR | 88.9871035906 |
| SR | 64.6342733358 |
| Full-run FPS | 18.3249008860 |
| Mean latency | 54.5705543633 ms |
| P95 latency | 77.9020160044 ms |
| Total parameters, including frozen | 103,374,215 |
| Mean tracked-frame GMACs | 57.8178674413 |

PR/SR are descriptively above the original goal88.7/62.9 by
0.2871035906/1.7342733358 points. They remain below the November3 plan's
stronger DropMoE row91.2/65.4 by2.2128964094/0.7657266642 points.
This is an exploratory TEST result, test_eligible=false, not independent
final validation or evidence that the overall three-benchmark goal is met.
It is a fixed P1A control BEFORE P1B refresh, not a refreshed-policy or
C2/M1/M2 result. It does not change the predeclared P1B recipe.

FPS includes image decode and initialization under concurrent service
load; model loading is excluded. This is not isolated throughput.
MACs weight the actual ten-route counts over104,496 tracked frames;
initialization arithmetic and nonlinear/normalization/ROI/memory/bookkeeping
operations are excluded. The matching-source fixed-route CPU profile
was executed on P25.

Exact raw score/curves/compute/native/config/receipt/progress/review and
operation logs are archived in
`reports/p1a_completed_scores_r1_20261103/dropmoe/rgbt210/`.
Score SHA256:
`671f93e9c6a2b0c9f86a79d783bbd40da35495eab454c99159bda2d1499b0005`;
compute SHA256:
`44dd1eedcf2567bc9bdfb2e5fed6983f8053905eb682ab4408f244f63292ee83`.
The archive audit binds every raw file. Do not rerun the scorer.
