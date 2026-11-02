# Fixed epoch39 full RGBT234 result

The P25 epoch39 run completed with native exit0,234sequences116650frames.
The existing CPU controller scored it once and full compute passed.

| Benchmark | Fixed epoch39 metrics | Full-run FPS | Parameters | Mean tracked GMAC |
|---|---|---:|---:|---:|
|RGBT210, previously completed on26|PR89.01895215001173/SR65.40234868473182|10.606678877852053|103374215|53.6136161127|
|RGBT234, completed on25|MPR91.02117157206567/MSR67.77700631717165|15.18302924381185|103374215|53.30830997497526|

Both score files bind the exact same checkpoint SHA256:
4405b608e961ff3bd07b6cf1c26c6201e1f6b82b7d429f9f1d9d53b99707d3ae

RGBT234 descriptively exceeds target90.9/65.6 by0.1211715721/2.1770063172
percentage points. RGBT210 exceeds88.7/62.9 by0.3189521500/2.5023486847.
These fixed exploratory TEST results are test_eligible=false.
They do not select this checkpoint or change the TRAIN recipe; neither
LasHeR target completion nor independent final validation is established.
All five retained checkpoint dependencies remain protected.

RGBT234 mean65.86300954452618ms/P95 92.66114164984174ms.
Exact score, curves, compute and runtime/native artifacts are archived in
e0_ep20_p25_full_scores_r1_20261102.

RGBT234 score SHA256:
f1753beec22a510e329dac5fa1196fca638b4be8cc1cb10fb3d995f6be82dc39

RGBT234 compute SHA256:
1c0214d3507924339bc7ca63b16061e37963c373586571f22ee8f751c2f2e49e

FPS reflects different concurrent server workloads and is not an isolated
speed comparison. Matching49-source profile708d3aaafb486a5f21a28bf134207a3ae2bf59e0c61607336e9a6addd4a8989e
used epoch1 TRAIN smoke weights with matching shapes/routes; compute is
weighted by actual epoch39 routes/initMACexcluded, not a fresh per-epoch
profiling forward.
