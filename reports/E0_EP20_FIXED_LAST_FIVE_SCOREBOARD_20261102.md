# Fixed last-five full result inventory

| Epoch | Benchmark | Full metrics | FPS | Parameters | Mean tracked GMAC |
|---|---|---|---:|---:|---:|
|36|LasHeR|PR 74.872236 / NPR 71.131043 / SR 59.629239|13.245673|103374215|54.391436|
|36|RGBT210|PR 89.071394 / SR 65.355879|14.803747|103374215|53.634118|
|36|RGBT234|MPR 90.286485 / MSR 67.408980|15.970845|103374215|53.411775|
|37|LasHeR|PR 74.548199 / NPR 70.973900 / SR 59.482261|12.554254|103374215|54.206327|
|37|RGBT210|PR 89.107993 / SR 65.395642|16.375906|103374215|53.632201|
|37|RGBT234|MPR 90.860554 / MSR 67.635093|17.265811|103374215|53.403843|
|38|LasHeR|PR 75.246164 / NPR 71.202626 / SR 59.776097|12.871714|103374215|54.365942|
|38|RGBT210|PR 88.561048 / SR 65.030110|16.787917|103374215|53.675356|
|38|RGBT234|MPR 89.957146 / MSR 67.289565|17.175447|103374215|53.325408|
|39|LasHeR|No archived full score||||
|39|RGBT210|PR 89.018952 / SR 65.402349|10.606679|103374215|53.613616|
|39|RGBT234|MPR 91.021172 / MSR 67.777006|15.183029|103374215|53.308310|
|40|LasHeR|PR 75.587385 / NPR 71.614425 / SR 60.096002|7.799877|103374215|54.354810|
|40|RGBT210|PR 88.368511 / SR 65.080225|11.686902|103374215|53.697104|
|40|RGBT234|MPR 90.362006 / MSR 67.470107|12.559872|103374215|53.336867|

Every fixed retained epoch, ordered by epoch. No TEST ranking, checkpoint promotion, seed tuning or final independent validation. Missing local archives do not prove a live or failed remote task. FPS reflects concurrent server loads; route-weighted MAC provenance remains in each compute artifact.

The JSON inventory binds each score, compute, native and receipt to exact byte hashes.
All complete entries verify full benchmark counts and native exit0.
The same epoch must use the same checkpoint hash across available benchmarks.
No missing score is replaced with a partial score or a different epoch.
Per-epoch compute files retain the actual profile provenance; initialization MACs are excluded.
