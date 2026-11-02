# Fixed epoch37/38 full RGBT234 results

Both P25 native tasks completed with exit0,234sequences116650frames.
The existing CPU controller ran the official scorer once and produced
COMPLETE_FULL_RUN_COMPUTE_SUMMARY. Exact raw score, curves, compute,
config, COMPLETE receipt and native status are archived in
reports/e0_ep20_p25_full_scores_r1_20261102/.

| Epoch | MPR | MSR | FPS | Parameters | Mean tracked GMAC |
|---|---:|---:|---:|---:|---:|
|37|90.86055376061938|67.63509304941921|17.265811068598083|103374215|53.403842500714674|
|38|89.95714563512905|67.2895653604898|17.17544709438687|103374215|53.32540764646949|

Epoch37 is 0.0394462394 percentage points below MPR target90.9 and
2.0350930494 above MSR target65.6. Epoch38 is 0.9428543649 below MPR
and 1.6895653605 above MSR. These descriptive fixed TEST results remain
exploratory_test=true/test_eligible=false. No checkpoint promotion,
training/threshold selection or independent final validation is inferred.
No complete fine-tuned LasHeR result was available at this check.

Epoch37 score SHA256:
58b2de71fc8f8ae3d057bb46716def1dd8b9d13570996d3c7f2b2c5a90fdf2fa

Epoch37 compute SHA256:
1cdda746bf2c239ae1759c82be1d65dfb440e358309bfe57fb4b2c0cae75e1f3

Epoch38 score SHA256:
2c804dadf29b66d7d55f4dc7181a300a5b103e41273dc13ce09ca88def4b099c

Epoch38 compute SHA256:
1aa9f8bf3ee0bd4cc4c2b86138478241100dd1d178952fa0746bc6d61c544d3b

FPS was measured under concurrent workloads. The matching49-source
MAC profile708d3aaafb486a5f21a28bf134207a3ae2bf59e0c61607336e9a6addd4a8989e
used epoch1 TRAIN smoke weights with matching model shapes and routes.
Compute weights that profile by actual evaluation route counts and
excludes initialization MACs. No per-epoch profiling forward is claimed.
