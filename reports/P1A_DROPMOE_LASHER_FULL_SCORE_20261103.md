# P1A DropMoE LasHeR Full Score

At 2026-11-03 09:32 UTC, the fixed epoch-20 DropMoE E0 full LasHeR inference
was verified native `COMPLETE`, exit 0, with 245/245 sequences and 220,703
frames. The evaluation used checkpoint SHA256
`e0aedf099c5ee044d3ebfa56479e76c706f7afc475f4869ec24af67243600a6b`.
The official scorer and existing full-run compute summary were run once.

| Metric | Result | Target | Delta |
| --- | ---: | ---: | ---: |
| PR | 74.6503142464 | 75.8 | -1.1496857536 |
| NPR | 70.6056404634 | 71.9 | -1.2943595366 |
| SR | 59.3842943442 | 60.2 | -0.8157056558 |

The score is `exploratory_test=true` and `test_eligible=false`. It must not
be used for training, threshold or checkpoint selection, or an independent
final-validation claim. This is a fixed-control measurement, not evidence
that the active C2 variants improve tracking.

Concurrent-load throughput was 12.1624 FPS. The source-matched mean tracked
frame compute was 56.9204 GMAC; parameter count was 103,374,215. FPS includes
decode and initialization and reflects concurrent host load, not isolated
throughput.

Raw score, curves, compute summary, native status, config, completion receipt,
progress, source review and logs are hash-bound in
`reports/p1a_completed_scores_r1_20261103/dropmoe/lasher/audit.json`.
The native run is
`/data2/gaob/Dual-RGBT/p1a_eval_runs/eval_p1a_e020_lasher_dropmoe_p25_r1_20261102`.
