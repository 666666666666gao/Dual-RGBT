# P25 Remaining LasHeR Evaluations Completed

The native tasks for the fixed retained epochs36,37,38 all completed exit0,
each245 sequences/220703 frames. The existing P25 score controller alone
performed the official score and full compute aggregation once for each task.
This continuation fetched the existing artifacts and verified their exact
native/config/receipt/checkpoint/score/compute bindings; it did not rescore.

| Epoch | PR | NPR | SR | FPS | Tracked-frame GMAC |
|---|---:|---:|---:|---:|---:|
|36|74.8722359263|71.1310428026|59.6292388009|13.245672894|54.391436093|
|37|74.5481986943|70.9739001752|59.4822610692|12.554254151|54.206327282|
|38|75.2461644981|71.2026264526|59.7760973237|12.871714089|54.365942181|

All use103374215 parameters. FPS includes image decoding and initialization,
excludes model load, and reflects concurrent server load. MACs are weighted
by actual tracked-frame routes, exclude initialization, and reuse the
earlier TRAIN-smoke route profile with matching architecture; this does not
claim a newly profiled network for every checkpoint.

| Epoch | Score SHA256 | Compute SHA256 |
|---|---|---|
|36|832020944f203731a48bbe3f2f34675237061d7d44e7cacf0434a3df5373291c|aae487c6d1ddb2e86dd2f031e8a1cd2a601827349f1d2c551da1c1d80f2a08ff|
|37|88a1c659b426b45904ff2f800f05f6c7be616b9d900af1d154e9166b84244cc9|9c9264ed1c3dcbff239dca383ab72321b1059cbe56161467e4e09b6317b44e7b|
|38|b62c1250929fb439460320aea9dc909236651f0a94f7970dd6f576baa9e899e7|5d9f11fac0df4fd0fd032fc494d80acf4ab114ae734079466d179bf7fc382b38|

Full raw artifacts reside in `reports/e0_ep20_p25_full_scores_r1_20261102`,
including native status, config and completion receipts. The fixed last-five
inventory now contains14 full scores/compute summaries; only epoch39 LasHeR
is pending. At11:13UTC its actual P26 child1140162 remained live with220
sequences/210763 frames, native RUNNING and no score.

At11:13UTC P25 all4 GPUs were idle at12MiB and250W after normal completion,
with no remaining eval or score-controller PID. This is completion, not an
unexpected exit. Do not restart these evaluations or the completed40-epoch
fine-tune. Keep retained036-040 and all active/audit dependencies.

These are exploratory full-TEST results, `test_eligible=false`. None of these
three runs meets all LasHeR75.8/71.9/60.2 targets. Epoch40's previously
archived75.5873846215/71.6144248298/60.0960017977 also misses the targets.
This inventory does not select/promote a checkpoint, alter thresholds, or
establish independent final validation. The original three-benchmark and
protected E0/M1/M2/M3 goal remains unmet.
