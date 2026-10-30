# Experiment execution tracker

| Run | Stage | Evidence / scope | Status |
|---|---|---|---|
| size_tail_sanity_b8_20261030 | S1 | Train 32 pairs / 4 real updates; 66 parameters; strict reload | SANITY_PASS, native exit 0; peak allocated 0.5906 GiB |
| size_tail_sanity_b64_20261030 | S2 | Train 256 pairs / 4 updates | SANITY_PASS, native exit 0; peak allocated 2.0113 GiB |
| size_tail_sanity_b384_20261030 | S2 | Train 1536 pairs / 4 updates, LR 4.8e-5 | SANITY_PASS, native exit 0; 4 updates; allocated10.14/reserved12.66 GiB; 363.02 seconds; no convergence claim |
| size_tail_sanity_b576_20261030 | S2 | Train 2304 pairs / 4 updates, LR7.2e-5 | RUNNING, GPU0 wrapper1105161; projected reserved19GiB; result pending |
| size_tail_b384_eval_smoke_20261030 | S3 | Adapted checkpoint provenance and 11-frame tracking | COMPLETE, native exit 0; two changed tensors admitted; frozen evaluation passed |
| size_tail_parent_control_smoke_20261030 | S3 | First evaluator launch | FAILED before prediction: missing OpenCV environment variable; preserved |
| size_tail_parent_control_smoke_r2_20261030 | S3 | 11 dev frames; serialized predictions and action/modality equality | COMPLETE, native exit 0; all 265 state tensors frozen |
| size_tail_parent_control_dev_20261030 | S3 | Full dev 97 sequences / 54,646 frames; exact comparison against original parent | RUNNING; GPU1 wrapper 1100268; CPU scorer 1100867 waits at 180s intervals |
| size_tail_b384_dev_20261030 | S3 | Full dev of four-update checkpoint; diagnostic only | RUNNING, GPU2 wrapper1106853; CPU scorer1107101 |
| size-tail formal | T1 | At most ten sampled epochs | NOT ADMITTED; full parent equivalence still pending |
| last-five full dev | V1 | Official dev metrics and convergence stopping | Retention tested; end-to-end epoch evaluation/selection remains pending |

The independent evaluator and scorer are documented in
`docs/SIZE_TAIL_DEV_EVALUATION_20261030.md`. All new evaluations are dev diagnostics,
not test-eligible runs. Concurrent service FPS is not an isolated speed benchmark.
The eight-card recovery queues continue in their separate roots.
