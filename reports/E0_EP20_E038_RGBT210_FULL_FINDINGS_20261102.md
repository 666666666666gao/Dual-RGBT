# E0 Epoch38 Full RGBT210 Result

Native task`eval_e0_ep20_e038_rgbt210_r1_p25_20261101` completed with
exit0, all210sequences and104706frames. The already running P25CPU
controller called the unchanged official scorer once, then the full
route-compute summarizer. No manual scorer call was made.

| Measurement | Value |
| --- | ---: |
| PR | 88.5610475532 |
| SR | 65.0301100409 |
| FPS | 16.7879174773 |
| Mean frame milliseconds | 59.5666497259 |
| P95 frame milliseconds | 79.3503559999 |
| Parameters | 103374215 |
| Mean tracked-frame GMACs | 53.6753561206 |

PR is0.1389524468points below the requested88.7; SR is2.1301100409points
above62.9. Previously fixed epoch39result was89.0189521500/65.4023486847,
and epoch40was88.368511/65.080225. These are reported fixed checkpoint
outcomes, not TEST-based checkpoint selection or evidence that another seed
or hyperparameter change will improve the method. No recipe was changed.

This result is exploratory_test=true/test_eligible=false. It does not
establish independent final validation, LasHeR target attainment or
three-module scientific acceptance. The complete epoch39result remains a
descriptively observed210target crossing, without promotion based on TEST.

FPS measures actual concurrent-load execution, including the original
run's initialization timing. It is not an isolated throughput comparison
with the26epoch39run. Mean tracked-frame MACs exclude initialization and
aggregate actual E0route/token counts. The matching49source fixed-shape CPU
profile was executed using epoch1TRAIN-smoke weights, then reused for the
same model shapes/routes. No new epoch38CPU profiling forward is claimed.
Normalization/nonlinear/ROI/decode/memory work is outside the standard
conv/matrix MAC convention documented by that profile.

The score's native hash, checkpoint identity and complete counts were
checked; compute output binds the exact official score/profile hashes.
Exact score, curves, compute, config, COMPLETE receipt and native status
are archived under`e0_ep20_p25_full_scores_r1_20261102/`.

- Score SHA256:`192660a441eb929848bc77142dcf13424abc00f23f2b1b9f9e3eb668cad90b7e`.
- Compute SHA256:`fba39e70051b54789c33cb6d5a2d08b498aca4d7e52660b32dd586392aedfc62`.
- Profile SHA256:`708d3aaafb486a5f21a28bf134207a3ae2bf59e0c61607336e9a6addd4a8989e`.

At07:47UTC epoch37RGBT210was also nativeCOMPLETE0; its scoring remains
owned by the same controller. Do not start a competing scorer or infer
metrics from completion alone. Other fixed evaluations remain in progress.

At07:50UTC the controller had also scored epoch37once and completed compute.
Its verified PR89.1079925776/SR65.3956417928 is documented separately in
`E0_EP20_E037_RGBT210_FULL_FINDINGS_20261102.md`. No duplicate scoring occurred.
