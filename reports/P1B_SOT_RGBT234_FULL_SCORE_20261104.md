# P1B SOT RGBT234 Full Score

## Run

- Task: `eval_p1b_e005_rgbt234_sot_p26_r2_20261104`
- Host/GPU: P26 / GPU1
- Coverage: 234 sequences, 116650 frames
- Checkpoint: fixed P1B epoch005, cost weight `0.05`
- Test policy: `exploratory_test=true`, `test_eligible=false`
- Native state: `COMPLETE`, exit code `0`
- Optimizer updates/backward calls: `0/0`
- Model state: unchanged

## Metrics

| Metric | Result | Target | Gap |
| --- | ---: | ---: | ---: |
| MPR | 90.48423518357431 | 90.9 | -0.41576481642569 |
| MSR | 67.05901204183532 | 65.6 | +1.45901204183532 |

- FPS: `13.745663465151463`
- Mean frame time: `72.75021700736663 ms`
- P95 frame time: `100.59848400123883 ms`
- Parameters: `103374215`
- MACs: not newly profiled in this run

## Archive

The one-use official CPU score and raw audit files are archived under:

`reports/p1b_rgbt_completed_scores_r1_20261104/sot/rgbt234/`

Do not rerun the scorer or overwrite this archive. The result clears the MSR
target but does not clear the MPR target or the overall Dual-RGBT goal.
