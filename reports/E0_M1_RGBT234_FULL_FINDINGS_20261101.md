# M1 RGBT234 Full Paired Result

Both native runs completed with exit0 on2026-11-01, each covering all234
sequences and116650 frames. Each was officially scored once after completion;
the matching-source compute summarizer then verified full trace coverage.

| Configuration | MPR | MSR | FPS | Parameters | Mean Tracked-Frame GMAC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original E0 control | 90.459440 | 68.444253 | 11.810636 | 103374215 | 64.020143 |
| E0 + fixed epoch5 M1 guard | 90.762147 | 68.720338 | 11.631646 | 103491722 | 63.868695 |
| Guard minus control | +0.302707 | +0.276086 | | +117507 | -0.151448 |

Guard MPR remains0.137853 points below the requested90.9; MSR exceeds65.6
by3.120338 points. Do not round90.762 to a completed90.9 target. These are
exploratory full TEST results, not independent final validation. No test
threshold or checkpoint was selected from this comparison.

## Interpretation

The guard blocked171 of18716 proposed dynamic-template writes; the control
blocked0 of19068. The recorded full trajectories show an observed positive
effect on both210 and234, but LasHeR remains pending. These observations do
not establish robustness across training seeds or justify a general recovery
claim. The protected E0 and M1 head are the same fixed hashes used in the210
comparison. This is not the ongoing CADTrack-ep20 E0 finetuning result.

The234 paired per-sequence MSR changes include24 improvements,17 regressions
and193 ties within1e-6 percentage points. Gains include child3(+34.81),
notmove(+22.05) and scooter(+12.96); regressions include hotglass(-3.22),
rmo(-3.08) and kettle(-1.31). The210 SR comparison similarly has21 gains,
16 regressions and173 ties. Mean improvement does not establish uniform
benefit. These are descriptive TEST findings, not a sequence blacklist or
input to TRAIN, threshold or checkpoint selection.

The lower mean route-weighted GMAC reflects changed future action counts;
M1 still adds117376 head MACs per tracked frame. Mean GMAC excludes
initialization and non-matrix operations, with116416 tracked frames. FPS
includes initialization/decode under concurrent load; the observed lower
guard FPS is not an isolated model throughput measurement.

A single seed2028 TRAIN-only replication completed with unchanged
architecture/cache/split/recipe except seed. It transparently records that
TEST outcomes were already observed. It admits neither TEST inference nor
TEST-driven model/epoch selection. TRAIN component holdout remains the only
epoch selection criterion. Its exact plan/launcher review passed before
execution. Best epoch3 has component-holdout macro recall53.531759%.
Separate evaluation drafts await fresh review and TRAIN smoke. The original
fixed head remains protected; this is not a promoted replacement or a
near-target tuning success.

## Evidence

Archive: `reports/e0_m1_full_scores_r1_20261101/`, including234 scores,
curves, compute summaries and config/native/COMPLETE under `runtime/`.

| Artifact | SHA256 |
| --- | --- |
| rgbt234_control.json | 05890367c73d58249bf4a5b37bd24cfa93b209ee321d815b553e5d7a0415b9f0 |
| rgbt234_guard.json | 3912f684b22ee12f2cd2d4fde0a58cc899d14055f2ebb46645de4791236c36c5 |
| rgbt234_control.npz | 9c7b7012f529d57f75283d4c414cd41c89ade50291783df6695e9996d36a8e1c |
| rgbt234_guard.npz | c321176f0cc72f052ad6294d31d2e7de38d58ed04e2a3ab10f1d570cff7325a7 |
| rgbt234_control_compute.json | 37e1ba51d01572369807e8bf798c24dfd9174e854163ba8bf3c2ee88f0465bad |
| rgbt234_guard_compute.json | d6f0c07066c79a29e656914e23a8f20e3c2e2e67df175e2fe3d26db033585a4f |

All results retain exploratory_test=true/test_eligible=false; scientific
acceptance remains false and engineering review same-family/provisional.
