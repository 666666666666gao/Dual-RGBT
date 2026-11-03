# Current P1 Fixed Route CPU Measurement

The exact P25 P1A DropMoE epoch020 executor was loaded and ten routes
executed on CPU at 2026-11-03 03:22:32 UTC. Native SSH execution exited0.
All route profiler counts matched independent module formulas plus
explicit functional patch projection. Model state stayed bit equal,
gradients remained absent and CUDA was not initialized.

| Quantity | Actual Result |
| --- | ---: |
| Total unique model parameters, including frozen | 103,374,215 |
| STOP, either primary modality | 40.042316672 GMAC |
| PRIMARY / SECONDARY / PREVIEW local query | 61.608700800 GMAC |
| EXPAND full query | 80.074910592 GMAC |
| Primary / local auxiliary / full auxiliary tokens | 385 / 193 / 385 |

These are batch1 convolution, dense and attention arithmetic counts,
using synthetic templates [1,2,6,128,128] and search [1,6,256,256].
One MAC equals two counted profiler FLOPs. Elementwise operations,
normalization, decoding images, crops, scalar box calculations,
memory traffic and tracker bookkeeping are excluded. This is neither
whole-system FLOPs nor a measured FPS or tracking-accuracy result.

The current forward_action includes the gain head called by predict.
The new source-bound profile replaces reuse of an older measurement
whose model.py, data.py and wrapper hashes differ. Its route numbers
may support later matched-source trace weighting, with initialization
excluded from the tracked-frame denominator. No such full-dataset
aggregation has occurred yet. P26 source/parameter compatibility must
be explicitly checked before using these rows there.

## Exact Evidence

- Raw result: `reports/p1b_route_profile_r1_20261102/measurement.json`
- Result SHA256:
  `9d63719a1d16d83310b7100939a64183dcf5c010c96b7d4522c1602651e8542a`
- Profiler: `tools/profile_p1b_e0_route_macs.py`,
  `04db7c1c48732e3d5fcf93541654b46eb9029df213391cb84c057897ef1e31ed`
- Source/admission review:
  `reports/E0_EXECUTOR_V2_P1B_CPU_ROUTE_PROFILE_SOURCE_REVIEW_R1_20261102.json`,
  `2c80fefb3f6c0fee9eb9933698dc05db8e908128e57c12c07f2c33a96dd80088`
- Exact original disabled draft, bound execution argv, raw stdout/stderr
  and terminal exit receipt: `reports/p1b_route_profile_r1_20261102/`
- P1A epoch020 checkpoint:
  `e0aedf099c5ee044d3ebfa56479e76c706f7afc475f4869ec24af67243600a6b`
- Factory checkpoint:
  `d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`

Only the new profiler and its report were added to the P25 snapshot.
Runtime rechecked all99 reviewed sources and both checkpoint hashes;
the original98 collector sources were not changed. Engineering review
is honestly UNATTESTED/same-family/provisional, not a mandated-model
scientific audit or proof that the benchmark targets have been met.
