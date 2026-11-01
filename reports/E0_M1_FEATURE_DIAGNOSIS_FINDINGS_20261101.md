# M1 TRAIN Feature Diagnosis

The failed weight-decay control does not identify why M1 performs poorly.
This CPU-only audit revalidated the original eight-shard TRAIN collection
through the admitted loader, checked2812 input bindings, and reproduced the
saved normalization and epoch5 confusion matrices exactly. No training,
threshold choice, TEST input or live inference source change occurred.

Raw report: `reports/E0_M1_TRAIN_FEATURE_DIAGNOSTIC_20261101.json`,
SHA256 `801e9033a7a74f3a51f8a23b0cb6f34154b451ef80530e90bab98f2b4c6ef7e4`.
Executable: `tools/audit_m1_train_features.py`, one CPU thread, CUDA hidden.

| Quantity | Fit | TRAIN Component Holdout |
| --- | ---: | ---: |
| Total states | 71843 | 19248 |
| Reliable states | 48570 | 13966 |
| Reliable predicted nonreliable | 2524 | 1842 |
| Reliable predicted lost | 2033 | 1568 |
| Reliable predicted lost with probability >=0.9 | 46 | 122 |
| Macro recall | 68.3946% | 53.1125% |
| Lost recall | 81.4711% | 42.3843% |
| Reliable interrupted fraction | 5.1966% | 13.1892% |

The interruption fraction refers to sampled localization states. It is not the
observed fraction of blocked dynamic-template writes, which also depends on
the original write proposal rule. Full-run traces and official complete scores
are needed to measure the actual guard effect.

## What The Audit Supports

- Fit-only normalization is exactly reproduced. Holdout standardized RMS is
  0.928 for preview features,0.929 for history,0.968 for deep ROI,1.071 for
  score statistics,0.961 for boxes and0.848 for modality. These aggregate
  values do not show a broad normalization explosion.
- The largest absolute standardized history/box values exceed20 in some
  channels, but similar large values already occur on fit. This alone does
  not identify a holdout-specific failure or justify clipping.
- The ROI coverage channel has zero standardized variance in both partitions.
  It is constant for these cached states and supplies no variation to this
  head on this collection. This does not imply that coverage can be omitted
  from M2 candidates or future observations.
- Reliable-to-lost confident mistakes are more numerous on holdout even
  though holdout contains fewer reliable states. The lost-class recall gap
  and observed fit/held divergence deserve a sequence-aware diagnostic or
  a targeted training intervention, not further blind weight-decay sweeps.
- Uncertain states are5749/71843 on fit and1264/19248 on holdout. Class
  imbalance is real, but not proof that balancing will improve reliable
  false positives or tracking SR. Any loss/sampling change needs a separate
  TRAIN-only experiment with reliable interruption reported alongside recall.

The audit is descriptive. It does not distinguish memorization, background or
identity dependence, within-class distribution changes, or calibration as a
causal explanation. Upstream E0 saw component holdout sequences; no independent
generalization or benchmark improvement is established.

The protected E0, existing fixed epoch5 M1 evaluations and M2 collection remain
unchanged. Current complete benchmarks should finish before choosing whether
the guard has useful end-to-end behavior.
