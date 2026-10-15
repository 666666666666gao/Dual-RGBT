# Scale-candidate checker SOURCE review

**SOURCE_PASS_WITH_WARNINGS — source_verdict PASS, 0 blocking issues, no required code changes.** This verdict admits the new checker source only. **Actual new M0 remains unrun and unadmitted.**

Reviewed `tools/check_scale_candidate_sanity.py` and `docs/SCALE_CANDIDATE_M0_VALIDATION.md` against the frozen training plan, current producer/checkpoint path, accepted training SOURCE review, both prior checkers and M57. All **68 registered source files** and **81 sealed inputs** retained their original hashes. No reviewed source or plan was edited.

| Reviewed artifact | SHA256 |
|---|---|
| `tools/check_scale_candidate_sanity.py` | `0268ba1ab578f4173fa40a186846b1fb5714ca73a522ecce292bb441c6e8b602` |
| `docs/SCALE_CANDIDATE_M0_VALIDATION.md` | `0bfb9fa88771eae25356ddde6869f3fab21e1b8b54ce3da61b220a6b3412fa26` |
| Frozen training plan | `410023bc9c687c5a0a7829040908ff2126835b3d70eb9e2a669d6e6f48afdc0c` |
| Accepted training SOURCE review JSON | `78c3cd5b478d90f6590b794ae8fa53ea436230e5667e3d8ba0cb60898f685abc` |

The 14 probe fields match `train_scale_candidate_policy.py` exactly. The checker reads the new family count, union count and request cell fields; it rejects the old 12-field/single-count schema. Its native command, stage, saved configuration, update/sample counts and row ordering agree with the current producing path.

M57 is addressed directly: all three IoU/gain scalars are finite before entering `max`, and all three loss scalars are finite before the float32 total is checked. The signed target remains executed query IoU minus STOP IoU. The clipping helper agrees with the isolated original margin10 clip function. The global-cell mapping agrees with the current top-left square geometry, and the selected centre must lie inside the actual image dimensions.

Independent verification used Python 3.12.12, NumPy 1.26.4 and stdlib only. **10 actual checker subprocess cases passed their expectations: 1 accepted, 9 rejected.** Four hand-derived clipping/IoU examples cover a left boundary, an interior partial overlap, a smaller overlap and a bottom/right clipped zero overlap. The accepted 64-row artificial fixture contains **16 positive, 32 negative and 16 zero gains**. Sampling uses only the actual four NumPy sampler statements extracted by AST; the manifest and dimensions remain artificial. The new checker also parses under Python 3.7 grammar.

| Independent case | Actual result | Rejection location |
|---|---|---|
| Hand-derived clips and signed gains | exit 0; all numerical errors 0 | — |
| M57 NaN main IoU / query IoU / delta | three exit 1 results | checker line 80 |
| M57 incorrect finite loss total | exit 1 | line 67 |
| Old 12 reload fields | exit 1 | line 41 |
| Old single candidate count | exit 1 | line 95, missing new union field |
| Geometrically consistent cell in padded image area | exit 1 | line 105 |
| Rank within union but outside selected family | exit 1 | line 101 |
| Incorrect finite GT IoU with consistent delta | exit 1 | line 111 |

All expected rejection stderr and process records were retained and inspected at the listed assertions. The independent worker had no unexpected failure: native session **40016**, initial chunk `45237a`, terminal chunk `3382d7`, **exit 0**. Worker SHA256: `4f9891683fdb0044a435ba7df4da5673db9cc8aa01c5fda13694b6f1e8a50952`. Result SHA256: `1d9f956e646717c520037e77cd846f4baa5640e1b9362beb376cdecd699d7d89`.

The root's separate closed report records **16 synthetic/mock cases, 1 accepted and 15 rejected**, with SHA256 `75fe9049ec0f9fb277d5490b9e2461868b33e7dddc974e5ab0b3c8c834d59232`. It was read only after the root reported its original runner terminal. These counts are separate protocol evidence, not actual M0 samples or neural executions.

Limits are explicit:

- Supplied receipts, GT manifests and dimensions are not authenticated by this checker. Native gradients, Adam updates, frozen weights, reload tensors and causal replay remain supplied runtime assertions requiring actual records and independent result review.
- Family/union counts are checked for schema and bounds; full preview score maps/pools are absent, so NMS ranking and exact union membership are not independently reconstructed.
- No Torch import, original image/weight/model load, NN execution, SSH, GPU/resource/progress/status/CIM/process read, environment change or queue modification occurred. Existing E0/E1 work is untouched.
- This SOURCE verdict does not admit actual new M0, formal B, future C/evaluation integration, metrics, targets or latency. The reviewed document states the same boundary.
- Requested reviewer `gpt-6-astra` / `max`; actual model, effort and routing are **UNATTESTED**. Review independence is **same-family**, acceptance **provisional**.

Private deterministic evidence: `C:/Users/gb/.codex_tmp/dual_rgbt_scale_checker_source_review_20261015_agent`. Historical M57 failure records and earlier source reviews were preserved. Only this MD and its paired JSON are public review outputs.
