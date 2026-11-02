# P1B Implementation and P26 Allocation

Status: current local source passed independent R3 review with zero blocking
issues and all12 source hashes verified. No host/input/recipe admission,
deployment, real TRAIN smoke, full collection, policy fit or benchmark is
established by this report. Review report SHA256:
`97cf80bc10319dd7babe9d128d5b144e908680b8b23bcfd928ea23e10a18f6ac`.
Its admitted execution-plan bindings are empty.

R1 found two real issues: missing explicit TF32 settings and missing previous
full-fit evidence for round2. Both are fixed, effective precision is recorded,
and round2 requires the completed first-round five-epoch policy/native/artifact
bindings. Ten local CPU checks passed. R2 predates the precision-record
addition; only R3 matches the current files. The review is same-family and
provisional, with honest unverified model attribution.

The user's latest authorization includes all four P26 GPUs. P25's two active
P1A jobs continue unchanged. The predefined P1B input is epoch20 from each
completed P1A branch, rather than a checkpoint selected with TEST metrics.

## Implementation

New versioned entrypoints preserve direct E0 STOP plus four actions:

- `dual_rgbt/e0_policy_refresh.py`: explicit factory and strict completed-P1A
  loading; frozen executor; preview LR1e-5, quality/gain LR1e-4; current-preview
  counterfactual targets, both modalities and all four query actions.
- `scripts/collect_e0_policy_refresh.py`: actual frozen-policy trajectories;
  GT labeling after the executed decision; compact template frame/box
  provenance, pre-state, history, search geometry, predictions and outcomes.
  Smoke verifies reconstruction against actual template/search tensors.
  Counterfactuals must preserve the actual tracker, and the model state
  must remain identical throughout collection.
- `scripts/train_e0_policy_refresh.py`: full declared TRAIN shard validation,
  reconstruction and offline fitting; fresh labels each update; preview,
  quality and gain gradients; fixed executor parameters and buffers;
  five retained policy epochs and strict reload/route/box/gain checks.
- `tests/test_e0_policy_refresh.py`: synthetic CPU contract checks only.

The objective reuses direct-E0 signed localization-loss quality/gain targets,
plus 0.1 spatial CE for centers inside the crop. It is not the candidate-E1
Joint-Quality ranking architecture. Existing localization loss clips target
corners to the crop; this inherited objective cannot distinguish all different
GT positions wholly outside the crop. Spatial CE omits such centers, whereas
quality/gain supervision remains present. No recovery improvement is claimed.

## Resource Plan

After exact source/input/recipe admission:

1. Complete and verify each P25 P1A native run, 20 epochs and 50,000 updates,
   retained epochs16-20, finite logs, epoch20 strict loading and branch binding.
2. Bind P26's relocated manifest and the original P25 manifest separately.
   Their current SHA256 values differ because the absolute data paths differ:
   P26 `dfc64432210b053fc9db517bae0771fdeb750a5d373b732949ef43e8b65b1a7a`;
   original P25 `389ba37de9f2256fd8a07feee5896510727268715c3b4b1f508696ac8a8babbc`.
   Compare ordered TRAIN names, boxes, valid frames and frame filenames;
   bind source data provenance before launch.
3. Run one real TRAIN collection smoke and two actual policy updates for
   each branch. Verify all three policy modules change, frozen tensors and
   buffers do not, reconstructed inputs match, and strict reload succeeds.
4. Partition all882 TRAIN sequences into four disjoint collection shards per
   branch, on P26 GPUs0-3. If both branches are ready, two workers per card
   are possible only after measured concurrent capacity admission.
   Use label stride1 in the draft; full trajectories must not be confused
   with the number of labeled states. Measure time/memory in smoke before
   publishing a duration estimate.
5. Fit five predefined policy epochs on the complete admitted collections,
   then collect fresh trajectories with the fitted policy and repeat fitting.
   A single round is incomplete closed-loop refresh. Sequential collection
   and fitting dependencies take priority over occupying every GPU.

Only new immutable task IDs and run paths may be used. Existing active P1A
sources, outputs and historical caches remain intact. No dev run is planned.
Full TEST reporting follows a completed admitted pipeline and remains
exploratory/test-ineligible, without TEST-based selection or tuning.

## Observations

At 2026-11-02 23:32 UTC P25 DropMoE was at 40,550/50,000 updates
(CSV epoch17) and had written its first retained checkpoint, epoch16.
SOT was at 46,600/50,000 (CSV epoch19), with epochs16-18 on disk.
All four original wrapper/trainer PIDs were alive. These are progress
observations, not new benchmark results.

At 23:27 UTC P26 had about6.3TB available on `/data2` and about120GB
available RAM. Its four GPUs were idle at the last direct query.

The factory and original manifest input transfers subsequently completed,
with destination SHA256 matching. All882 TRAIN manifest entries match
semantically after path relocation; this is not a full image-byte audit.
P26's existing root model/data differ from the reviewed source version.
Deploy only to an explicitly bound new source snapshot after host/input/recipe
admission. See `E0_EXECUTOR_V2_P1B_INPUT_PREFLIGHT_20261102.json`.
