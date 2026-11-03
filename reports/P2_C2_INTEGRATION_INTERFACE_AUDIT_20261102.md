# P2 C2 Integration Interface Audit

The supplied November 3 plan specifies missing-stream embeddings/FiLM and
route-specific low-rank updates as P2 ablations. Two standalone primitives
and CPU tests now exist. They are not wired into the executor or any
training/collection entrypoint, and do not establish improved accuracy,
speed, full-executor equivalence or scientific novelty.

## Concrete Integration Requirements

1. `scripts/train_executor_v2.py:104` creates optimizer groups only from
   `model.backbone`, `model.decode` and `model.box_head`. A new top-level
   missing/FiLM module would be absent from those groups. Adapters installed
   under backbone would receive its lower learning rate. P2 must explicitly
   group original backbone parameters at 1e-5 and decode/head plus new
   missing/FiLM/LoRA parameters at 1e-4, with every trainable parameter
   present exactly once. This is an observed interface limitation for adding
   C2, not a defect in the unchanged P1A recipe.
2. `DualRGBT.encode` invokes `block.attn` and `block.mlp` without a route
   argument. The standalone RouteLoRA requires an explicit route and cannot
   simply replace qkv/proj/fc1/fc2 while leaving those calls untouched.
   P2 needs explicit routing through the attention and MLP operations:
   primary=0, local auxiliary actions1-3=1, full auxiliary action4=2.
   Shared adapters use one bank for all three execution shapes.
3. `DualRGBT.primary` concatenates a zero stream; `correct` starts from
   zeros then scatters queried auxiliary tokens. P2 must use the missing
   modality's template descriptor for filling and preserve true auxiliary
   evidence at queried spatial locations. Full auxiliary queries replace
   all 256 filled positions. No descriptor may use future frames or GT.
4. `forward_action` and `predict` both already calculate preview descriptors,
   but their primary/correct calls do not pass preview. In addition,
   `dual_rgbt/e0_policy_refresh.py:205` and `:210` directly call these methods.
   Updating only predict would leave the supervised and counterfactual
   training paths inconsistent. P2 must update all these versioned callers.
5. RouteLoRA owns its original linear as `linear`. Installing it changes
   state-dict names, for example qkv.weight becomes qkv.linear.weight.
   Strictly load the original executor before installing adapters, and
   construct the same declared C2 structure before strictly reloading a C2
   checkpoint. Do not silently accept missing or unexpected keys.
6. RouteLoRA.merge returns a separate frozen nn.Linear; it does not mutate
   the trainable module or maintain a persistent cache. Integrated inference
   must build and select complete per-route merged weights explicitly.
   Primitive merged-output tests are not an integrated FPS measurement.

## Implemented Scope and Checks

- `dual_rgbt/route_lora.py`: independently initialized A banks and zero B
  banks, explicit selected-route forward, independent frozen merged linear.
- `dual_rgbt/missing_modality.py`: zero/learned/learned_film modes;
  two spatial embeddings and two modality-specific FiLM projections.
  Default learned_film adds 494,592 parameters, matching the detailed plan.
- `tests/test_c2_primitives.py`: 13 CPU checks passed locally with
  `E:/anaconda/python.exe -B -m pytest tests/test_c2_primitives.py -q -p no:cacheprovider`.
  Checks cover zero-init output, route gradient isolation and an actual
  optimizer update, merged/unmerged equality with/without bias, nonaliasing,
  per-modality FiLM math and the zero-init trainable path.

The authorized reviewer returned the separate source-only report
`P2_C2_PRIMITIVES_SOURCE_REVIEW_R1_20261102.json`, stage `P2_C2_primitives`,
SOURCE_PASS with zero blockers. Its exact three-source hash map matches.
The reviewer independently reran 13 tests and checked actual 768-to-2304
rank-eight merges on CPU float32/float64. Maximum float32 absolute error was
1.61e-6 in that fixture. Attribution is UNATTESTED/same-family/provisional;
scientific acceptance remains false.

The review reproduced two integration boundaries: zero inactive bank
gradients do not prevent AdamW decay/momentum updates to those bank slices;
constructing a fresh merged Linear consumes CPU RNG before copying weights.
The learned-only missing embedding is an expanded view, so scatter must be
functional rather than in-place. These are recorded requirements for the
future integration, not extra branches added to these primitives.

The original E0 source and running snapshots were unchanged. P2 still needs
full integration, optimizer coverage/reload/zero-mode equivalence checks,
new dataset/checkpoint/recipe admission and real training before any claim
about the three benchmarks. Existing P1B collection and six fixed P1A full
controls remain the current execution path; no TEST-based P2 selection.
