# C2 Policy Refresh Loader R1

## Scope

`dual_rgbt/c2_policy_refresh.py` is a new local-only primitive for the
post-training C2 boundary. It does not replace either host's original P1B
loader, deploy files, launch a collector or admit a GPU run. The existing
eight tasks continue unchanged. No full C2 executor has finished yet.

The old P1B loader expects P1A config/argv and constructs `DualRGBT`.
The new loader validates C2 completion and constructs `C2DualRGBT` with the
serialized missing-modality/LoRA configuration, then loads its state strictly.
This preserves learned FiLM and later shared/routed adapter state instead of
discarding additional keys or reconstructing the original architecture.

## Exact Binding

The executor binding has stage `P2_C2_policy_refresh`, host, `c2_task_id`,
variant and an `inputs` map containing exactly eight path/SHA256 pairs:
factory, manifest, executor, c2_plan, c2_config, c2_native, c2_complete and
c2_precision. This is an executor binding, not an independently admitted
collection or FIT plan.

The verifier requires:

- Exact input hashes, equal original full plan/config, correct host/task/
  variant, full20-epoch non-smoke configuration and TRAIN-only flags.
- The original full-training review hash, source map and admitted recipe
  binding, plus unchanged training sources in the recorded snapshot.
- Native COMPLETE/exit0 with the exact command, physical GPU and `--plan`.
- Actual completion fields:20epochs/50000updates, ten routes5000each,
  preserved policy, mandatory later refresh and no benchmark claim.
- Exactly the five retained checkpoints16-20, their current hashes, and the
  fixed epoch20 executor in its recorded run directory.
- Saved epoch20/step50000, matching training config, exact C2 architecture
  metadata and finite serialized CPU state before strict reconstruction.

`completed_executor_state` only reads CPU artifacts. The reconstruction
function requires an explicit `device` argument; no default CUDA activation.
Full source review and a separately bound future entry are still required
before any deployment or GPU action.

## Validation

Actual command:

```powershell
E:/anaconda/python.exe -B -m pytest -q tests/test_c2_policy_refresh.py tests/test_c2_training_admission.py tests/test_c2_primitives.py
```

Result at07:26UTC:110passed in4.48s. Tests exercise all four variant metadata
contracts, incomplete/failed native state, wrong GPU, incomplete routes,
missing or changed retained weights, changed sources, precision mismatch,
unbound training review, and rehashed checkpoints with incorrect architecture,
config, step or nonfinite state.

These are synthetic CPU artifact-contract tests, not actual C2 model
reconstruction, a completed TRAIN acceptance, collection, policy FIT or
benchmark evidence. The real full-training tasks have not completed and
their evidence must not be fabricated from smoke checkpoints.

Exact source SHA256:

- Loader:8a8d17d08d25089c768a697e0ce521b03d99a9add0390a458c2e46f4204205d0
- Tests:a477fe42ec271faf10df24a040604a231cf9c7a6790e8b92c384d67b0c544aa2

Supplemental source-only review requested from the already authorized review
chat01a18f7a-b049-78f1-b7ef-52764c88a6b5. Expected report:
`reports/P2_C2_POLICY_REFRESH_LOADER_SOURCE_REVIEW_R1_20261103.json`,
stage `P2_C2_policy_refresh_loader`, exact two-file source map and no
admitted plan bindings. Actual source review is SOURCE_PASS/0, with the exact
two-file map verified. Report SHA256:
`f59bf3c7ccc44f0db798ea1885972816bbfb45cf050825fab7658f132766f73b`.
The reviewer independently reran the110 CPU tests in4.83s. Attribution is
UNATTESTED/same-family/provisional, not an Astra or independent-family result.
The report has no admitted plan bindings and explicitly grants no deployment,
collection, policy FIT or GPU admission. Actual C2 reconstruction remains
pending the real full-training artifacts.
