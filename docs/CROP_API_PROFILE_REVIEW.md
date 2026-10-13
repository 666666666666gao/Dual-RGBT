# Crop API CPU profile — independent source review

2026-10-13. **SOURCE_ONLY correctness: PASS; overall: WARN; blocking findings: 0.**
This review supports one CPU execution of the exact reviewed diagnostic. New
profile execution and its tensor/RNG/timing results are **NOT_RUN** at review time.

```yaml
reviewer: /root/review_crop_api_profile
review_scope: SOURCE_ONLY
review_independence: same-family
acceptance_status: provisional
policy_requested_model: gpt-6-astra
policy_requested_reasoning_effort: max
actual_model_route: UNATTESTED
actual_reasoning_effort: UNATTESTED
source_correctness: PASS
overall_verdict: WARN
blocking_findings: 0
fail_findings: []
new_native_profile: NOT_RUN
new_native_tensor_target_geometry_parity: NOT_RUN
new_native_rng_parity: NOT_RUN
new_native_timing: NOT_RUN
production_integration: NOT_PERFORMED
```

The independent reviewer read the artifacts directly in a separate context.
The model and effort above are the local policy request, not an independently
attested backend identity. This is not cross-family acceptance. Applicable
instructions were the project `AGENTS.md` and
`C:/Users/gb/.codex/skills/shared-references/local-codex-policy.md`.

The review performed local source reads, SHA256 checks, and parsing of the
already completed interior profile JSON. It did not import or execute the new
entry point, execute an NN, access SSH/GPU, query a live process, inspect pending
dev outputs, or change a source file. Its only write is this review document.
The supplied first-dev observation node remains 21:23 BJT; no observation was
made here. Full verification of the active eleven-file seal belongs to the main
task; this review does not claim to have independently rechecked all eleven.

| Source check | Verdict | Evidence and implication |
|---|---|---|
| Reviewed identity | PASS | `scripts/profile_crop_api.py` SHA256 is exactly `62e445f247a83571d0a318552b6767b142338b3d177dca65478b0e7b25860a9c`. |
| Local import chain | PASS | Lines 19–20 import only `dual_rgbt.data` and the existing interior diagnostic. `dual_rgbt/__init__.py` contains only a docstring. Interior imports contiguous; both diagnostics place `main()` behind `if __name__ == '__main__'`. Data imports `Dataset` but neither constructs a model nor runs its dataset class in this diagnostic. No local model, checkpoint, training, tracker, evaluation, subprocess, or CUDA invocation is on this call path. |
| CPU constraint | PASS | `main()` requires an explicitly empty `CUDA_VISIBLE_DEVICES` at line 33. The visible work is image decoding, NumPy/OpenCV geometry and interpolation, CPU `torch.from_numpy`, equality, and CPU thread reporting. The source-level assessment is not a new runtime attestation of installed libraries. |
| Six-channel API | PASS | Lines 23–25 preserve `(image, box, factor, size, rng=None)`, copy channels `0:3` and `3:6` to contiguous arrays inside the wrapper, delegate to the existing kernel, and return its first three tuple members. This preserves the tensor, normalized target, and crop geometry contract of `data.crop_pair`; the classifier boolean is excluded from the API return. |
| Crop and RNG semantics | PASS | The existing kernel uses the same centre, side, normal then uniform draws, float32 affine, normalization constants, and target/geometry construction as `crop_pair`. Neither path mutates the input image, box, or shared global RNG. The supplied per-call `RandomState` is advanced in the same order. |
| Equality gate | PASS | Lines 45–57 create matched seeded RNGs and check the tensor with `torch.equal`, target/geometry with `np.array_equal`. For search cases, all five `RandomState.get_state()` components are compared: algorithm name, key array, position, cached-Gaussian flag, and cached value. Template calls pass `None`, so their RNG check is vacuous. Assertions precede record publication. |
| Case selection | PASS | The same first ten manifest dev sequences and first/middle valid frames used by the completed interior diagnostic are selected. One 128 template and three separately seeded 256 search jitters yield 80 planned cases, comprising 20 template and 60 search cases. The script reports actual case/sequence counts; 80 remains a prospective count for this new execution. No test split or tracking-result selection is read. |
| Balanced repeated timing | PASS | Lines 63–69 use four repeats and offset `(identity + repeat) % 2`. Each arm is first twice and second twice within every case. Each gets four observations per case, hence 320 per arm / 640 timed calls for 80 cases. Each repeat resets to the same case seed. |
| Timing boundaries | PASS | The clock starts before function-argument evaluation, so search RNG construction is included. The timed candidate calls the wrapper itself, including both full-image contiguous copies, the interior guard, conditional means, warp, normalization, tensor construction, and tuple slicing. Both arms use the same decoded six-channel input. Decoding, box conversion, equality assertions, summary generation, and the separate classification call at lines 58–60 are outside the interval. |
| Existing boundary guard | PASS | The kernel hash matches the completed native artifact. It inverts the same float32 affine promoted to float64, maps the four output corners, and skips border means only when every source coordinate lies within `[1, width-2] × [1, height-2]`. Otherwise it retains each modality's full-image mean and `BORDER_CONSTANT` / `INTER_LINEAR`. The wrapper does not alter this rule. |
| Result wording | PASS | Report scope explicitly restricts results to paired CPU crop API calls. It makes no production-change, whole-system latency, tracking accuracy, or module-effectiveness claim. New speedup and new equality results cannot be asserted before the one actual execution completes. |

The boundary branch has observed motivation: the completed interior report
contains 47 mean-skipped and 33 mean-retained cases, and the stable explanation
records real out-of-bounds crops. A guard-negative case can also be merely too
close to the boundary for the conservative test; the 33 count should be called
“mean retained,” not automatically “33 physically out-of-bounds crops.” The
source review retains the existing guard and does not propose additional
fallbacks, exception handlers, compatibility branches, or production changes.

The completed evidence read here is `docs/INTERIOR_CPU_PROFILE.md`,
`reports/interior_crop_cpu_profile.json`, and
`reports/interior_crop_cpu_native.json`. The last name is the actual native
artifact named in the stable document; there is no
`interior_crop_cpu_profile.native.json` at that path. The native receipt records
exit code 0, CPU-only execution, and source SHA256
`89abe94cbd0c6504bb0528dca4842d2c5350b9c6f021b243068493e9c823dc6e`, equal to the
currently read interior script. Its embedded report equals the standalone
report after JSON parsing. Both contain 80 records / 10 distinct sequences,
all stored equality flags true, and three timings per arm per record. The
47/33 split consists of 16/4 template and 31/29 search cases. These are completed
old-kernel observations; they do not substitute for the new wrapper run.

Nonblocking WARN findings:

- **New runtime remains unmeasured.** The new script has not been imported,
  compiled, launched, or tested by this reviewer. Tensor/target/geometry and RNG
  assertions need to pass in the actual CPU invocation with assertions enabled,
  an empty `CUDA_VISIBLE_DEVICES`, the reviewed files, and a fresh report path.
  Record the real exit and source identity in the native receipt. A failed
  assertion is an experiment failure to retain, not a result to overwrite.
- **The statistic is a warm, repeated component microbenchmark.** Validation and
  one extra classification call precede the timings; decoded images are reused.
  The four repeats do not create 320 independent image cases. Shared CPU load,
  caches, and library thread settings can affect the observed mean/P95.
  Classification output is only a guard stratum and has no accuracy meaning.
- **Bounded parity is not production acceptance.** The old 80-case bitwise
  result supports the unchanged kernel on that sample. The new run additionally
  checks the wrapper and complete local RNG state. Neither establishes every
  future trajectory's equivalence, end-to-end service latency, PR/NPR/SR, or
  independent module benefit. In particular, the prior 61.649% crop-only mean
  reduction excluded contiguous-array preparation and cannot be reused as this
  new API's measured reduction.
- **Reviewer provenance is provisional.** Same-family independent context is
  documented; actual model route and reasoning effort remain UNATTESTED.

No blocking FAIL was found in the reviewed source. No code revision is requested
for this bounded CPU diagnostic. Its next evidence is the single actual CPU
run and retained native/report artifacts; NN or production admission remains
outside this source review.

Read artifact SHA256 pins:

| Artifact | SHA256 |
|---|---|
| `scripts/profile_crop_api.py` | `62e445f247a83571d0a318552b6767b142338b3d177dca65478b0e7b25860a9c` |
| `dual_rgbt/__init__.py` | `83428a55c2c6507402130ee1ede4cba6454eaf9104f71be8bb70d50b4bc99c7e` |
| `dual_rgbt/data.py` | `858e41c8e5218c7f5323a6e18cec00997a48d0323101384b71a73d61ea0280cb` |
| `scripts/profile_interior_crops.py` | `89abe94cbd0c6504bb0528dca4842d2c5350b9c6f021b243068493e9c823dc6e` |
| `scripts/profile_contiguous_crops.py` | `cbe7c7d6a7115d4ee383bec5ddac84b0de8179d4f2f085b628b8523e421e700f` |
| `docs/INTERIOR_CPU_PROFILE.md` | `7843b776d6e6dd3e1d2730d7283f709a562f9a10837a6414b113d5a717b9c030` |
| `reports/interior_crop_cpu_profile.json` | `a4d5b45730667c9aea5eb144d6269239b824091a9278214a3e1ba0d0c3d634ee` |
| `reports/interior_crop_cpu_native.json` | `75e9d93ff8c26e5ea412465f62491e84f1ed822e49ce93ac40c1aa1a3f93096d` |
