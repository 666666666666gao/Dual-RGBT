# Global candidates result audit

**PASS — frozen training-state geometry only; 0 blocking issues.** Fresh native Codex review, same-family/provisional. Requested `gpt-6-astra` with `max`; actual model, effort and routing are **UNATTESTED**. No deployment, official tracking metric or efficiency acceptance is implied.

The auditor read source and artifacts directly and independently reconstructed the results with pure NumPy1.26.4. The stable-sort selector is separate from the production repeated-argmax implementation. No torch/project import, NN/GPU/SSH/process query, Git operation, install or production/plan/RESULTS edit occurred.

| Check | Verdict | Evidence and finding |
|---|---|---|
| GT provenance / type | PASS / `real_gt` | `scripts/prepare_data.py:7,17,24` reads LasHeR training `init.txt`; `scripts/diagnose_global_candidates.py:47,60,67–84` uses the hashed manifest. All1752 rows are from20 training sequences; train/dev/test names are disjoint. Initial GT defines the ordinary template; current-frame GT values are consumed after proposals. |
| Score normalization | PASS | `scripts/diagnose_global_candidates.py:39,84` uses cosine priorities and overlap divided by GT area. Reported coverage has no prediction-statistic denominator. The maximum is over individual ROI fractions, not ROI union or a score rescale. |
| Native/source/result existence | PASS | 181 checks bind source32/203542 bytes, plan, checkpoint identities, predecessor, M0, launch, native exit states, raw query stdout/stderr and artifact hashes. `scripts/run_task.py:28–32` records the child exit; the diagnostic rehashes source/inputs/protected weights before and after (`:112–124,157–164`). |
| Frozen M0 / formal | PASS | Actual two-state sanity is native exit0 with strict reload of scores/indices/points; formal follows it and binds its artifact hashes. Sanity candidates and complete score arrays equal the formal first two rows bitwise. All actual state entries are checked unchanged and all gradients absent (`scripts/diagnose_global_candidates.py:134–155`); receipts report265 tensors, zero backward/optimizer updates. |
| Greedy proposals / Top1 | PASS | All1752 lists and13940 candidates reconstructed exactly from full float32 score maps. All padding is masked, valid scores finite and candidates valid. Every Top1 matches the fitted prior. Maximum point error0. |
| Float32 coverage | PASS | All14016 formal prefix/ROI membership, best-coverage and oracle-index combinations reproduce exactly; maximum coverage error0. Formal and sanity aggregates also match. `scripts/diagnose_global_candidates.py:79–102`. |
| Random control | PASS | All10 post-hoc seeds2027–2036,17520 random lists,70080 equal-prefix pairs and280320 geometric outcomes reproduce every per-seed aggregate and printed mean. The common prefix is `min(K,CNN_count,random_count)` (`tools/compare_global_candidates_random.py:55–99`).329 pairs use fewer than K;34 random lists are shorter than CNN. |
| Dead code / claimed outputs | PASS | Diagnostic, analyzer and control functions lie on invoked main paths; native CPU transports exit0, stdout matches artifacts, and independent arithmetic verifies outputs. `scripts/diagnose_global_candidates.py:68,144`; both tools' main guards. |
| Scope / claims | PASS | Both tables, candidate exhaustion counts, completion time and55/239 remaining wide misses match. `docs/GLOBAL_CANDIDATES_RESULTS.md:14,18,27` limits claims to fixed training-state single-ROI oracle opportunity and count-matched random control. |

The failed original collector is preserved as **actual exit1**, including its native tool-result object and SSH stderr/transport. It requested nonexistent `comparisons.json`. The corrected remote collector changes only that name to `candidates.json`, embeds the identical launch/task/PID/command, and performs read-only collection. The same NN task completed **exit0**; resumed collection exited0. This was not an NN rerun. The publication-preparation SyntaxError is also retained as an explicitly labeled executor note, not a fabricated transport receipt.

| K | All1752 local / wide | Outside239 local / wide | Random mean all local / wide | Random mean outside local / wide |
|---:|---:|---:|---:|---:|
|1|348 /720|25 /55|148.2 /482.8|7.6 /27.1|
|2|571 /1055|36 /80|271.1 /738.3|12.7 /45.9|
|4|841 /1346|63 /130|470.3 /1014.1|24.0 /72.8|
|8|1171 /1526|113 /184|740.6 /1295.7|45.5 /116.3|

These are centre-coverage counts; the random columns average the10 seeds. The machine report preserves the independently verified GT-area aggregates. Candidate-count matching is **not matched executed auxiliary cost**. The fitted preview was trained on882 sequences, but this diagnostic evaluates1752 old visited states from20 sequences. The239 main-outside labels remain native-derived and were matched to the CSV/prior, not independently recovered from saved main geometry.

The supplied snapshot supports geometric action-set opportunity and this post-hoc random comparison. It does not establish a GT-free deployable selector, useful executed auxiliary pixels, new trajectories, localization gains, generalization, PR/NPR/SR, latency, final C or three effective core controls. Remote checkpoint tensors and original remote `init.txt` bytes were not reloaded; their identities are bound by the archived native chain and reviewed source assertions. The manifest is loaded before proposals, although current-frame annotation values enter only the later diagnosis.

All **115 unique audited inputs** remained unchanged, including all68 supplied files, source32 and added provenance dependencies. Final input-manifest SHA256: `6b71224eb37df2b5feac75f266b6acf2d741d2c0ac99ad8ffae7d7629dbf74ea`. The initial/final manifests, two passing checker receipts, full request/response trace and observed environment/discovery errors are retained privately under `C:/Users/gb/.codex_tmp/dual_rgbt_global_candidates_result_audit_20261014/`. Evidence checks181/181; geometry checks124423/124423. No checker failure was hidden; the initial broken Python, missing NumPy in uv3.13, and an absent guessed optional sanity-launch directory are separately recorded.

Exact key identities:

- Plan: `34eac262416ec39a8196ce0b64e8e20ef794af05df66f5978af88ad2c403a756`.
- Fitted checkpoint: `ad0599434e4b0024f206a108e6813e7c00127ecec829339832d00526890aef85`.
- Formal completion: `9e7bd7dccf63cb5df8596bd74f7a609a5f0a42a3fed8047bf934a01650e3c657`.
- Private candidates: `47e6e835e3cac7843d223e8b6bcf92293c8b930a9fa94d6dc598cb78cb979a59`.
- Private scores: `1235b97c6ebe71c6488d5e72be4ee52c18c4e7242ea4c902e4b2b47998971fc3`.
- Audited RESULTS snapshot: `e003f75f690439c9f90091be0eca30923bd4d2ffeb5f89496f1dd0706f6a5388`.

The RESULTS snapshot said its audit was pending at writing. A later status-only update may cite this report while retaining this exact pre-update snapshot hash and all tables/scope. Semantic acceptance remains same-family/provisional; deterministic equality does not imply cross-family acceptance.
