# Revision2 signed-gain/difference review

**Revision2 preliminary code/CPU checks pass; 0 concrete source blockers. No formal-plan or runtime approval.** Requested reviewer: gpt-6-astra/max. Actual route: UNATTESTED. Same-family/provisional.

Source SHA: `6cede0c60dbba2cb76483ffef0e0dd31571838b6bd63fb3cb8e122d538f8357b`. Exact-byte snapshot: `candidate_source.py`. The previous utility-hinge candidate and preliminary findings are preserved unchanged; that candidate is not approved for launch.

The new gain objective is the mean half-MSE over the same two sampled gains and optional selected-extra gain, plus half-MSE of the selected-extra minus same-role sampled gain difference (trainer lines94-111). All query utility/fixed-margin hinges and target-dependent weighting are removed. Quality regression/ranking remains unchanged. Existing inference still compares signed predicted gain minus token cost against STOP0.

Both predictions in the pair retain their gradient paths. For residual vector r=p-g and a vector a with+1 at selected-extra and-1 at its sampled role, raw gradient is r/n and pair gradient is a(a^T r). The other modality receives no pair gradient. Ties target a zero difference. No-extra yields zero pair loss and the original two gain predictions. Raw regression anchors absolute gain: adding a common offset can leave the pair difference correct, but incurs positive raw loss. The target difference cancels the shared main IoU algebraically; there is no claim of bitwise equivalence to a separately rounded query-IoU subtraction.

**Conditional objective identity.** Fix a complete common state, observed candidate set and selection event. Let G be their joint random gain labels, mu=E[G], and allow the prediction vector p to vary freely. Require finite second moments and positive raw-regression weights independent of the realized labels under that conditioning. For an extra pair, n=3 and

`E[L(p,G)] - E[L(mu,G)] = ||p-mu||^2 / (2n) + (a^T(p-mu))^2 / 2`.

Its Hessian is I/3+aa^T, with eigenvalues1/3,1/3,7/3. Thus the scored conditional-mean vector is the unique unrestricted minimizer. For no-extra, the two scored predictions have Hessian I/2; the absent extra prediction is not scored. Label correlation changes the constant risk term, not this minimizer. The common-main-IoU dependence is allowed.

The prior scalar Bernoulli(.25) counterexample now has gradient0 at.25; the old hinge optimum.2958333 has gradient+.0458333 and higher squared-error risk. Twelve finite-support correlated joint-label cases verify the mean gradients, Hessians and60 quadratic risk decompositions.

This identity is not a finite-network or population calibration guarantee. Actual query inputs may expose different information, weights are shared with other predictions and the quality objective, optimization is nonconvex, and online state/selection distributions change. The proof fixes the selection event and assumes freely specifiable common-state predictions. No measured calibration, accuracy, causality or novelty improvement follows from these fixtures.

All49 inherited source hashes remain exact. Five helpers are AST-identical to the previously reviewed quality trainer: trajectory_counts, causal_batch, selected_target, save_checkpoint and write_label. The loop is identical apart from new logging; main differs only in the objective description. Existing quality/CNN paths,251-frozen/14-changed checks, optimizer groups, causal GT timing, teacher state preservation, strict reload/probes and original-B formal reset remain unchanged. No teacher/model forward call or label is added.

Validation:32 pair-gradient combinations;32 exact-target cases;32 quality value/gradient comparisons;10 tie/negative/zero/common-offset cases;8 no-extra cases;one scalar random-label case;12 joint random-label cases and60 decomposition checks. Torch1.12.1+cu113 remained CPU-only with CUDA uninitialized. No trainer import, neural execution, active training/dev/resource query, remote write, experiment-source edit, previous-record edit or job operation occurred.

This is preliminary only. The future plan and completed evaluation gate require separate binding before any runtime approval. Files: `preliminary_findings.json`, `cpu_verification.json`, `remote_cpu_stdout.json`, `verify_cpu.py`, `remote_cpu_fixture.py`.
