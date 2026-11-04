# C2 Evaluation Operator

`work/operate_c2_policy_evaluation.py` passed exact operation review R1 and
was invoked for isolated deployment and TRAIN smoke on November4.
Operator plus preparer tests:32 passed. The
tests cover exact review/source/canonical plan binding, preservation of
disabled drafts, enabled request construction and remote payload
compilation/early rejection. They do not simulate genuine CPU/GPU work.

The preceding exact disabled-plan review R1 is SOURCE_PASS/0 but has
empty admitted/execution bindings. It cannot satisfy this operator.
Separate operation review R1 is SOURCE_PASS/0 and authenticates
exact117sources, both fixed plan bindings and the operator hash in
operational_sources. Its SHA256 is
`40ffab8f8b0083e74035738da5d97f790ab74645040e5eb308f99ecc67ed8ef4`.
Attribution remains UNATTESTED/same-family/provisional. Static review is
not a runtime capacity witness.

Deployment creates only the new evaluation root. Original112host-specific
sources come from the completed isolated FIT root; five new C2 evaluation
sources come from their reviewed local bytes. Every source and input pin
is rechecked. The original/active roots and disabled local plans remain
unchanged. A failed or partial new root is preserved and cannot be reused.

Deployment runs actual load_completed_policy on CPU, authenticating the
original C2 architecture, retained FIT checkpoints and tensor constraints.
Only an actual success writes a source/input/review-bound CPU witness
with CUDA uninitialized. Actual deployment returned
C2_POLICY_CPU_RECONSTRUCTION_PASS with103374215 parameters and CUDA
uninitialized; the source/input/review-bound witness and raw result are
preserved in the isolated root and local deploy_result_r1.json. This
includes actual tensor and strict reconstruction checks, beyond the
earlier metadata-only FIT archive.

Fresh GPU0 admission passed at18:12:03UTC,250W limit/40C/12MiB used.
TRAIN11 smoke wrapper/child2503559/2503560 started with exact argv and
its own CUDA process. Do not infer completion from launch; inspect
native COMPLETE/exit0 and SMOKE.json before full admission. Actual smoke
completed18:12:43UTC with unchanged state and zero backward/optimizer
updates. Full's independent terminal and fresh capacity gates passed;
LasHeR task started18:16:10UTC,wrapper2511436/child2511437,GPU0.
Its initial native state is RUNNING, not benchmark completion.

Smoke requires that CPU witness. Full requires its own TRAIN11 native
COMPLETE/exit0, exact plan/config/receipt/precision/source/checkpoint
bindings, unchanged state, no updates/backward and exited wrapper. Both
launch paths recheck source/input hashes, unused task/run IDs, free GPU
memory, no live CUDA owner, temperature below85C, correct host power
limit and disk capacity, then record admission and use the native wrapper.
Only the full LasHeR plan is selected by the full action.

Future invocation, only with an actual accepted operation review:

```text
E:/anaconda/python.exe -B work/operate_c2_policy_evaluation.py <deploy|smoke|full|inspect> --plans reports/c2_policy_evaluation_r1_20261104/dropmoe/zero --review <exact-operation-review.json> --output <new-action-result.json>
```

Do not execute all actions at once. Verify actual CPU reconstruction,
then native smoke completion and own GPU child before full admission.
Do not use old P1B factories, change epoch005/cost0.05 or tune on TEST.
All full TEST is exploratory/test-ineligible; no benchmark improvement
or scientific acceptance follows from these engineering checks.
