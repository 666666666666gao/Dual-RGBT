# Additional direct SSH provider25

User authorization: 2026-10-27, all physical GPUs0/1/2/3.
Project and conda: `/home/gaob/Dual-RGBT`, `conda-envs/dual_rgbt`.
Four RTX3090 cards were empty, with no compute processes, before setup.
Host RAM: 263760256KiB; available 254097712KiB at preparation.

Warm restore: original conda copied from the retired project archive back to
its original prefix. No package installation or environment upgrade.
Python3.7.16 / Torch1.12.1+cu113 / timm0.9.12.
Parent seeded kernel/update/reload and real eleven-frame TRAIN tracking passed;
independent verbatim document verification also passed (nativeexit0, 17.12seconds). Environment is accepted; new experiments still need their own source and real-update admission.

Evidence: `reports/SERVER25_REJOIN_ENVIRONMENT_PLAN_20261027.json`,
`SERVER25_REJOIN_RUNTIME_COPY_20261027.json`,
`SERVER25_ENVIRONMENT_PRIMARY_NATIVE_20261027.json`,
`SERVER25_REJOIN_DATASET_INVENTORY_20261027.json`.
Documented invocation: `docs/SERVER25_REJOIN_20261027.md`.

This provider does not change the accepted server26 prefix, source, weights,
or jobs. The historical archive and its old queues remain retired. New plans
must use new run/task identities, the correct25 manifest, unchanged SOTep14
initialization, and a fresh real-update/reload admission before full training.

Acceptance: reports/SERVER25_ENVIRONMENT_ACCEPTANCE_20261027.json. Fresh document run is same-family/provisional with actualmodel UNATTESTED. SmokeFPS is not formal benchmark performance.

## Separate CPU scoring environment

Spec: cpu-scoring-env.json; canonical spec SHA256 5e574a086db1d396a0be0f1852abc0d8a77809bf882f567ff5684a5948e9ea14.
Prefix: /home/gaob/Dual-RGBT/conda-envs/cpu-scoring.
Builder: tools/prepare_cpu_scoring_env.py; current create run cpu_scoring_build_r3_20261027; native export/witness receipt CPU_SCORING_ENV_POST_INSTALL_COMPLETION_20261027.json.
Status: PARENT_CPU_WITNESS_PASS_DOCUMENT_VERIFICATION_PENDING.
Python3.10.14, NumPy1.26.4, Matplotlib3.8.4; no Torch/GPU use.
The original NN prefix/spec remains unchanged. Actual inherited-toolkit
Python3.7 list[str] import failure is preserved; this CPU environment
resolves that observed runtime requirement without editing the toolkit.
Witness/document: docs/CPU_SCORING_ENV_20261027.md.
