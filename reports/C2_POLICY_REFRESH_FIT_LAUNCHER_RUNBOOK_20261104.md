# C2 Policy FIT Launcher

`work/launch_c2_policy_refresh_fit.py` closes the deployment step between
the disabled FIT preparer and the C2-specific trainer. It does not make a
RUNNING replay eligible for FIT. No genuine replay PASS, real FIT draft or
FIT launch is established by its CPU tests.

After a genuine matching replay PASS and native COMPLETE/exit0:

1. Archive the exact acceptance and native terminal bytes once. Use the
   reviewed preparer to create a disabled draft with a fresh task, absent
   independent FIT root, and the matching host/variant/executor.
2. Run `--preflight-only` with that draft and a new local witness path.
   This reads remote source/input/native/receipt evidence, authenticates the
   completed executor's CPU checkpoint and queries capacity. It exits before remote
   directory creation, deployment, plan enablement or process launch.
3. Obtain a separate exact FIT plan/source/input review. Its stage must be
   `P2_C2_policy_refresh_fit`, sources must equal the draft's inherited
   original-host map plus the FIT entry, and admitted_plan_bindings must
   contain the canonical draft binding. The report must also pin the exact
   launcher under operational_sources. Place its bytes at the local path
   corresponding to the draft's review path inside the new FIT root.
4. Launch with `--plan <local-draft> --witness <new-local-witness>`.
   Admission repeats the live checks. Target GPU must have at least
   20,480 MiB free, at most 999 MiB used, no compute PID, temperature below
   85 C and the exact power limit (P25 250 W, P26 350 W); destination disk
   must have at least 12 GiB free.
5. Inspect the actual native wrapper/child, config, precision and GPU
   ownership. Launch evidence is not an optimizer update or FIT completion.
   Observe long tasks at estimated boundaries, 180-300 seconds apart.

The launcher reads original collection source bytes on their own host and
adds only the reviewed FIT entry. It hashes every copied file and verifies
the completed executor inputs and four collection metadata chains.
Deployment uses an absent independent source root; original collection
snapshots stay intact. The deployed trainer rechecks the enabled plan and
acceptance before the native wrapper is spawned. It verifies all saved
artifact bytes when constructing its dataset.

The admission witness is `<run_root>/<task>_ADMISSION.json`. Native status
and log are inside the new FIT root at
`runs/pipeline_tasks/<task>/status.json` and `run.log`; FIT artifacts go to
the plan's independent run directory. Keep partial evidence if an actual
attempt fails; use a fresh reviewed task/root for any later attempt.

The fixed recipe remains five epochs, B512, workers4, preview LR4e-5,
quality/gain LR4e-4, AdamW weight decay1e-4 and clip1.0. This step does not
change recipes, select a checkpoint or establish tracking accuracy.
TEST/dev remain disabled.

Validation: 47 local CPU contracts pass across the new FIT launcher,
disabled preparer and existing replay launcher. CLI `--help` also passes.
These are fixture/source checks; no real remote FIT admission was run.

Source-only review:
`reports/P2_C2_POLICY_REFRESH_FIT_LAUNCHER_SOURCE_REVIEW_R1_20261104.json`
is SOURCE_PASS/0, SHA256
`1890d59fe40f390067f33e16a3894ee56e7438de3efa8b47926e749de0758f49`.
Attribution is UNATTESTED/same-family/provisional. It admits no plan,
runtime execution or scientific result. Its two recorded limits are that
artifact payload hashes are checked by the trainer before optimization,
and LAUNCHED only witnesses Popen until native execution is observed.
