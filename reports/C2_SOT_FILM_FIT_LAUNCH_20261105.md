# P26 SOT-FiLM FIT Launch

The exact P26 SOT-FiLM FIT plan/source/input review passed with zero
blockers. Review SHA256:
`bb6d0df237fbbb54e8487c810fdf527f9f00b919f2036af4bf2a28103f4dd50c`.
Attribution remains UNATTESTED/same-family/provisional; scientific
acceptance is false.

The disabled draft SHA256 is
`614e7cba3b7632f63037c7181c8c1edb414c7cc984370d45a6ddf93ee7b5eb87`;
its canonical binding is
`70305f1d0ff84a43e1ad8a49a3aefac654ac462e5c1d1c9cf21da0d4c3a47d9f`.
The fixed recipe is five epochs, batch512, workers4, seed2027, preview
LR4e-5, quality/gain LR4e-4 and cost0.05. Dev and TEST remain disabled;
retain all five epoch weights.

Read-only canonical preflight passed at 2026-11-05 03:48:38 UTC. It
admitted GPU2 with 24,112 MiB free, 15 MiB used, 41 C and a 350 W limit;
disk free was 6,810,257,129,472 bytes. The separate launcher repeated
admission and returned LAUNCHED at 03:54:38 UTC. The enabled plan SHA256
is `284b82a6ef4e19590592896384563b73c61b554cb44d430ab1e930e36869f5d7`;
wrapper PID86037.

At 04:05:09 UTC the native wrapper and child86054 were RUNNING with exact
child argv and CUDA ownership matching GPU2 UUID
`GPU-442835f1-3e38-07d1-c823-57f7b12ad5bd`. The child held 19,424 MiB;
GPU2 was 70 C/135.36 W at a 350 W limit. Active config matched the plan
and all three precision flags were false. `steps.csv` contained zero
flushed updates; no epoch, checkpoint, completion receipt or benchmark
result is claimed. The one-time start evidence is archived in
`reports/c2_policy_fit_r1_20261104/sot/film/start_witness/`; all eight
raw-file hashes in `SHA256SUMS` verify.

The separate P26 SOT-zero FIT remains active on GPU3. Both runs use their
reviewed fixed recipes; this launch does not alter either task. No dev/TEST
tuning, checkpoint selection, scoring or RGBT launch is authorized by this
engineering gate. Benchmark targets remain unmet.
