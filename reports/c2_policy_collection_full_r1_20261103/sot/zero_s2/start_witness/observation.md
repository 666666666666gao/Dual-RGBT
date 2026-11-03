# P26 SOT Zero S2 Collection Start

Observed at 2026-11-03 16:17 UTC. This is a live TRAIN collection
observation, not a completed shard, policy FIT, or benchmark result.

- Task: `collect_c2_policy_sot_zero_s2_full_r1_p26_20261103`
- Admitted plan binding: `aa3a772e02aa007d73add2bbfc4f39635e8c3f47791eabf42d4948622fc8fa49`
- GPU: 2, UUID `GPU-442835f1-3e38-07d1-c823-57f7b12ad5bd`
- Native state: RUNNING; wrapper PID 1886538; actual collector PID 1886539
- The actual child was present in the GPU compute-process table with 2262 MiB.
- At 16:17 UTC: 6 sequences, 1,419 tracked frames, 282 records; current
  sequence `outerfoam`; last sequence took 5.8179 seconds.
- Six partial sequence artifacts were present. Their original remote bytes
  remain in this local directory; verified SHA256 values:
  `blackpantsman.pth`:
  `901d4a01066345696e503e176bfbbf83d6976b0d8232a5c967e7dd8d2c2109c0`;
  `carturn.pth`:
  `75dec17621010c98fd3f2d2cb3ecc9dbaf68bf7b5749c9388749212a4ffd3f23`;
  `comebike.pth`:
  `0f3b93e25ca3d1d48f05b3f6d06f22b836a627e92e7f3af2f8e70ed7167846fb`;
  `nightboy.pth`:
  `620160c0b35030716095112c3e6e6b248d7ef7787084f1d2a84c841cc909bf96`;
  `outerfoam.pth`:
  `b28c87cac9b563ac8073b4112322ba7237f242b94ce636457659067a22338302`;
  `wanderingly_1202.pth`:
  `6544e9be02173cda176ef4c8bb8de6335bfa2caff5db9b5639cabf3d86053889`.
  Binary artifacts remain local-only; verified hashes and provenance are
  committed with this witness.
- P26 GPU2 was 80 C / 193.82 W; its existing GPU0/1 collection tasks and
  GPU3 P1B FIT were left untouched. P25 GPUs0-2 were active at 69/67/64 C
  and 222.25/181.05/180.54 W under the 250 W limit; GPU3 was idle but is
  outside the admitted plan GPU set.
- No COMPLETE receipt was present or claimed.

The launch passed fresh canonical runtime admission on host 2026.
The exact source review remains provisional same-family engineering
admission with `actual_model=UNATTESTED`, not independent scientific review.

The native run log emitted PyTorch `__floordiv__` deprecation warnings at
`lib/models/layers/head.py:143` and `dual_rgbt/model.py:110,113,118,127`.
No exception or task failure was observed. Do not mutate this active source
snapshot; review operand ranges and focused tests before any future change.
