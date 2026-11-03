# P26 SOT Zero S1 Collection Start

Observed at 2026-11-03 15:58 UTC. This is a live TRAIN collection
observation, not a completed shard, policy FIT, or benchmark result.

- Task: `collect_c2_policy_sot_zero_s1_full_r1_p26_20261103`
- Admitted plan binding: `3b0e4eff482205bb650d981c293705a0be891a2b30e49a39bf11cb8d8ade6dab`
- GPU: 1, UUID `GPU-bc13d99e-845a-f985-262e-1918d2a39d6b`
- Native state: RUNNING; wrapper PID 1848330; actual collector PID 1848339
- The actual child was present in the GPU compute-process table with 2262 MiB.
- Progress: 5 sequences, 841 tracked frames, 166 records; current sequence
  `rightboywitjbag`; last sequence took 36.5497 seconds.
- Five partial sequence artifacts existed at observation time. Their hashes
  and the original remote bytes are retained in this local directory:
  `blkcarcome155.pth`:
  `00afe8f1f2050ee9a4a1a12e9f1ea728037b7a39fbe3279dcac5449e9c61f478`;
  `comingboy'shead.pth`:
  `c69f57eb8e00bee95d1201fa795a5d8faa36802a462bca653b3621980f847c83`;
  `farwhitecarturn.pth`:
  `1f3dc001347a65a14c8464afc73fe8cdd11ebdbd77b5c15b190f3e5b8ee223e5`;
  `motocross.pth`:
  `14026c554acd9e2b3de81135df1e58fccbb48db155a02f7b1d0aecb75ba3f615`;
  `rightboywitjbag.pth`:
  `c9230fdf3f7ad324baec9a27ce4031f658f58f1e40bf872f47520e8ee19f3ec6`.
  The binary artifacts remain local-only; their verified hashes and
  provenance are committed with this witness.
- P26 GPU1 was 62 C / 152.13 W at the progress snapshot; GPU3's existing
  P1B FIT was left untouched. P25 GPUs0-2 were active at 71/68/67 C and
  197.57/214.19/207.09 W (250 W limit); P25 GPU3 was idle but is not in
  the admitted plan GPU set.
- No COMPLETE receipt was present or claimed.

The launch passed the fresh canonical runtime admission on host 2026.
The source review is provisional same-family engineering admission with
`actual_model=UNATTESTED`; it is not an independent scientific review.
The original P25 and P26 tasks were not changed. TEST remains ineligible
for training, tuning, model selection, or final claims.
