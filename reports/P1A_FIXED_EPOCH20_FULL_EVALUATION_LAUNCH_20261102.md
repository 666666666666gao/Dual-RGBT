# P1A Fixed Epoch20 Full Controls Launched

At 2026-11-03 03:38 UTC, both branches launched their three complete
exploratory TEST controls after actual successful TRAIN11 smokes.
These controls measure the fixed recipe-final P1A executor-only
checkpoint before P1B policy refresh. They are not P1B FIT results.
Do not use TEST outcomes to change the preplanned P1B recipe or choose
checkpoint, cost or branch. Report all six outcomes descriptively.

| Host / Branch | GPU | Benchmark | Wrapper | Actual Child |
| --- | ---: | --- | ---: | ---: |
| P25 DropMoE | 0 | LasHeR | 3440098 | 3440101 |
| P25 DropMoE | 1 | RGBT210 | 3440099 | 3440102 |
| P25 DropMoE | 2 | RGBT234 | 3440100 | 3440103 |
| P26 SOT | 0 | LasHeR | 3613576 | 3613580 |
| P26 SOT | 1 | RGBT210 | 3613577 | 3613579 |
| P26 SOT | 2 | RGBT234 | 3613578 | 3613581 |

Task IDs are `eval_p1a_e020_<benchmark>_<branch>_p<host>_r1_20261102`,
with lowercase lasher/rgbt210/rgbt234. Existing four collectors per host
continue on GPU0-3. At 03:39 UTC, all six new native states were RUNNING,
their wrappers/children existed and traces were 5.2-11.3 seconds old.
GPU contexts independently confirmed all fourteen actual workers.
P25 GPU0-2 used4538MiB each, GPU3 used2277MiB; all four power limits250W.
P26 used4559/4551/4551/2451MiB. No full accuracy scores exist yet.

## Admission And Smoke

Only24 previously absent source files per snapshot, the exact draft and
its review were copied. Host-specific results/contract payload bytes
were preserved. Existing98 collector sources and7 inputs revalidated
unchanged; complete49 evaluation maps matched after deployment.
Both accepted parent CPU-scoring environments imported the unchanged
toolkit/scorer and returned dataset sizes245/210/234.

Both TRAIN smokes completed native exit0:

- P25 `smoke_p1a_e020_dropmoe_p25_r1_20261102`, wrapper3434583,
  native23.951238s, one sequence/11 frames.
- P26 `smoke_p1a_e020_sot_p26_r1_20261102`, wrapper3609813,
  native29.526558s, one sequence/11 frames.

Both receipts certify zero backward/optimizer updates and bit-equal
model state. Configuration hashes, exact argv/GPU/checkpoint/source/
manifest bindings were checked before full launch. Each smoke loaded
103,374,215 total unique parameters. These short runs are admission
evidence, not benchmark metrics or throughput measurements.

Reviews, both SOURCE_PASS/zero blockers:

- DropMoE:
  `e235db468249c6eedef5ba7107bd9c59fdbd930d6e575a0662a8fff7276be195`
- SOT:
  `e44be6ea803715b6820246af8fbbbbd18c8cf6dd0afc6a947665eeaa2c379808`
- Local transport helper `work/operate_p1a_evaluation.py`:
  `975880b7db75b0c829a5cd59e1d66240362d42f87180332896d7aa8e1237c824`

Both reviews are engineering supplements honestly attributed
UNATTESTED/same-family/provisional; no mandated-model scientific audit
or scientific acceptance is claimed. During delegation the reviewer
twice drifted to obsolete R15 scope after compaction; explicit scope
corrections preceded the actual P1A artifacts. The obsolete R15 report
remained unchanged at SHAee17e7faebb9f72d98d6654bb3f06bdc1a3da8cc6f1dd25dc72c5f5084b1ac17.

## Evidence And Next Steps

Local exact host audits, disabled drafts,24-file payloads, deployment/
smoke/full receipts, raw terminal smoke native/log/config/receipt and
post-launch observations:
`reports/p1a_evaluation_r1_20261102/{dropmoe,sot}/`.
The two review reports specify exact CPU scorer commands and outputs.

Roots:

- P25 snapshot `/home/gaob/Dual-RGBT/snapshots/p1b_dropmoe_p25_r1_20261102`;
  runs `/data2/gaob/Dual-RGBT/p1a_eval_runs/<task>`.
- P26 snapshot `/data2/gb/Dual-RGBT/snapshots/p1b_direct_r3_20261102`;
  runs `/data2/gb/Dual-RGBT/p1a_eval_runs/<task>`.
- Native `<snapshot>/runs/pipeline_tasks/<task>/status.json`.

Score each full run once only after actual native COMPLETE/exit0 and
the full native/count/artifact checks: LasHeR245/220703 frames,
RGBT210210/104706, RGBT234234/116650. Use the respective accepted parent
`conda-envs/cpu-scoring/bin/python`, parent toolkit/protocol and snapshot
`tools/score_e0_finetuned.py`. Require score JSON/NPZ absent; preserve
existing scores. No partial scoring or duplicate scorer.

Both branches use cost0.05. Same checkpoint/config across their three
benchmarks; exploratory_test=true/test_eligible=false. Full-run FPS
includes initialization/decode under concurrent load, excludes model
loading; no isolated-speed claim.

Current CPU arithmetic evidence is
`reports/p1b_route_profile_r1_20261102/measurement.json`,
SHA9d63719a1d16d83310b7100939a64183dcf5c010c96b7d4522c1602651e8542a.
All25 source paths shared with either evaluation map match it exactly;
both real smoke parameter counts also match. Later use unchanged
`tools/summarize_e0_full_compute.py` only after deploying its exact
reviewed source and the profile, full scores and matching config/traces
are present. Its runtime source/parameter/route checks remain required.
No dataset-weighted MACs or full scores have yet been computed.

Per-branch all4 collectors COMPLETE0 still precede full882 acceptance,
a disabled B512/workers4/scaled-LR five-epoch FIT draft, separate exact
concrete review/admission and real training. Prefer GPU3 once all four
collectors of that branch finish; leave GPU0-2 evaluations running.
R3 capacity succeeded already: do not repeat it. Added inference
contention invalidates reliance on the earlier linear completion ETA.
Preserve best/latest5, active/audit dependencies, partial caches and
failed evidence; no blind restart/reset or occupied-card filler jobs.
