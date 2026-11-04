# Fixed P1B Epoch005 RGBT Evaluations

These four separate plans extend the fixed recipe-final epoch005 evaluation
to full RGBT210 (210 sequences / 104706 frames) and RGBT234
(234 sequences / 116650 frames). Cost remains 0.05. No TEST selection,
training, threshold changes or independent validation claim is allowed.

The existing LasHeR jobs continue on each GPU0. Existing C2 replay tasks
continue on P25 GPU1 and P26 GPUs2/3. Do not change their sources or tasks.

| Branch | Benchmark | GPU | Scheduling |
| --- | --- | --- | --- |
| DropMoE / P25 | RGBT210 | 2 | Separate exact review and fresh runtime admission |
| DropMoE / P25 | RGBT234 | 3 | Separate exact review and fresh runtime admission |
| SOT / P26 | RGBT210 | 1 | Separate exact review and fresh runtime admission |
| SOT / P26 | RGBT234 | 1 | R2 predecessor gate; wait for SOT210 COMPLETE/0 |

The first three disabled drafts are under
`reports/p1b_rgbt_evaluation_r1_20261104/<branch>/<benchmark>/draft.json`.
Do not regenerate existing plans. Each has an independent root, task and
run path, the exact 58 original-host source map, immutable original
factory and completed FIT lineage. Never copy P25 core sources onto P26.

Use `work/operate_p1b_rgbt_evaluation.py <branch> <benchmark> <action>`.
The operator imports the unchanged reviewed evaluation runtime. Deployment
requires an exact SOURCE_PASS/0 report for that draft and all five local
operational sources. It rehashes remote inputs and sources and performs
an actual CPU factory load. Smoke requires a fresh empty GPU and uses
11 TRAIN frames. Full requires its own native COMPLETE/0 smoke, unchanged
state, zero optimizer/backward calls and all-false precision. It repeats
the empty-GPU, power, temperature and disk gate before launch.

SOT234 R1 lacks an enforced predecessor-terminal gate and must never deploy.
Preserve its failed review. The separate R2 draft is under
`reports/p1b_rgbt_evaluation_r2_20261104/sot/rgbt234/draft.json`.
Use only `work/operate_p1b_sot_rgbt234_followup.py <action>` for R2.
Before deploy/smoke/full this authenticates pinned SOT210 draft/review bytes,
native COMPLETE/0, exact command/task/GPU/config/full receipt and exited
wrapper. The unchanged base runtime then applies its own source/input,
review, smoke and fresh empty-GPU/power/temperature/disk checks.
SOT234 must wait; do not call its launch actions while SOT210 is active.

P25 must stay at 250 W / below85 C; P26 at 350 W / below85 C.
Record actual native children and GPU UUID ownership after launch.
Use 180-300 second checks at predicted boundaries. A quiet log or observer
timeout is not failure. Preserve failed/partial evidence and never retry
an existing task ID or completed archive.

Archive each smoke and full-start witness once via
`work/archive_p1b_rgbt_evaluation_witness.py <branch> <benchmark> <kind>`,
where kind is `smoke` or `full_start`. It imports the existing read-only
native/argv/receipt/GPU checks and preserves raw bytes with hashes.

The current `work/score_p1b_completed.py` is LasHeR-specific. Do not use it
for RGBT210/234. The separate `work/score_p1b_rgbt_completed.py` requires
its own exact source review before use. It must verify exact
native/source/input/config/precision/GT/trace/full counts and official
host-specific scoring provenance. Existing score outputs must be
authenticated rather than rescored. FPS includes decode/init under
concurrent load; it does not establish isolated speed.

The archive/scoring CLIs select the exact SOT234 R2 folder for that pair
and R1 folders for the first three evaluations. Their source review must
cover that selection; never consume the failed SOT234 R1 plan.

Engineering reviews remain UNATTESTED/same-family/provisional. Launch
and benchmark completion do not meet the research goal by themselves.

## Actual Admission And Launch

At09:16UTC the first three full jobs have actual native RUNNING children:
P25GPU2 wrapper/child2085914/2085916 (210), P25GPU3
2085913/2085915 (234), P26GPU1 2611231/2611232 (210).
Their TRAIN smokes are COMPLETE/0 and raw smoke/full-start witnesses exist.
Do not repeat deployment, smoke, full launch or those archives.

SOT234 R1 SOURCE_FAIL is preserved. Its separate R2 review is SOURCE_PASS/0,
SHA256 `06dad9c10c98271ed1373310f767afa29736947ae0b34cc3349693456217bd5b`.
It remains disabled until actual SOT210 COMPLETE/0 and fresh runtime checks.
Then invoke the separate R2 operator's deploy, smoke and full actions,
verifying actual smoke completion and each launch before continuing.

RGBT scorer/archive source review is SOURCE_PASS/0, SHA256
`6dd632e5873edfd47eb8d5a5870447b50b3ef4b1355bbfca3d57df98df3c5cc2`.
Scorer SHA256:
`2d2e97fc97616d2c3a1ae85ce7fcf2dce16338c9322749190f27346f76f5d4da`.
Only after full terminal checks invoke
`work/score_p1b_rgbt_completed.py <branch> <benchmark>` once. Expected
archives: `reports/p1b_rgbt_completed_scores_r1_20261104/<branch>/<benchmark>/`.
Do not use it as a periodic monitor or rescore existing metrics.
