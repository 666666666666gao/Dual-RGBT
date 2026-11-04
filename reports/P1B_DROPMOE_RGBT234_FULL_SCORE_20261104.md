# Fixed P1B DropMoE RGBT234 Full Score

Native task `eval_p1b_e005_rgbt234_dropmoe_p25_r1_20261104` completed
with exit0 at 2026-11-04 10:58:48.790085 UTC. Full coverage is234 sequences/
116650 frames. Fixed policy epoch005 SHA256:
`6f29659e3f03b5d972296059d3c2d2f00ca93cb743e155b7ad5a71a53e63f0f4`;
cost0.05, unchanged model state, zero optimizer/backward calls and all-false
precision. Wrapper2085913/child2085915 exited and GPU3 was released.

The reviewed official CPU scorer/archive helper ran ONCE after exact native/
source/input/config/GT/trace/coverage checks. Nine files are preserved under
`reports/p1b_rgbt_completed_scores_r1_20261104/dropmoe/rgbt234/`.
Do not repeat the helper or scorer. The archive audit binds eight raw SHA256
values, including the score, curves, review and original scoring log.

| Fixed Configuration | MPR | MSR |
|---|---:|---:|
| P1A epoch20 before policy refresh |89.27325894072766|66.17343101331666|
| P1B policy epoch005 |90.12152972185736|67.06223794662317|
| P1B minus P1A, percentage points |+0.848270781129699|+0.888806933306512|
| Corresponding stronger ep20 target |92.8|67.7|
| Remaining gap, percentage points |2.67847027814264|0.63776205337683|

Both RGBT234 metrics improved over fixed P1A. RGBT210 previously showed
PR-0.04544/SR+0.22250 points, and LasHeR is still running; these results do
not establish consistent improvement across benchmarks or attainment of the
ep20 comparison target. The lower SOT target is not the ep20 success criterion.
This is exploratory TEST evidence, test_eligible=false; no threshold, branch,
epoch, recipe or architecture is selected from these scores.

RTX3090/search256/template128/two templates, total103374215 parameters.
Observed FPS18.820725322567043/mean53.13291506363718ms/
P9573.4098031418398ms include decode/init, exclude model loading and reflect
concurrent load. They are not isolated throughput or a speed advantage.

Separate local archive evidence audit passed: eight of eight archived raw
hashes and234 sequences/116650 frames match. Reviewer did not rescore or
reread the remote original artifact tree. Attribution is UNATTESTED/
same-family/provisional; scientific acceptance remains false. No statistical
significance or goal completion is established. Goal remains active/unmet.
