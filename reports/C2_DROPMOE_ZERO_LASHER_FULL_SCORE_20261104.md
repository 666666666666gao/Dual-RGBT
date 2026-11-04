# C2 DropMoE Zero Full LasHeR Score

On2026-11-04 the fixed epoch005/cost0.05 C2 policy completed its first
full LasHeR evaluation and was scored once with the unchanged registered
official toolkit on the dedicated CPU interpreter. All targets remain unmet.

| Metric | Full TEST | Target | Shortfall (pp) |
| --- | ---: | ---: | ---: |
| PR | 75.01403930612416 | 75.8 | 0.7859606938758361 |
| NPR | 70.9941943042925 | 71.9 | 0.9058056957074996 |
| SR | 59.76717315301757 | 60.2 | 0.43282684698242946 |

These are exploratory full TEST observations with test_eligible=false.
No dev/TEST tuning, threshold, epoch, branch or checkpoint selection was
performed. This result neither establishes benchmark success nor changes
the admitted training recipe.

## Terminal And Scoring Gates

Task `eval_c2_policy_dropmoe_zero_e005_p25_r1_20261104_lasher` is native
COMPLETE/exit0 at22:36:07.237638UTC. Wrapper2511436/child2511437 exited
and GPU0 was released. Scope is exactly245sequences/220703frames.
Root:`/data2/gaob/Dual-RGBT/c2_policy_evaluation_dropmoe_zero_r1_20261104`.
Run:`/data2/gaob/Dual-RGBT/c2_policy_eval_runs/eval_c2_policy_dropmoe_zero_e005_p25_r1_20261104_lasher`.

The pinned C2 scorer authenticated117original-deployed sources, exact
plan/config/review/native argv/GPU, eight FIT evidence documents, fixed
epoch005 hash, full manifest/progress/prediction/GT/trace frame coverage
and positive finite decode-inclusive timings. The unchanged official
toolkit and protocol hashes matched. The receipt confirms unchanged
model state, zero optimizer/backward calls and allfalseprecision. No old
P1B model loader, repeated CPU reconstruction or repeated smoke was used.

Both score.json and score.npz were absent before scoring. The terminal
gate passed at22:37:58.521284UTC, then the exact pinned scorer exited0
using `/home/gaob/Dual-RGBT/conda-envs/cpu-scoring/bin/python`, with
CUDA_VISIBLE_DEVICES empty and CPU thread counts1. Metric computation
ran once. Later archive checks authenticated the existing curves and all
740unchanged run artifact hashes without executing metrics again.

Service timing:14.197404979906878FPS, mean70.43540713357598ms,
p95=100.20529610919766ms. Decode and initialization are included, model
loading excluded. This is concurrent system service time, not isolated
throughput or a fair speed comparison. Total parameters103374215.

## Evidence

One completion archive is
`reports/c2_policy_evaluation_r1_20261104/dropmoe/zero/completion_witness/`.
It contains15raw/derived evidence files plus observation.json; all16
SHA256SUMS entries and copied bytes were verified at22:39:17UTC. The
original remote predictions, GT, traces and timing evidence remain intact.

- Fixed epoch005 SHA256:`5d1f33e09e9dd28b076bc444d8156340fc2318f2f4d10b3b4dbcfd6d60d40d04`.
- Runtime plan:`67bedd73ac9ef53231d9095dfd7053798cf0cd55887f0ea485c11064e3ea3f82`.
- Native terminal:`a1784cc9bf175b620e3961f437cd7e71b1821237d58baad32940d8f520d3b553`.
- Score JSON:`03adf6ae49de51663630d7a5358e819ab2991a5827c0f7443179ec0636233623`.
- Curves:`f3fd89b6ddeb17cea6f234c947192f223a6102bd0c96fda09b9d768ff6b9296b`.
- Protocol:`899c14feddceaead8973067e5fe34fdc84284804e6ea389c92bf84d77ad932d2`.
- Scorer:`f98e22b04d3cc417b070eb84f41920d6d4ed65d4d545af968c905760e429411e`.
- Exact operation review:`40ffab8f8b0083e74035738da5d97f790ab74645040e5eb308f99ecc67ed8ef4`.

Review attribution remains UNATTESTED/same-family/provisional. Preserve
all prior FIT/replay/start/first25 archives; do not repeat scoring,
collection, FIT, deployment or this completion archive.

At22:36UTC P26 SOT-zero FIT remained nativeRUNNING with125flushed
finite epoch1updates and no weights; both FiLM replay verifiers remained
live with exact argv/CUDA ownership and no acceptance receipts.

The LasHeR-first completion/scoring prerequisite is now satisfied. No
RGBT210/234 task was launched: those exact C2 plans still require their
own separate review and fresh runtime admission. IdleGPU0 grants no
execution admission. Do not select a different checkpoint or branch from
this TEST result. Continue the existing admitted tasks.
