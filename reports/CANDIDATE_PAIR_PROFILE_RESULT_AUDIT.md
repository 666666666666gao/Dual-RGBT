# Candidate pair runtime-profile result audit

**WARN; 0 blocking issues for the limited descriptive profile.** The independent CPU checker passed all 336 checks with native exit 0. The captured controlled pair run supports higher observed aggregate B-shaped update throughput on GPU1 in this campaign. It does not establish whole-project completion-time speedup, a causal GPU0 slowdown, final inference cost, tracking accuracy, or three-module effectiveness.

Fresh Codex reviewer: `/root/audit_candidate_pair_actual_20261015`; requested `gpt-6-astra` / `max`, actual routing and effort **UNATTESTED**. `review_independence: same-family`; `acceptance_status: provisional`. This reviewer used only local standard-library parsing, hashes and arithmetic: zero model imports, NN/GPU/SSH calls or current training-status queries. Source, plans, summaries and original private artifacts were left unchanged.

## Counts and timing

All counts below were recomputed from preserved per-update completion timestamps using `[start, end)`. Rates use their own interval durations.

| Window | Seconds | Primary updates | Secondary updates | Total B updates/s |
| --- | ---: | ---: | ---: | ---: |
| Serial before | 181.483082056 | 173 | 0 | 0.953256899 |
| Whole parallel, secondary startup included | 242.148848295 | 212 | 168 | 1.569282706 |
| Serial after | 181.666357994 | 163 | 0 | 0.897249231 |
| Pooled serial | 363.149440050 | 336 | 0 | 0.925238932 |
| Strict measured overlap | 180.062206984 | 156 | 164 | 1.777163600 |

The whole-parallel ratio is **1.696083738402x** pooled serial; strict-overlap ratio is **1.920761804298x**. The whole parallel interval contains 212 primary completions and 168 secondary completions, including the secondary's four M0 updates. Strict overlap contains 156+164 measured completions and excludes those four M0 updates. Whole-parallel and strict-overlap rates must not be added to rates measured over different durations.

Primary measurement contains 550 updates over 606.004887104 seconds, plus four M0 updates (554 ledger rows). It was deliberately stopped after serial-after, before the 660-second ceiling. Secondary measurement contains 164 updates over 180.062206984 seconds, plus four M0 updates (168 rows), and ends on its 180-second timer. One primary completion falls in the transition between serial-before and parallel, and one follows serial-after; both are correctly excluded from the three comparison windows. Their absence from the phase totals is not a missing-row defect.

The primary's native start-to-measurement interval is 35.811369896 seconds. For the secondary it is 40.539081097 seconds; measurement-to-native-exit is 9.508415222 seconds, and the measured native terminal overhang before the parallel phase closes is 11.991848707 seconds. Whole-parallel exceeds strict overlap by 62.086641312 seconds. The primary is already running throughout the comparison; the whole-parallel denominator includes only secondary startup/M0/reload/termination and polling. It is not an equal-fixed-work end-to-end job comparison. Evidence: `scripts/profile_candidate_pair.py:70`, `:77`, `:82`, `:87`, `:95`; private controller `:56`, `:98`, `:104`; public summary `:4`, `:36`, `:68`, `:101`.

## Native execution, state checks and context membership

Both captured native wrapper records say `COMPLETE`, exit 0, physical GPU1, with commands bound to the exact role plans. The wrapper propagates the child return code (`scripts/run_task.py:28`). Recorded SSH and SCP transports also exit 0 with empty stderr; captured stdout parses to exactly the retained actual JSON. Source archive members are byte-identical to the reviewed local files; both rendered-controller and helper hashes match the source review.

Each actor has four M0 updates and a captured SANITY PASS. The called code checks finite loss/backward/gradient norm and AdamW; M0 checks nonzero finite gradients for each trainable prefix,257 frozen state tensors, exactly eight changed quality/query-gain tensors, optimizer step 4 and strict clone reload equality over 12 probe fields. The probe is the first example of the fourth M0 batch. Final actor execution rechecks 257 frozen tensors and optimizer step 554/168. Strict reload occurs at M0 only; no final post-measurement reload or causal-C probe is established. The auditor did not load weights or independently recalculate tensor equality. Evidence: `scripts/train_global_evidence_policy.py:40`, `scripts/train_candidate_evidence_policy.py:157`, `scripts/profile_candidate_pair.py:75`, `:90`.

Actor and wrapper identities are distinct. Primary child identity is bound to the captured wrapper child, and its measurement interval spans all comparison windows. Secondary measurement is inside both the primary lifetime and the wider parallel phase. All 12 serial-before and 12 serial-after GPU1 samples contain exactly the primary context. All 12 strict-overlap samples contain exactly the two expected actor contexts. Across all 16 wider parallel samples there is no third context:14 samples show two contexts and 2 show the primary only. This is sampled membership evidence at roughly 15-second intervals; it does not prove continuous occupancy between samples. Process IDs and commands remain private. Evidence: private actual JSON `:3`, `:137`, `:311`; private controller `:96`, `:109`.

## Provenance and GT classification

All 39 current source hashes match both role plans and their reviews. The original 37-file source set is an unchanged subset, as is the older 38-file profiler set. The recorded executed gates check source and four input hashes before/after actors, and the helper checks source 37 again on completion. The protected original checkpoint has a preload hash gate. Factory and fitted-preview checkpoints, dataset manifest and protected checkpoint are absent locally; no independent remote-byte rehash is claimed.

The workload reads manifest train GT and earlier train templates. It generates offline GT-jittered search states with zero history, executes primary/query predictions and computes IoUs against dataset boxes plus signed query-minus-primary IoU. It is **real_gt / runtime_training_profile**, not a tracking-metric evaluation. The relevant functions are called from the measured update path; no fabricated reference or self-output normalization is used. Label-coordinate bytes were not collected for this result audit, so label-row quantities are source-implied, not independently recounted. Evidence: `dual_rgbt_global/training.py:18`, `scripts/train_candidate_evidence_policy.py:50`, `scripts/profile_candidate_concurrency.py:25`.

Raw timing CSVs are not present locally. Their string-valued cells are preserved in captured stdout/actual JSON. Re-encoding those cells with the exact two-column CRLF `csv.DictWriter` format reproduces both COMPLETE digests:

| Actor | Rows | COMPLETE digest = reconstructed CSV SHA-256 |
| --- | ---: | --- |
| Primary | 554 | `080081fd1d8e1c5526fb894dc25ad2fd00c242a7944df40895ac857427965390` |
| Secondary | 168 | `93d1c11883ac9bfe152d3f9015ec95554aa0ea0403c257c37121c79df8c9b59c` |

The executed helper checks raw remote CSV SHA against COMPLETE before parsing it, then checks contiguous steps and row totals. This confirms content/digest consistency. Native status, COMPLETE and SANITY are also parsed objects inside the capture, not separately acquired raw files with their own byte receipts. Canonical object digests are identified as such in the JSON audit; they are not presented as hashes of standalone remote files. This custody limit remains WARN. Evidence: private controller `:87`, `:91`; actor `:100`.

## GPU0, resources and claim limits

GPU0 logged deltas are 213,275 and 275 for the three phases. Pooled serial gives 488 /363.149440050 =1.343799401 coarse updates/s; wider parallel gives 275 /242.148848295 =1.135665116, ratio 0.845115064. This apparent 15.49% reduction is not an identified causal slowdown.

The audited input summary says “up to 24” updates of uncertainty. **24 is only nominal 25-step flush-phase scale, not a hard wall-clock error bound.** GPU0 performs the optimizer update before row creation/flush; counter reads occur after the recorded phase timestamps. A read can intersect logging, and the source also flushes at sequence boundaries. Neither raw counter endpoints nor per-update GPU0 completion timestamps were retained. Even if one provisionally applies only ±24 to parallel and ±48 to the two pooled serial phases, the illustrative ratio range is 0.702282591 to 1.019110985. That is neither a rigorous bound nor a confidence interval. Evidence: `scripts/train_trajectory_policy.py:120`, `:127`, `:157`; private controller `:58`, `:67`.

GPU0 traverses changing sequences, image sizes, valid-label positions and tracked frames, while B actors use batch 8 offline sampling with two modalities. E0/C trajectory update units and B batch update units must not be summed. The B serial rates themselves differ (0.953256899 before and 0.897249231 after). Primary strict-overlap rate is 0.866367255, or 0.936371380 of pooled serial, while aggregate B throughput rises. Different sampling positions, I/O/cache state and shared CPU work remain confounds. Seeds 2027 and 2028 create overlapping `seed+index` data streams, not disjoint-data repetitions. There is one campaign and no randomized repeat or uncertainty estimate.

GPU0 sampled memory is 2295 MiB throughout. GPU1 maximum sampled usage is 2835 MiB in serial windows and 5656 MiB in parallel; all 12 strict-overlap samples report 5656 MiB. Mean GPU1 utilization is 16.5%,31.9375% and 29.9167% across the full three phases; strict-overlap mean is 37.6667%. CPU busy reports 43.5303%,50.9903% and 43.3985%. GPU0/1 are the only sampled cards; CPU load is aggregate. Raw `/proc/stat` deltas were not captured, so CPU busy values are reported and range-checked rather than independently recalculated. CUDA per-actor peaks are 1,008,155,136 allocated and 1,193,279,488 reserved bytes, including M0 clone/reload lifetime. Utilization is not SM occupancy or final inference cost. The earlier three-sample resource snapshot also proves no concurrency speedup.

The old attempt remains **INVALID_FOR_CONCURRENCY_COMPARISON**: formal B's captured native exit precedes the old disposable profiler start by 84.359455109 seconds, with zero original GPU1 counter growth in the old parallel interval. The old profiler's 168 measured updates and 180.965318203 seconds establish only its own completion. The old artifact and its original status are preserved.

## Audit evidence and disposition

The JSON report contains 74 exact input SHA-256 values, all 39 source hashes, native/object/ledger provenance, numeric recomputation and six explicit warnings. Important identities:

| Artifact | SHA-256 |
| --- | --- |
| Audited pair public summary | `ac4bce85f1f15c72b028c39230a67e440a9baf4aa4016a9c1b24aae02060d250` |
| Private pair actual JSON | `21c2094f7d1ec8cfca39d690fa00c54a9c276f63563eb594b372db0d8f9ed305` |
| Private captured pair stdout | `9e24d34f650a85a8c65be5979b508aa99ee252bf4644cdde01cb709fb84d7af5` |
| Executed rendered controller | `2aa1036d689472bd1870309d6355d52658c0b394db7adc1dabd65a890b34a984` |
| Reviewed pair helper | `333fc21916b043762c03467e89740febd66b8b1c2ad4a38bb9aeb7508c734900` |
| Pair actor source | `2d66ea0f6aebe397af85dcf2f24ab3917b38c44c9666350c9aecea2b0d180f17` |
| Primary plan | `9fda3592f3e9c0b9c6a5658cce893b7b7f0a97ccdf85aa7b142cf75b86d68729` |
| Secondary plan | `c5c09dcb3b4357fc73ebad1476a4f93bd3db8bfc55ae7c60da268b66a44b41e1` |
| Old actual JSON | `a738b7a48b0ff5da1177775eb1ba39a451a298a9b775498b49e7d164025399c0` |

The private audit trace is `C:/Users/gb/.codex_tmp/dual_rgbt_pair_result_audit_20261015_61b2`. Initial checker source/output/exit 1 are preserved: it incorrectly presumed both outside-window primary completions occurred after the final phase. Direct timestamp inspection corrected that reviewer assumption; one is in a transition gap. An initial bare-Python launcher error was corrected by using the installed offline uv interpreter. No experiment input was changed to obtain the final 336-check PASS.

Retain the B-only descriptive ratios and the startup/strict-overlap distinction. Future human summaries should replace the “up to 24” hard-bound wording with nominal 25-step flush-scale uncertainty while preserving audited original hashes. The scope and custody WARNs do not invalidate the observed single-campaign B completion-rate comparison; they prohibit broader elapsed-time, accuracy or inference claims.
