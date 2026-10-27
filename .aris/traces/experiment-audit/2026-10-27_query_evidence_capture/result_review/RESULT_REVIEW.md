# Query evidence capture result audit

**RESULT_PASS; blocking_issues = 0.**

Six paired trajectories agree exactly for 66 saved boxes, 60 routes, six final histories and six recorded final-template SHA256 pairs. The 60 evidence rows contain 50 STOP and 10 QUERY frames. Every STOP primary box equals its final box at float32 byte level; all 10 QUERY primary boxes differ from final boxes. Token-cost and float32-utility errors are zero.

Native task completed with exit 0 on server25 GPU0. Command, plan/checkpoint hashes, log summary and every bound source hash match. Exact result SHA256: `e70542a2fc764e9781d3c5d54626734830f3aac11e6ad9f0c61cd6558e5d58d3`.

Scope: fixed historical B6250, three real TRAIN prefixes, eleven frames each, costs 0 and 100. QUERY coverage is action 1 on bluegirl only. Templates were compared through stored SHA256 values; raw tensors were not retained. No new full-model accuracy or complete formal evaluator runtime is asserted.

Requested Astra/max; actual model/effort UNATTESTED; same-family/provisional. Reviewer performed local standard-library checks only.
