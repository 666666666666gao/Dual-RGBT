# C2 Evaluation Plan Preparation

`work/prepare_c2_policy_evaluation.py` is a local-only disabled-plan
preparer. Initial focused tests: 19 passed on 2026-11-04; the combined
evaluation/loader/remote-wrapper suite passed172 tests. R1 source review
returned SOURCE_FAIL for one actual schema defect: generated plans omitted
the remote scorer's mandatory plan_path field. The minimal correction adds
that field before binding and a real prepare_request JSON roundtrip test.
Corrected focused tests:20 passed. R2 source review returned SOURCE_PASS/0;
both reviewed source hashes were rechecked locally at17:33UTC. Report:
`reports/P2_C2_POLICY_EVALUATION_PREPARER_SOURCE_REVIEW_R2_20261104.json`,
SHA256 `4a105db5679d4f004efc48eac4b2c2456f4ac2ccc75caefb6f247d61a959175b`.
Attribution is UNATTESTED/same-family/provisional, execution bindings empty
and scientific acceptance false. R1 FAIL is preserved. This source-only
PASS admits no concrete evaluation plan or GPU task.

The preparer consumes eight archived FIT evidence documents and their
remote paths, plus the actual TRAIN-smoke/LasHeR manifests and scoring
protocol. It rejects RUNNING FIT, nonzero native exit, incomplete
five-epoch/890-update receipts, altered pinned documents, wrong replay
acceptance and incomplete LasHeR manifests. It preserves the FIT's
host-specific dependency hashes and adds the separate C2 evaluation
loader/entry/scorer sources. It emits only disabled TRAIN11 and full
LasHeR plans, fixed final epoch005/cost0.05, and a preparation receipt.

This is metadata preparation, not actual checkpoint/source verification,
CPU reconstruction, runtime capacity admission, deployment or inference.
The reviewed C2 loader must authenticate all retained checkpoint bytes
and original C2 state, and the exact new plans require separate review,
actual CPU reconstruction, own native COMPLETE0 TRAIN smoke and fresh
runtime admission before full LasHeR. No plan was generated from the
currently RUNNING FIT. No active root or recipe was changed.

At 17:23:40 UTC, P25 C2-zero FIT remained RUNNING at epoch5/global825
of890; GPU1 was 71 C/249.26 W under250 W. P25 FiLM replay R3 and P26
zero R6/FiLM R1 remained RUNNING without terminal acceptance. P25 GPU0/3
and P26 GPU0/1 remain available for tasks only after their own gates.
This observation is not a completion milestone or a new benchmark result.

Follow-up order remains: complete matching zero/FiLM loops; full
LasHeR-first evaluation; separately reviewed TRAIN-only diagnostics of
candidate visibility, localization, policy choice and state update. The
FiLM preview functional-drift concern needs fixed-input measurements;
frozen executor parameter equality alone is insufficient. Off-crop
clipping is source-confirmed but its frequency and impact are unmeasured.
Raw/Hann mismatch is not established. Do not change the active recipe,
start LoRA/token variants or tune using exploratory TEST results.

No routine HANDOFF commit is warranted by this nonterminal observation.
On actual terminal completion, archive raw evidence once and synchronize
both handoffs/GitHub using only the newly added HANDOFF section.
