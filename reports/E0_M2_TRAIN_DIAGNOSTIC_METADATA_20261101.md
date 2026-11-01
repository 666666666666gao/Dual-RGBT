# M2 TRAIN Diagnostics From Existing Accepted Metadata

## Bound Evidence

Read on host26 with CUDA_VISIBLE_DEVICES empty, without a model forward,
optimizer, collection job or TEST annotation. The full collection acceptance
SHA256 is74eb5104ee223c8fa4eec4992c47f1e466efc6f27a9679a95d372fdd521f2522.
Its collection-plan hash and sequence-disjoint fit/holdout membership were
checked before aggregating trajectory rows. The existing best checkpoint
SHA256 is7e7df5d25d3e8667fe38a4ca8f6ec3662308ffcb52a09bd16aecaab59a9cb4be;
torch.load(map_location='cpu') read saved epoch9 metrics and confirmed
torch.cuda.is_initialized()=False.

The checkpoint binds training plan
6764d31dc8345a519cc66f4be5a6a728c9ee0c2ae77365ddb42604329e9381e8
and the same full collection acceptance. It was selected by component
holdout positive top1, not TEST scores. Upstream E0 saw this holdout.

## Complete Candidate Availability

| Partition | Sequences | Frames | Candidate sets | Positive candidate | No-match | Ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fit | 706 | 362360 | 14762 | 12281 | 793 | 1688 |
| Component holdout | 176 | 97075 | 3949 | 3318 | 132 | 499 |
| All TRAIN | 882 | 459435 | 18711 | 15599 | 925 | 2187 |

These totals come from every accepted trajectory row, unlike the earlier
110-sequence partial snapshot. They do not stratify E0 reliability/lost state.
Presence of a positive candidate is not learned accuracy or online recovery.

## Saved Epoch9 Head Metrics

| Partition | Supervised sets | Positive top1 | Correct candidate picks / available | Correct no-match / no-match | Ignored picks | Negative picks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fit | 13074 | 82.338993% | 10238 / 12281 | 527 / 793 | 689 | 1620 |
| Component holdout | 3450 | 79.623188% | 2685 / 3318 | 62 / 132 | 148 | 555 |

On the component holdout only62/132 (46.969697%) true no-match sets were
recognized, versus2685/3318 (80.922242%) correct candidate picks when a
positive candidate existed. Thus overall79.62% positive top1 conceals a
much weaker rejection decision. There are703 nonpositive picks among3450
supervised holdout sets, including148 ignored labels and555 known negatives.
These are frozen-cache classification counts, not closed-loop failures.

## What Still Requires The Original Cache

Host26 has the accepted plan, acceptance and best checkpoint but none of the
882 NPZ paths bound by acceptance.inputs_sha256. Do not regenerate the
collection solely to recover these files. Original accepted caches were
collected on25; no job or GPU was started there during this audit.

Reliable/uncertain/lost stratification and selected-box overlap changes
require the original predictions, candidate boxes, sampled frames and head
features. No-match confusion metadata alone cannot identify false replacement
of a currently correct target. For a future TRAIN replay, keep parent/M2
weights frozen, preserve per-sequence partitions, and report both false
replacement and recovery. A frozen replay cannot establish subsequent
closed-loop gains after state divergence.

This diagnosis supports investigating rejection and the missing M1 failure
gate; it does not specify a TEST-fitted threshold or authorize describing
E0+M2 as the full M1/M2/M3 architecture. R15 LasHeR/RGBT234 sources and plans
remain unchanged while their full exploratory inference continues.
