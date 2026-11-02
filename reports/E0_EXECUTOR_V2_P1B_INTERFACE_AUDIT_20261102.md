# P1B Interface Audit

This is an implementation preparation note, not source admission or evidence
of training. No GPU command is authorized by this note.

## Verified Starting Point

P1A writes a direct `DualRGBT` state dictionary with `backbone`, `decode`,
`box_head`, `preview`, `quality`, and `gain` modules. Its config explicitly
requires policy refresh before full E0 reporting. The formal recipe retains
epochs 16-20 and writes `COMPLETE.json` only after all 50,000 updates.
At 2026-11-02 23:17 UTC, both native trainer children were alive: DropMoE
38,200/50,000 steps and SOT 43,850/50,000.

## Existing Entry Points and Gaps

| Entry | Existing behavior | P1B gap |
|---|---|---|
| `scripts/train_policy.py` | Direct E0; trains preview/quality/gain; generates both primary and four-action labels | Old `LasHeRPairs`, zero history, labels called without Hann window; factory path implicit; one LR; latest-only saving |
| `scripts/train_trajectory_policy.py` | Direct E0; pre-decision templates/history/crop; causal predicted state; both primary/four-action labels with Hann | Freezes preview; quality/gain only; latest-only saving; historical GPU0/1 restriction; no P1A native completion binding |
| `scripts/train_candidate_joint_quality_trajectory.py` and gainpair variant | Candidate E1; `executor.preview`, `executor.quality`, `query_gain`; candidate geometry/rank | Different model, parameter namespace and output contract; cannot accept the P1A direct-E0 checkpoint as the same architecture |
| `scripts/collect_e0_reliability_train.py` | Direct E0 observation for M1; exact historical parent SHA, cost0 | Historical fixed parent and M1 label schema; not preview/quality/gain policy-refresh data |
| `scripts/collect_e0_current_full.py` | Replay and descriptor supplement of existing original trajectories/caches | Requires old per-sequence caches and exact original prediction equality; cannot label a newly trained executor by changing its checkpoint alone |
| `scripts/collect_policy_query_features.py` | Cached gain overlay refresh on candidate E1 | Requires cached-query overlay and E1 schema, fixed cost0.15; not direct E0 |

## Implementation Direction Requiring Review

Preserve direct E0 and the STOP plus four existing actions. Reuse the direct
E0 labeling and causal state logic, not the candidate E1 network. A new,
versioned P1B path must explicitly bind a completed P1A checkpoint, factory,
source snapshot, TRAIN manifest and native completion evidence.

The plan asks to refresh preview/quality/gain, with preview LR 1e-5 and heads
1e-4, while freezing backbone/decode/box_head including their buffers.
Collection must expose both modality primary outcomes and all four action
outcomes at declared TRAIN states. GT may label outcomes after the executed
decision; it must not control subsequent tracker state or template updates.
Counterfactual evaluation must leave the actual trajectory unchanged.

The plan's reference to Joint-Quality and K8 does not directly specify a
direct-E0 implementation: E0 has four fixed action candidates rather than
the candidate E1 pool. An independent review should identify the minimal
architecture-preserving objective and whether to separate frozen-policy
collection from offline head fitting before causal refresh. Do not silently
launch the candidate E1 trainer as P1B.

Before launch, the new sources need CPU checks, exact source/input review,
and real TRAIN smoke with nonzero updates in all three policy modules,
unchanged executor tensors/buffers, finite losses and strict reload. Full
collection/training must consume complete declared TRAIN shards with no dev
or TEST data, preserve final-five policy checkpoints, and publish terminal
receipts. No TEST-based checkpoint or hyperparameter selection is permitted.

The missing original ep20 initialization on P26 is not, by itself, proof that
P1B cannot run there. P1B host admission must determine the exact factory and
completed P1A model inputs needed, transfer only those after binding, and
check capacity. Existing P25 P1A jobs must continue unchanged.
